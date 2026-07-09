from __future__ import annotations

import copy
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Any, Callable, Optional


MIC_SILENCE_RMS_THRESHOLD = 0.0001
DEFAULT_NOISE_FLOOR = 0.0008


def rms_to_meter_percent(rms: float) -> int:
    if rms <= MIC_SILENCE_RMS_THRESHOLD:
        return 0
    return max(0, min(100, int(rms * 5000)))


def microphone_health_message(block_count: int, peak_rms: float, chunk_count: int) -> str | None:
    if block_count <= 0:
        return (
            "No microphone audio was received. Check Windows microphone permission, make sure the microphone is not "
            "muted, or choose a different input device."
        )
    if peak_rms <= MIC_SILENCE_RMS_THRESHOLD:
        return (
            "Microphone input looks silent. Check Windows input volume/privacy settings, unmute the microphone, "
            "or choose a different input device."
        )
    if chunk_count <= 0:
        return "Audio was received, but no transcription chunk was produced. Record for longer or reduce chunk seconds."
    return None


def preferred_input_sample_rate(sd: Any, device: Optional[int], target_sample_rate: int) -> int:
    try:
        sd.check_input_settings(device=device, channels=1, samplerate=target_sample_rate, dtype="float32")
        return int(target_sample_rate)
    except Exception:
        pass
    try:
        device_info = sd.query_devices(device, "input")
        default_rate = int(float(device_info.get("default_samplerate", target_sample_rate)))
        return default_rate if default_rate > 0 else int(target_sample_rate)
    except Exception:
        return int(target_sample_rate)


def resample_audio(samples: Any, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    target_size = max(1, int(round(audio.size * (float(target_rate) / float(source_rate)))))
    source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        high_pass_hz: float = 80.0,
        dc_decay: float = 0.995,
        gate_ratio: float = 1.3,
        gate_attenuation: float = 0.35,
        noise_floor: float = DEFAULT_NOISE_FLOOR,
        limiter_level: float = 0.98,
    ):
        self.sample_rate = max(1, int(sample_rate))
        self.high_pass_hz = max(1.0, float(high_pass_hz))
        self.dc_decay = float(dc_decay)
        self.gate_ratio = float(gate_ratio)
        self.gate_attenuation = float(gate_attenuation)
        self.noise_floor = float(noise_floor)
        self.limiter_level = float(limiter_level)
        self._dc_prev_input = 0.0
        self._dc_prev_output = 0.0
        self._hp_prev_input = 0.0
        self._hp_prev_output = 0.0
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2.0 * pi * self.high_pass_hz)
        self._high_pass_alpha = rc / (rc + dt)

    def process(self, samples: Any) -> Any:
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)

        filtered = np.empty_like(audio)
        for index, value in enumerate(audio):
            sample = float(value)
            dc_blocked = sample - self._dc_prev_input + self.dc_decay * self._dc_prev_output
            self._dc_prev_input = sample
            self._dc_prev_output = dc_blocked

            high_passed = self._high_pass_alpha * (
                self._hp_prev_output + dc_blocked - self._hp_prev_input
            )
            self._hp_prev_input = dc_blocked
            self._hp_prev_output = high_passed
            filtered[index] = high_passed

        rms = float(np.sqrt(np.mean(np.square(filtered)))) if filtered.size else 0.0
        self._update_noise_floor(rms)
        filtered = self._apply_soft_gate(filtered, rms)
        filtered = self._apply_soft_limiter(filtered)
        return filtered.astype(np.float32, copy=False)

    def _update_noise_floor(self, rms: float) -> None:
        if rms <= 0.0:
            return
        quiet_ceiling = max(DEFAULT_NOISE_FLOOR * 2.0, self.noise_floor * 1.8)
        if rms <= quiet_ceiling:
            self.noise_floor = (self.noise_floor * 0.95) + (rms * 0.05)

    def _apply_soft_gate(self, samples: Any, rms: float) -> Any:
        threshold = max(DEFAULT_NOISE_FLOOR, self.noise_floor * self.gate_ratio)
        if rms >= threshold or threshold <= 0.0:
            return samples
        attenuation = max(self.gate_attenuation, min(1.0, rms / threshold))
        return samples * attenuation

    def _apply_soft_limiter(self, samples: Any) -> Any:
        import numpy as np

        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak <= self.limiter_level:
            return samples
        return np.tanh(samples / self.limiter_level).astype(np.float32) * self.limiter_level


def audio_chunk_has_activity(samples: Any, noise_floor: float = DEFAULT_NOISE_FLOOR) -> bool:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    threshold = max(MIC_SILENCE_RMS_THRESHOLD * 3.0, noise_floor * 1.1)
    return rms >= threshold or peak >= threshold * 2.5


def audio_chunk_duration_seconds(chunk: "AudioChunk") -> float:
    try:
        sample_count = len(chunk.samples)
    except TypeError:
        sample_count = 0
    if chunk.sample_rate <= 0:
        return 0.0
    return max(0.0, sample_count / float(chunk.sample_rate))


def _force_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@dataclass(frozen=True)
class EngineConfig:
    whisper_model_dir: str = r"C:\models\faster-whisper"
    output_file: str = "meeting_transcript.txt"
    sample_rate: int = 16_000
    chunk_seconds: float = 5.0
    overlap_seconds: float = 0.5
    compute_type: str = "int8"
    input_device: Optional[int] = None
    language: Optional[str] = "en"
    max_queue_chunks: int = 8

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.sample_rate <= 0:
            errors.append("Sample rate must be positive.")
        if self.chunk_seconds <= 0:
            errors.append("Chunk seconds must be positive.")
        if self.overlap_seconds < 0:
            errors.append("Overlap seconds cannot be negative.")
        if self.overlap_seconds >= self.chunk_seconds:
            errors.append("Overlap seconds must be smaller than chunk seconds.")
        whisper_path = self._validate_folder("Whisper model", self.whisper_model_dir, errors)
        if whisper_path and not (whisper_path / "model.bin").is_file():
            errors.append(
                "Whisper model is incomplete: expected model.bin in "
                f"{whisper_path}. Copy a CTranslate2 faster-whisper model folder into this path."
            )
        return errors

    @staticmethod
    def _validate_folder(label: str, raw_path: str, errors: list[str]) -> Path | None:
        if not raw_path:
            errors.append(f"{label} path is required.")
            return None
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"{label} path does not exist: {raw_path}")
            return None
        if not path.is_dir():
            errors.append(f"{label} path must be a folder: {raw_path}")
            return None
        return path


@dataclass
class AudioChunk:
    index: int
    start_time: float
    samples: Any
    sample_rate: int
    captured_at: float = field(default_factory=time.monotonic)
    is_final: bool = False


@dataclass
class TranscriptRow:
    id: str
    chunk_index: int
    start: float
    end: float
    text: str
    is_final: bool = False
    updated_at: float = field(default_factory=time.time)


class InputBlockBuffer:
    """Thread-safe queue fed by sounddevice callback; callback work stays tiny."""

    def __init__(self, max_blocks: int = 64):
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_blocks)

    def push_from_callback(self, block: Any) -> None:
        try:
            if hasattr(block, "shape") and hasattr(block, "copy"):
                copied = block.copy()
            else:
                copied = copy.deepcopy(block)
            self._queue.put_nowait(copied)
        except queue.Full:
            # Prefer dropping audio block over blocking PortAudio callback.
            return

    def pop(self, timeout: float | None = None) -> Any:
        return self._queue.get(timeout=timeout)


class TranscriptStore:
    def __init__(self):
        self.rows: list[TranscriptRow] = []
        self._lock = threading.RLock()
        self._next_row_id = 1

    def add_transcript(self, chunk_index: int, start: float, end: float, text: str) -> TranscriptRow:
        with self._lock:
            row = TranscriptRow(
                id=f"row_{self._next_row_id}",
                chunk_index=chunk_index,
                start=start,
                end=end,
                text=text.strip(),
            )
            self._next_row_id += 1
            self.rows.append(row)
            return row

    def snapshot(self) -> list[TranscriptRow]:
        with self._lock:
            return [copy.copy(row) for row in self.rows]


class DuplicateSuppressor:
    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = window_seconds
        self._seen: list[tuple[float, str]] = []

    def is_duplicate(self, timestamp: float, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return True
        self._seen = [
            (seen_at, seen_text)
            for seen_at, seen_text in self._seen
            if timestamp - seen_at <= self.window_seconds
        ]
        for seen_at, seen_text in self._seen:
            if abs(timestamp - seen_at) <= self.window_seconds and seen_text == normalized:
                return True
        self._seen.append((timestamp, normalized))
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()


def transcription_segment_is_usable(segment: Any, text: str) -> bool:
    if not text:
        return False
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    compression_ratio = getattr(segment, "compression_ratio", None)
    if no_speech_prob is not None and float(no_speech_prob) > 0.75:
        return False
    if avg_logprob is not None and float(avg_logprob) < -1.2:
        return False
    if compression_ratio is not None and float(compression_ratio) > 2.4:
        return False
    return True


class AtomicTranscriptWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.RLock()

    def refresh(self, rows: list[TranscriptRow]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    for row in sorted(rows, key=lambda item: (item.start, item.id)):
                        handle.write(f"[{format_timestamp(row.start)}] {row.text}\n")
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)


def format_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    remaining = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


EventHandler = Callable[[str, dict[str, Any]], None]


class MeetingTranscriberEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.store = TranscriptStore()
        self.writer = AtomicTranscriptWriter(config.output_file)
        self.duplicate_suppressor = DuplicateSuppressor(config.overlap_seconds + 0.5)
        self._event_handlers: list[EventHandler] = []
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._raw_chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue(config.max_queue_chunks)
        self._transcription_queue: queue.Queue[AudioChunk | None] = queue.Queue(config.max_queue_chunks)
        self._pending_transcription_chunks: set[int] = set()
        self._audio_block_count = 0
        self._audio_peak_rms = 0.0
        self._audio_chunk_count = 0
        self._state_lock = threading.RLock()

    def on_event(self, handler: EventHandler) -> None:
        self._event_handlers.append(handler)

    def start(self) -> None:
        _force_offline_mode()
        errors = self.config.validate()
        if errors:
            raise ValueError("\n".join(errors))
        self._stop_event.clear()
        self._pause_event.clear()
        with self._state_lock:
            self._audio_block_count = 0
            self._audio_peak_rms = 0.0
            self._audio_chunk_count = 0
        self._emit("status", {"message": "Loading models"})

        self._threads = [
            threading.Thread(target=self._recording_loop, name="recorder", daemon=True),
            threading.Thread(target=self._fanout_loop, name="chunk-fanout", daemon=True),
            threading.Thread(target=self._transcription_loop, name="transcription", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._emit("status", {"message": "Recording"})

    def stop(self) -> None:
        self._emit("status", {"message": "Stopping"})
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=30)
        self.writer.refresh(self.store.snapshot())
        with self._state_lock:
            diagnostic = microphone_health_message(
                self._audio_block_count,
                self._audio_peak_rms,
                self._audio_chunk_count,
            )
        if diagnostic and not self.store.snapshot():
            self._emit("error", {"message": diagnostic})
        self._emit("status", {"message": "Stopped"})

    def pause(self) -> None:
        self._pause_event.set()
        self._emit("status", {"message": "Paused"})

    def resume(self) -> None:
        self._pause_event.clear()
        self._emit("status", {"message": "Recording"})

    def _recording_loop(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Audio stack unavailable: {exc}"})
            self._raw_chunk_queue.put(None)
            return

        block_buffer = InputBlockBuffer()
        model_sample_rate = self.config.sample_rate
        input_sample_rate = preferred_input_sample_rate(sd, self.config.input_device, model_sample_rate)
        chunk_samples = int(self.config.chunk_seconds * input_sample_rate)
        overlap_samples = int(self.config.overlap_seconds * input_sample_rate)
        keep_samples = max(0, overlap_samples)
        preprocessor = AudioPreprocessor(input_sample_rate)
        accumulated = np.empty((0,), dtype=np.float32)
        chunk_index = 0
        stream_start = time.monotonic()

        def callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
            if status:
                self._emit("log", {"message": str(status)})
            if not self._pause_event.is_set():
                block_buffer.push_from_callback(indata)

        try:
            with sd.InputStream(
                samplerate=input_sample_rate,
                channels=1,
                dtype="float32",
                device=self.config.input_device,
                callback=callback,
            ):
                try:
                    device_info = sd.query_devices(self.config.input_device, "input")
                    device_name = device_info.get("name", self.config.input_device) if isinstance(device_info, dict) else self.config.input_device
                except Exception:
                    device_name = self.config.input_device if self.config.input_device is not None else "default input"
                self._emit("log", {"message": f"Microphone stream opened: {device_name}"})
                if input_sample_rate != model_sample_rate:
                    self._emit(
                        "log",
                        {
                            "message": (
                                f"Microphone sample rate: {input_sample_rate} Hz; "
                                f"resampling to {model_sample_rate} Hz for Whisper"
                            )
                        },
                    )
                while not self._stop_event.is_set():
                    try:
                        block = block_buffer.pop(timeout=0.2)
                    except queue.Empty:
                        continue
                    mono = np.asarray(block, dtype=np.float32).reshape(-1)
                    cleaned = preprocessor.process(mono)
                    level = float(np.sqrt(np.mean(np.square(cleaned)))) if cleaned.size else 0.0
                    with self._state_lock:
                        self._audio_block_count += 1
                        self._audio_peak_rms = max(self._audio_peak_rms, level)
                    self._emit("level", {"rms": level, "percent": rms_to_meter_percent(level)})
                    accumulated = np.concatenate([accumulated, cleaned])
                    while accumulated.size >= chunk_samples:
                        start_time = max(
                            0.0,
                            ((chunk_index * (chunk_samples - keep_samples)) / input_sample_rate),
                        )
                        samples = resample_audio(accumulated[:chunk_samples].copy(), input_sample_rate, model_sample_rate)
                        with self._state_lock:
                            self._audio_chunk_count += 1
                        if audio_chunk_has_activity(samples, preprocessor.noise_floor):
                            self._emit("log", {"message": f"Audio chunk {chunk_index} captured ({self.config.chunk_seconds:.1f}s)"})
                            self._raw_chunk_queue.put(
                                AudioChunk(
                                    chunk_index,
                                    start_time,
                                    samples,
                                    model_sample_rate,
                                    captured_at=time.monotonic(),
                                )
                            )
                        else:
                            self._emit("log", {"message": f"No speech detected in chunk {chunk_index}"})
                        chunk_index += 1
                        accumulated = accumulated[chunk_samples - keep_samples :]
                if accumulated.size > input_sample_rate // 2:
                    start_time = max(0.0, time.monotonic() - stream_start - (accumulated.size / input_sample_rate))
                    samples = resample_audio(accumulated.copy(), input_sample_rate, model_sample_rate)
                    with self._state_lock:
                        self._audio_chunk_count += 1
                    if audio_chunk_has_activity(samples, preprocessor.noise_floor):
                        self._emit("log", {"message": f"Final audio chunk {chunk_index} captured"})
                        self._raw_chunk_queue.put(
                            AudioChunk(
                                chunk_index,
                                start_time,
                                samples,
                                model_sample_rate,
                                captured_at=time.monotonic(),
                                is_final=True,
                            )
                        )
                    else:
                        self._emit("log", {"message": f"No speech detected in final chunk {chunk_index}"})
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Recording failed: {exc}"})
        finally:
            self._raw_chunk_queue.put(None)

    def _fanout_loop(self) -> None:
        while True:
            chunk = self._raw_chunk_queue.get()
            if chunk is None:
                self._transcription_queue.put(None)
                return
            dropped = self._drop_stale_transcription_chunks(max_queued=1)
            with self._state_lock:
                self._pending_transcription_chunks.add(chunk.index)
            self._transcription_queue.put(chunk)
            self._emit_transcription_lag(active=True)
            if dropped:
                noun = "chunk" if dropped == 1 else "chunks"
                self._emit("log", {"message": f"Catching up: skipped {dropped} stale audio {noun}"})

    def _drop_stale_transcription_chunks(self, max_queued: int = 2) -> int:
        dropped = 0
        while self._transcription_queue.qsize() > max_queued:
            try:
                stale = self._transcription_queue.get_nowait()
            except queue.Empty:
                break
            if stale is None:
                self._transcription_queue.put(None)
                break
            dropped += 1
            with self._state_lock:
                self._pending_transcription_chunks.discard(stale.index)
        return dropped

    def _queued_audio_seconds(self) -> float:
        total = 0.0
        with self._transcription_queue.mutex:
            queued_items = list(self._transcription_queue.queue)
        for item in queued_items:
            if isinstance(item, AudioChunk):
                total += audio_chunk_duration_seconds(item)
        return total

    def _emit_transcription_lag(self, active: bool, current_chunk: AudioChunk | None = None) -> None:
        behind_seconds = self._queued_audio_seconds()
        if current_chunk is not None:
            behind_seconds += audio_chunk_duration_seconds(current_chunk)
        self._emit("lag", {"behind_seconds": behind_seconds, "active": active and behind_seconds > 0})

    def _transcription_loop(self) -> None:
        try:
            self._emit("log", {"message": "Loading Whisper model..."})
            model = self._load_whisper_model()
            self._emit("log", {"message": "Whisper model ready"})
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Whisper model load failed: {exc}"})
            return

        while True:
            chunk = self._transcription_queue.get()
            if chunk is None:
                self._emit("lag", {"behind_seconds": 0.0, "active": False})
                return
            try:
                self._emit_transcription_lag(active=True, current_chunk=chunk)
                self._emit("log", {"message": f"Transcribing chunk {chunk.index}..."})
                started_at = time.monotonic()
                segment_count = 0
                segments, _info = model.transcribe(
                    chunk.samples,
                    language=self.config.language,
                    vad_filter=False,
                    word_timestamps=False,
                    beam_size=1,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )
                for segment in segments:
                    text = getattr(segment, "text", "").strip()
                    if not transcription_segment_is_usable(segment, text):
                        continue
                    start = chunk.start_time + float(getattr(segment, "start", 0.0))
                    end = chunk.start_time + float(getattr(segment, "end", start))
                    if self.duplicate_suppressor.is_duplicate(start, text):
                        continue
                    row = self.store.add_transcript(chunk.index, start, end, text)
                    segment_count += 1
                    self.writer.refresh(self.store.snapshot())
                    print(f"[{format_timestamp(row.start)}] {row.text}", flush=True)
                    self._emit("transcript", {"row": row})
                if segment_count == 0:
                    self._emit("log", {"message": f"No speech detected in chunk {chunk.index}"})
                elapsed = time.monotonic() - started_at
                behind_after = max(0.0, time.monotonic() - chunk.captured_at)
                self._emit(
                    "log",
                    {
                        "message": (
                            f"Chunk {chunk.index} transcribed in {elapsed:.1f}s; "
                            f"{behind_after:.1f}s behind realtime"
                        )
                    },
                )
            except Exception as exc:  # pragma: no cover - environment-dependent
                self._emit("error", {"message": f"Transcription failed for chunk {chunk.index}: {exc}"})
            finally:
                with self._state_lock:
                    self._pending_transcription_chunks.discard(chunk.index)
                self._emit_transcription_lag(active=True)

    def _load_whisper_model(self) -> Any:
        _force_offline_mode()
        from faster_whisper import WhisperModel

        if not Path(self.config.whisper_model_dir).exists():
            raise FileNotFoundError(self.config.whisper_model_dir)
        try:
            return WhisperModel(
                self.config.whisper_model_dir,
                device="cpu",
                compute_type=self.config.compute_type,
                local_files_only=True,
            )
        except TypeError:
            return WhisperModel(
                self.config.whisper_model_dir,
                device="cpu",
                compute_type=self.config.compute_type,
            )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        for handler in list(self._event_handlers):
            try:
                handler(event_type, payload)
            except Exception:
                continue


def list_input_devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    devices = []
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) > 0:
            hostapi_index = int(device.get("hostapi", -1))
            hostapi_name = ""
            if 0 <= hostapi_index < len(hostapis):
                hostapi_name = str(hostapis[hostapi_index].get("name", ""))
            devices.append(
                {
                    "index": index,
                    "name": device.get("name", f"Input {index}"),
                    "hostapi": hostapi_name,
                    "channels": device.get("max_input_channels", 0),
                    "default_samplerate": device.get("default_samplerate"),
                }
            )
    return devices
