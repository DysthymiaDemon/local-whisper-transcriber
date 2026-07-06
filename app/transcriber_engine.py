from __future__ import annotations

import copy
import math
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


UNKNOWN_SPEAKER_LABEL = "Speaker ?"


def _force_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_force_offline_mode()


@dataclass(frozen=True)
class EngineConfig:
    whisper_model_dir: str = r"C:\models\faster-whisper-small"
    pyannote_pipeline_dir: str = r"C:\models\pyannote-speaker-diarization"
    pyannote_embedding_model_dir: str = r"C:\models\pyannote-embedding"
    output_file: str = "meeting_transcript.txt"
    sample_rate: int = 16_000
    chunk_seconds: float = 8.0
    overlap_seconds: float = 2.0
    compute_type: str = "int8"
    speaker_match_threshold: float = 0.70
    min_speaker_confidence: float = 0.50
    min_overlap_ratio: float = 0.35
    input_device: Optional[int] = None
    language: Optional[str] = None
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
        for label, raw_path in (
            ("Whisper model", self.whisper_model_dir),
            ("Pyannote pipeline", self.pyannote_pipeline_dir),
            ("Pyannote embedding model", self.pyannote_embedding_model_dir),
        ):
            if not raw_path:
                errors.append(f"{label} path is required.")
            elif not Path(raw_path).exists():
                errors.append(f"{label} path does not exist: {raw_path}")
        return errors


@dataclass
class AudioChunk:
    index: int
    start_time: float
    samples: Any
    sample_rate: int
    is_final: bool = False


@dataclass
class TranscriptRow:
    id: str
    chunk_index: int
    start: float
    end: float
    text: str
    speaker_key: Optional[str] = None
    speaker_label: str = UNKNOWN_SPEAKER_LABEL
    is_final: bool = False
    updated_at: float = field(default_factory=time.time)


@dataclass
class DiarizationTurn:
    start: float
    end: float
    local_label: str
    embedding: Optional[list[float]] = None
    confidence: float = 1.0


@dataclass
class SpeakerIdentity:
    key: str
    display_name_value: str
    embedding: list[float]
    sample_count: int = 1


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


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize_embedding(embedding: Iterable[float] | None) -> Optional[list[float]]:
    if embedding is None:
        return None
    values = [float(value) for value in embedding]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return None
    return [value / norm for value in values]


class SpeakerRegistry:
    def __init__(self, match_threshold: float = 0.70, min_confidence: float = 0.50):
        self.match_threshold = match_threshold
        self.min_confidence = min_confidence
        self._lock = threading.RLock()
        self._identities: dict[str, SpeakerIdentity] = {}
        self._next_index = 1

    def assign(self, embedding: Iterable[float] | None, confidence: float = 1.0) -> Optional[SpeakerIdentity]:
        if confidence < self.min_confidence:
            return None
        normalized = _normalize_embedding(embedding)
        if normalized is None:
            return None

        with self._lock:
            best_identity: Optional[SpeakerIdentity] = None
            best_score = -1.0
            for identity in self._identities.values():
                score = cosine_similarity(identity.embedding, normalized)
                if score > best_score:
                    best_score = score
                    best_identity = identity

            if best_identity is not None and best_score >= self.match_threshold:
                total = best_identity.sample_count + 1
                averaged = [
                    ((old * best_identity.sample_count) + new) / total
                    for old, new in zip(best_identity.embedding, normalized)
                ]
                best_identity.embedding = _normalize_embedding(averaged) or best_identity.embedding
                best_identity.sample_count = total
                return best_identity

            key = f"speaker_{self._next_index}"
            identity = SpeakerIdentity(
                key=key,
                display_name_value=f"Speaker {self._next_index}",
                embedding=normalized,
            )
            self._identities[key] = identity
            self._next_index += 1
            return identity

    def display_name(self, speaker_key: str | None) -> str:
        if speaker_key is None:
            return UNKNOWN_SPEAKER_LABEL
        with self._lock:
            identity = self._identities.get(speaker_key)
            return identity.display_name_value if identity else UNKNOWN_SPEAKER_LABEL

    def rename(self, speaker_key: str, display_name: str) -> None:
        cleaned = display_name.strip()
        if not cleaned:
            return
        with self._lock:
            if speaker_key in self._identities:
                self._identities[speaker_key].display_name_value = cleaned

    def speakers(self) -> list[SpeakerIdentity]:
        with self._lock:
            return list(self._identities.values())


class TranscriptStore:
    def __init__(self, speaker_registry: SpeakerRegistry, min_overlap_ratio: float = 0.35):
        self.speaker_registry = speaker_registry
        self.min_overlap_ratio = min_overlap_ratio
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

    def apply_diarization(self, chunk_index: int, turns: list[DiarizationTurn]) -> list[TranscriptRow]:
        updates: list[TranscriptRow] = []
        with self._lock:
            for row in self.rows:
                if row.chunk_index != chunk_index:
                    continue
                best_turn, best_ratio = self._best_turn_for_row(row, turns)
                if best_turn is None or best_ratio < self.min_overlap_ratio:
                    continue
                identity = self.speaker_registry.assign(best_turn.embedding, best_turn.confidence)
                if identity is None:
                    continue
                new_label = self.speaker_registry.display_name(identity.key)
                if row.speaker_key != identity.key or row.speaker_label != new_label:
                    row.speaker_key = identity.key
                    row.speaker_label = new_label
                    row.updated_at = time.time()
                    updates.append(row)
        return updates

    def rename_speaker(self, speaker_key: str, display_name: str) -> list[TranscriptRow]:
        self.speaker_registry.rename(speaker_key, display_name)
        updated: list[TranscriptRow] = []
        with self._lock:
            new_name = self.speaker_registry.display_name(speaker_key)
            for row in self.rows:
                if row.speaker_key == speaker_key:
                    row.speaker_label = new_name
                    row.updated_at = time.time()
                    updated.append(row)
        return updated

    def snapshot(self) -> list[TranscriptRow]:
        with self._lock:
            return [copy.copy(row) for row in self.rows]

    @staticmethod
    def _best_turn_for_row(
        row: TranscriptRow, turns: list[DiarizationTurn]
    ) -> tuple[Optional[DiarizationTurn], float]:
        duration = max(row.end - row.start, 0.001)
        best_turn: Optional[DiarizationTurn] = None
        best_overlap = 0.0
        for turn in turns:
            overlap = max(0.0, min(row.end, turn.end) - max(row.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        return best_turn, best_overlap / duration


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
                        handle.write(
                            f"[{format_timestamp(row.start)}] "
                            f"{row.speaker_label}: {row.text}\n"
                        )
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
        self.speaker_registry = SpeakerRegistry(
            match_threshold=config.speaker_match_threshold,
            min_confidence=config.min_speaker_confidence,
        )
        self.store = TranscriptStore(self.speaker_registry, config.min_overlap_ratio)
        self.writer = AtomicTranscriptWriter(config.output_file)
        self.duplicate_suppressor = DuplicateSuppressor(config.overlap_seconds + 0.5)
        self._event_handlers: list[EventHandler] = []
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._raw_chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue(config.max_queue_chunks)
        self._transcription_queue: queue.Queue[AudioChunk | None] = queue.Queue(config.max_queue_chunks)
        self._diarization_queue: queue.Queue[AudioChunk | None] = queue.Queue(config.max_queue_chunks)
        self._pending_transcription_chunks: set[int] = set()
        self._pending_diarization_chunks: set[int] = set()
        self._state_lock = threading.RLock()

    def on_event(self, handler: EventHandler) -> None:
        self._event_handlers.append(handler)

    def start(self) -> None:
        errors = self.config.validate()
        if errors:
            raise ValueError("\n".join(errors))
        self._stop_event.clear()
        self._pause_event.clear()
        self._emit("status", {"message": "Loading models"})

        self._threads = [
            threading.Thread(target=self._recording_loop, name="recorder", daemon=True),
            threading.Thread(target=self._fanout_loop, name="chunk-fanout", daemon=True),
            threading.Thread(target=self._transcription_loop, name="transcription", daemon=True),
            threading.Thread(target=self._diarization_loop, name="diarization", daemon=True),
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
        self._emit("status", {"message": "Stopped"})

    def pause(self) -> None:
        self._pause_event.set()
        self._emit("status", {"message": "Paused"})

    def resume(self) -> None:
        self._pause_event.clear()
        self._emit("status", {"message": "Recording"})

    def rename_speaker(self, speaker_key: str, display_name: str) -> None:
        updated = self.store.rename_speaker(speaker_key, display_name)
        self.writer.refresh(self.store.snapshot())
        for row in updated:
            self._emit("speaker_update", {"row": row})
        self._emit("speakers", {"speakers": self.speaker_registry.speakers()})

    def lag_status(self) -> str:
        with self._state_lock:
            return f"Diarization: {len(self._pending_diarization_chunks)} chunks behind"

    def _recording_loop(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Audio stack unavailable: {exc}"})
            self._raw_chunk_queue.put(None)
            return

        block_buffer = InputBlockBuffer()
        sample_rate = self.config.sample_rate
        chunk_samples = int(self.config.chunk_seconds * sample_rate)
        overlap_samples = int(self.config.overlap_seconds * sample_rate)
        keep_samples = max(0, overlap_samples)
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
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.config.input_device,
                callback=callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        block = block_buffer.pop(timeout=0.2)
                    except queue.Empty:
                        continue
                    mono = np.asarray(block, dtype=np.float32).reshape(-1)
                    level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                    self._emit("level", {"rms": level})
                    accumulated = np.concatenate([accumulated, mono])
                    while accumulated.size >= chunk_samples:
                        start_time = max(
                            0.0,
                            ((chunk_index * (chunk_samples - keep_samples)) / sample_rate),
                        )
                        samples = accumulated[:chunk_samples].copy()
                        self._raw_chunk_queue.put(
                            AudioChunk(chunk_index, start_time, samples, sample_rate)
                        )
                        chunk_index += 1
                        accumulated = accumulated[chunk_samples - keep_samples :]
                if accumulated.size > sample_rate // 2:
                    start_time = max(0.0, time.monotonic() - stream_start - (accumulated.size / sample_rate))
                    self._raw_chunk_queue.put(
                        AudioChunk(chunk_index, start_time, accumulated.copy(), sample_rate, is_final=True)
                    )
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Recording failed: {exc}"})
        finally:
            self._raw_chunk_queue.put(None)

    def _fanout_loop(self) -> None:
        while True:
            chunk = self._raw_chunk_queue.get()
            if chunk is None:
                self._transcription_queue.put(None)
                self._diarization_queue.put(None)
                return
            with self._state_lock:
                self._pending_transcription_chunks.add(chunk.index)
                self._pending_diarization_chunks.add(chunk.index)
            self._emit("lag", {"message": self.lag_status()})
            self._transcription_queue.put(chunk)
            self._diarization_queue.put(chunk)

    def _transcription_loop(self) -> None:
        try:
            model = self._load_whisper_model()
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Whisper model load failed: {exc}"})
            return

        while True:
            chunk = self._transcription_queue.get()
            if chunk is None:
                return
            try:
                segments, _info = model.transcribe(
                    chunk.samples,
                    language=self.config.language,
                    vad_filter=True,
                    word_timestamps=True,
                    beam_size=1,
                )
                for segment in segments:
                    text = getattr(segment, "text", "").strip()
                    if not text:
                        continue
                    start = chunk.start_time + float(getattr(segment, "start", 0.0))
                    end = chunk.start_time + float(getattr(segment, "end", start))
                    if self.duplicate_suppressor.is_duplicate(start, text):
                        continue
                    row = self.store.add_transcript(chunk.index, start, end, text)
                    self.writer.refresh(self.store.snapshot())
                    print(f"[{format_timestamp(row.start)}] {row.speaker_label}: {row.text}", flush=True)
                    self._emit("transcript", {"row": row})
            except Exception as exc:  # pragma: no cover - environment-dependent
                self._emit("error", {"message": f"Transcription failed for chunk {chunk.index}: {exc}"})
            finally:
                with self._state_lock:
                    self._pending_transcription_chunks.discard(chunk.index)

    def _diarization_loop(self) -> None:
        try:
            pipeline, embedding_inference = self._load_pyannote_models()
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Pyannote model load failed: {exc}"})
            return

        while True:
            chunk = self._diarization_queue.get()
            if chunk is None:
                return
            try:
                turns = self._run_diarization(chunk, pipeline, embedding_inference)
                updated_rows = self.store.apply_diarization(chunk.index, turns)
                if updated_rows:
                    self.writer.refresh(self.store.snapshot())
                for row in updated_rows:
                    print(
                        f"[speaker update] [{format_timestamp(row.start)}] "
                        f"{row.speaker_label}: {row.text}",
                        flush=True,
                    )
                    self._emit("speaker_update", {"row": row})
                self._emit("speakers", {"speakers": self.speaker_registry.speakers()})
            except Exception as exc:  # pragma: no cover - environment-dependent
                self._emit("error", {"message": f"Diarization failed for chunk {chunk.index}: {exc}"})
            finally:
                with self._state_lock:
                    self._pending_diarization_chunks.discard(chunk.index)
                self._emit("lag", {"message": self.lag_status()})

    def _load_whisper_model(self) -> Any:
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

    def _load_pyannote_models(self) -> tuple[Any, Any]:
        import torch
        from pyannote.audio import Inference, Model, Pipeline

        pipeline = Pipeline.from_pretrained(self.config.pyannote_pipeline_dir)
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device("cpu"))
        embedding_model = Model.from_pretrained(self.config.pyannote_embedding_model_dir)
        embedding_inference = Inference(embedding_model, window="whole")
        return pipeline, embedding_inference

    def _run_diarization(self, chunk: AudioChunk, pipeline: Any, embedding_inference: Any) -> list[DiarizationTurn]:
        import numpy as np
        import torch
        from pyannote.core import Segment

        waveform = torch.from_numpy(np.asarray(chunk.samples, dtype=np.float32)).unsqueeze(0)
        audio = {"waveform": waveform, "sample_rate": chunk.sample_rate}
        diarization = pipeline(audio)
        turns: list[DiarizationTurn] = []
        for segment, _track, label in diarization.itertracks(yield_label=True):
            start = chunk.start_time + float(segment.start)
            end = chunk.start_time + float(segment.end)
            embedding = self._extract_embedding(
                embedding_inference,
                audio,
                Segment(float(segment.start), float(segment.end)),
            )
            turns.append(
                DiarizationTurn(
                    start=start,
                    end=end,
                    local_label=str(label),
                    embedding=embedding,
                    confidence=1.0 if embedding else 0.0,
                )
            )
        return turns

    @staticmethod
    def _extract_embedding(embedding_inference: Any, audio: dict[str, Any], segment: Any) -> Optional[list[float]]:
        try:
            result = embedding_inference.crop(audio, segment)
            if hasattr(result, "detach"):
                result = result.detach().cpu().numpy()
            if hasattr(result, "tolist"):
                values = result.tolist()
            else:
                values = list(result)
            while values and isinstance(values[0], list):
                values = values[0]
            return [float(value) for value in values]
        except Exception:
            return None

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
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) > 0:
            devices.append(
                {
                    "index": index,
                    "name": device.get("name", f"Input {index}"),
                    "channels": device.get("max_input_channels", 0),
                    "default_samplerate": device.get("default_samplerate"),
                }
            )
    return devices
