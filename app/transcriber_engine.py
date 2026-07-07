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
EMBEDDING_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
DIARIZATION_BACKEND_LOCAL_ECAPA = "local-ecapa"
DIARIZATION_BACKEND_PYANNOTE = "pyannote"
SPEECHBRAIN_CHECKPOINT_FILES = ("embedding_model.ckpt", "model.ckpt")
LOCAL_DIARIZATION_MIN_SECONDS = 0.25
LOCAL_DIARIZATION_TARGET_SECONDS = 1.0
MIC_SILENCE_RMS_THRESHOLD = 0.0001


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


def _force_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@dataclass(frozen=True)
class EngineConfig:
    whisper_model_dir: str = r"C:\models\faster-whisper-small"
    diarization_backend: str = DIARIZATION_BACKEND_LOCAL_ECAPA
    speaker_embedding_model_dir: str = r"C:\models\speechbrain-ecapa"
    speaker_cluster_distance_threshold: float = 0.55
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
        whisper_path = self._validate_folder("Whisper model", self.whisper_model_dir, errors)
        if whisper_path and not (whisper_path / "model.bin").is_file():
            errors.append(
                "Whisper model is incomplete: expected model.bin in "
                f"{whisper_path}. Copy a CTranslate2 faster-whisper model folder into this path."
            )

        if self.diarization_backend == DIARIZATION_BACKEND_LOCAL_ECAPA:
            speaker_path = self._validate_folder(
                "Speaker embedding model", self.speaker_embedding_model_dir, errors
            )
            if speaker_path:
                missing_speaker_files: list[str] = []
                if not (speaker_path / "hyperparams.yaml").is_file():
                    missing_speaker_files.append("hyperparams.yaml")
                if not any((speaker_path / name).is_file() for name in SPEECHBRAIN_CHECKPOINT_FILES):
                    missing_speaker_files.append("embedding_model.ckpt")
                if missing_speaker_files:
                    errors.append(
                        "Speaker embedding model is incomplete: expected "
                        + ", ".join(missing_speaker_files)
                        + f" in {speaker_path}. Copy or download the SpeechBrain ECAPA model folder into this path."
                    )
        elif self.diarization_backend == DIARIZATION_BACKEND_PYANNOTE:
            pipeline_path = self._validate_folder("Pyannote pipeline", self.pyannote_pipeline_dir, errors)
            if pipeline_path and not (pipeline_path / "config.yaml").is_file():
                errors.append(
                    "Pyannote pipeline is incomplete: expected config.yaml in "
                    f"{pipeline_path}. Copy the local pyannote pipeline folder into this path."
                )

            embedding_path = self._validate_folder("Pyannote embedding model", self.pyannote_embedding_model_dir, errors)
            if embedding_path:
                missing_embedding_files: list[str] = []
                if not (embedding_path / "config.yaml").is_file():
                    missing_embedding_files.append("config.yaml")
                if not any((embedding_path / name).is_file() for name in EMBEDDING_WEIGHT_FILES):
                    missing_embedding_files.append("pytorch_model.bin or model.safetensors")
                if missing_embedding_files:
                    errors.append(
                        "Pyannote embedding model is incomplete: expected "
                        + ", ".join(missing_embedding_files)
                        + f" in {embedding_path}. Copy the local pyannote embedding model folder into this path."
                    )
        else:
            errors.append(
                "Diarization backend must be local-ecapa or pyannote."
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


def cluster_local_embeddings(embeddings: list[list[float]], distance_threshold: float = 0.55) -> list[int]:
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]
    try:
        from sklearn.cluster import AgglomerativeClustering

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
        return [int(label) for label in clustering.fit_predict(embeddings)]
    except Exception:
        labels: list[int] = []
        centroids: list[list[float]] = []
        for embedding in embeddings:
            normalized = _normalize_embedding(embedding)
            if normalized is None:
                labels.append(-1)
                continue
            best_index = -1
            best_distance = 2.0
            for index, centroid in enumerate(centroids):
                distance = 1.0 - cosine_similarity(normalized, centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index >= 0 and best_distance <= distance_threshold:
                labels.append(best_index)
            else:
                centroids.append(normalized)
                labels.append(len(centroids) - 1)
        return labels


def extract_row_audio_window(chunk: AudioChunk, row: TranscriptRow, target_seconds: float = LOCAL_DIARIZATION_TARGET_SECONDS) -> Any:
    import numpy as np

    samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
    sample_rate = chunk.sample_rate
    relative_start = max(0.0, row.start - chunk.start_time)
    relative_end = max(relative_start, row.end - chunk.start_time)
    duration = relative_end - relative_start
    if duration < target_seconds:
        midpoint = (relative_start + relative_end) / 2.0
        relative_start = midpoint - (target_seconds / 2.0)
        relative_end = midpoint + (target_seconds / 2.0)

    start_sample = math.floor(relative_start * sample_rate)
    end_sample = math.ceil(relative_end * sample_rate)
    left_pad = max(0, -start_sample)
    right_pad = max(0, end_sample - samples.size)
    start_sample = max(0, start_sample)
    end_sample = min(samples.size, end_sample)
    window = samples[start_sample:end_sample]
    if left_pad or right_pad:
        window = np.pad(window, (left_pad, right_pad))
    return window.astype(np.float32, copy=False)


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
        model_sample_rate = self.config.sample_rate
        input_sample_rate = preferred_input_sample_rate(sd, self.config.input_device, model_sample_rate)
        chunk_samples = int(self.config.chunk_seconds * input_sample_rate)
        overlap_samples = int(self.config.overlap_seconds * input_sample_rate)
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
                    level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                    with self._state_lock:
                        self._audio_block_count += 1
                        self._audio_peak_rms = max(self._audio_peak_rms, level)
                    self._emit("level", {"rms": level, "percent": rms_to_meter_percent(level)})
                    accumulated = np.concatenate([accumulated, mono])
                    while accumulated.size >= chunk_samples:
                        start_time = max(
                            0.0,
                            ((chunk_index * (chunk_samples - keep_samples)) / input_sample_rate),
                        )
                        samples = resample_audio(accumulated[:chunk_samples].copy(), input_sample_rate, model_sample_rate)
                        self._emit("log", {"message": f"Audio chunk {chunk_index} captured ({self.config.chunk_seconds:.1f}s)"})
                        self._raw_chunk_queue.put(
                            AudioChunk(chunk_index, start_time, samples, model_sample_rate)
                        )
                        with self._state_lock:
                            self._audio_chunk_count += 1
                        chunk_index += 1
                        accumulated = accumulated[chunk_samples - keep_samples :]
                if accumulated.size > input_sample_rate // 2:
                    start_time = max(0.0, time.monotonic() - stream_start - (accumulated.size / input_sample_rate))
                    samples = resample_audio(accumulated.copy(), input_sample_rate, model_sample_rate)
                    self._emit("log", {"message": f"Final audio chunk {chunk_index} captured"})
                    self._raw_chunk_queue.put(
                        AudioChunk(chunk_index, start_time, samples, model_sample_rate, is_final=True)
                    )
                    with self._state_lock:
                        self._audio_chunk_count += 1
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
            self._emit("log", {"message": "Loading Whisper model..."})
            model = self._load_whisper_model()
            self._emit("log", {"message": "Whisper model ready"})
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Whisper model load failed: {exc}"})
            return

        while True:
            chunk = self._transcription_queue.get()
            if chunk is None:
                return
            try:
                self._emit("log", {"message": f"Transcribing chunk {chunk.index}..."})
                segment_count = 0
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
                    segment_count += 1
                    self.writer.refresh(self.store.snapshot())
                    print(f"[{format_timestamp(row.start)}] {row.speaker_label}: {row.text}", flush=True)
                    self._emit("transcript", {"row": row})
                if segment_count == 0:
                    self._emit("log", {"message": f"No speech detected in chunk {chunk.index}"})
            except Exception as exc:  # pragma: no cover - environment-dependent
                self._emit("error", {"message": f"Transcription failed for chunk {chunk.index}: {exc}"})
            finally:
                with self._state_lock:
                    self._pending_transcription_chunks.discard(chunk.index)

    def _diarization_loop(self) -> None:
        try:
            self._emit("log", {"message": "Loading diarization model..."})
            diarizer = self._load_diarization_backend()
            self._emit("log", {"message": "Diarization model ready"})
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._emit("error", {"message": f"Diarization model load failed: {exc}"})
            return

        while True:
            chunk = self._diarization_queue.get()
            if chunk is None:
                return
            try:
                turns = self._run_diarization_backend(chunk, diarizer)
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

    def _load_diarization_backend(self) -> Any:
        if self.config.diarization_backend == DIARIZATION_BACKEND_LOCAL_ECAPA:
            return self._load_local_speaker_model()
        if self.config.diarization_backend == DIARIZATION_BACKEND_PYANNOTE:
            return self._load_pyannote_models()
        raise ValueError(f"Unsupported diarization backend: {self.config.diarization_backend}")

    def _run_diarization_backend(self, chunk: AudioChunk, diarizer: Any) -> list[DiarizationTurn]:
        if self.config.diarization_backend == DIARIZATION_BACKEND_LOCAL_ECAPA:
            return self._run_local_ecapa_diarization(chunk, diarizer)
        pipeline, embedding_inference = diarizer
        return self._run_pyannote_diarization(chunk, pipeline, embedding_inference)

    def _load_pyannote_models(self) -> tuple[Any, Any]:
        _force_offline_mode()
        import torch
        from pyannote.audio import Inference, Model, Pipeline

        pipeline = Pipeline.from_pretrained(self.config.pyannote_pipeline_dir)
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device("cpu"))
        embedding_model = Model.from_pretrained(self.config.pyannote_embedding_model_dir)
        embedding_inference = Inference(embedding_model, window="whole")
        return pipeline, embedding_inference

    def _load_local_speaker_model(self) -> Any:
        _force_offline_mode()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # pragma: no cover - speechbrain older import path
            from speechbrain.pretrained import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        return EncoderClassifier.from_hparams(
            source=self.config.speaker_embedding_model_dir,
            savedir=self.config.speaker_embedding_model_dir,
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cpu"},
        )

    def _run_local_ecapa_diarization(self, chunk: AudioChunk, classifier: Any) -> list[DiarizationTurn]:
        rows = self._wait_for_chunk_transcript_rows(chunk.index)
        if not rows:
            return []

        embeddings: list[list[float]] = []
        row_indexes: list[int] = []
        for index, row in enumerate(rows):
            if (row.end - row.start) < LOCAL_DIARIZATION_MIN_SECONDS:
                continue
            audio_window = extract_row_audio_window(chunk, row)
            embedding = self._extract_speechbrain_embedding(classifier, audio_window)
            if embedding is None:
                continue
            embeddings.append(embedding)
            row_indexes.append(index)

        labels = cluster_local_embeddings(embeddings, self.config.speaker_cluster_distance_threshold)
        turns: list[DiarizationTurn] = []
        label_by_row = {row_index: labels[index] for index, row_index in enumerate(row_indexes)}
        embedding_by_row = {row_index: embeddings[index] for index, row_index in enumerate(row_indexes)}
        for index, row in enumerate(rows):
            local_label = label_by_row.get(index)
            embedding = embedding_by_row.get(index)
            turns.append(
                DiarizationTurn(
                    start=row.start,
                    end=row.end,
                    local_label=f"local_{local_label}" if local_label is not None and local_label >= 0 else "unknown",
                    embedding=embedding,
                    confidence=1.0 if embedding is not None else 0.0,
                )
            )
        return turns

    def _wait_for_chunk_transcript_rows(self, chunk_index: int) -> list[TranscriptRow]:
        deadline = time.monotonic() + max(30.0, self.config.chunk_seconds * 4.0)
        while True:
            rows = [row for row in self.store.snapshot() if row.chunk_index == chunk_index]
            with self._state_lock:
                transcription_pending = chunk_index in self._pending_transcription_chunks
            if rows or not transcription_pending or time.monotonic() >= deadline:
                return rows
            time.sleep(0.05)

    @staticmethod
    def _extract_speechbrain_embedding(classifier: Any, audio_window: Any) -> Optional[list[float]]:
        try:
            import numpy as np
            import torch

            waveform = torch.from_numpy(np.asarray(audio_window, dtype=np.float32)).float().unsqueeze(0)
            with torch.no_grad():
                result = classifier.encode_batch(waveform)
            if hasattr(result, "detach"):
                result = result.detach().cpu().numpy()
            values = result.tolist() if hasattr(result, "tolist") else list(result)
            while values and isinstance(values[0], list):
                values = values[0]
            return [float(value) for value in values]
        except Exception:
            return None

    def _run_pyannote_diarization(self, chunk: AudioChunk, pipeline: Any, embedding_inference: Any) -> list[DiarizationTurn]:
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
