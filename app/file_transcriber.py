"""Bounded-memory offline transcription of saved media."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

from transcriber_engine import (
    AtomicTranscriptWriter, DuplicateSuppressor, EngineConfig, TranscriptStore,
    load_whisper_model, transcription_segment_is_usable,
)


def audio_chunks(container, seconds: float, overlap: float, cancelled: threading.Event):
    """Yield 16 kHz mono windows, retaining only overlap and one decoded frame."""
    import av
    import numpy as np

    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    size, keep = round(seconds * 16000), round(overlap * 16000)
    step = size - keep
    if size <= 0 or keep < 0 or step <= 0:
        raise ValueError("Audio chunk size must exceed overlap at 16 kHz.")
    buffer = np.empty(0, dtype=np.float32)
    offset = 0
    fresh = 0

    def frames():
        for frame in container.decode(stream):
            if cancelled.is_set():
                return
            yield from resampler.resample(frame)
        if not cancelled.is_set():
            yield from resampler.resample(None)

    for frame in frames():
        samples = frame.to_ndarray().reshape(-1)
        buffer = np.concatenate((buffer, samples))
        fresh += len(samples)
        while len(buffer) >= size:
            if cancelled.is_set():
                return
            yield offset / 16000, buffer[:size].copy()
            buffer = buffer[step:].copy()
            offset += step
            fresh = len(buffer) - keep
    if fresh > 0 and len(buffer) and not cancelled.is_set():
        yield offset / 16000, buffer


class FileTranscriptionWorker:
    def __init__(self, config: EngineConfig, input_path: str, output_path: str, callback: Callable):
        self.config = replace(config, sample_rate=16000, capture_mode="mic", output_file=output_path)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.callback = callback
        self.cancelled = threading.Event()
        self.thread: threading.Thread | None = None
        self.store = TranscriptStore()

    def start(self):
        if self.thread is not None:
            raise RuntimeError("This file job has already started.")
        self.thread = threading.Thread(target=self.run, name="file-transcription", daemon=False)
        self.thread.start()

    def cancel(self):
        self.cancelled.set()

    def run(self):
        writer = AtomicTranscriptWriter(str(self.output_path))
        outcome = "failed"
        saved_path = None
        try:
            import av
            if self.input_path.resolve() == self.output_path.resolve():
                raise ValueError("Transcript output must differ from the source recording.")
            errors = self.config.validate()
            if errors:
                raise ValueError("\n".join(errors) + "\nComplete local model setup before transcribing files.")
            with av.open(str(self.input_path)) as container:
                if not container.streams.audio:
                    raise ValueError("The selected recording contains no audio track.")
                stream = container.streams.audio[0]
                total = (float(stream.duration * stream.time_base) if stream.duration is not None
                         else container.duration / av.time_base if container.duration else None)
                self.callback("status", {"message": "Loading local Whisper model"})
                model = load_whisper_model(self.config)
                if not self.cancelled.is_set():
                    self.callback("status", {"message": "Transcribing saved recording"})
                duplicates = DuplicateSuppressor(self.config.overlap_seconds + 0.5)
                for index, (start, samples) in enumerate(audio_chunks(
                    container, self.config.chunk_seconds, self.config.overlap_seconds, self.cancelled
                )):
                    if self.cancelled.is_set():
                        break
                    segments, _ = model.transcribe(
                        samples, language=self.config.language, vad_filter=False,
                        word_timestamps=False, beam_size=1, condition_on_previous_text=False,
                        no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
                    )
                    for segment in segments:
                        text = getattr(segment, "text", "").strip()
                        begin = start + float(getattr(segment, "start", 0.0))
                        end = start + float(getattr(segment, "end", len(samples) / 16000))
                        if transcription_segment_is_usable(segment, text) and not duplicates.is_duplicate(begin, text):
                            row = self.store.add_transcript(index, begin, end, text)
                            self.callback("transcript", {"row": row})
                    saved_path = writer.refresh_with_recovery(self.store.snapshot())
                    if saved_path != self.output_path:
                        raise OSError(f"Cannot save requested output. Partial transcript recovered to {saved_path}")
                    processed = start + len(samples) / 16000
                    self.callback("file_progress", {"processed": processed, "total": total})
                outcome = "cancelled" if self.cancelled.is_set() else "completed"
                if saved_path is None and not self.cancelled.is_set():
                    saved_path = writer.refresh_with_recovery(self.store.snapshot())
                    if saved_path != self.output_path:
                        outcome = "failed"
                        self.callback("error", {"message": f"Cannot save requested output. Recovered to {saved_path}"})
        except Exception as exc:
            if self.store.snapshot() and (saved_path is None or saved_path == self.output_path):
                try:
                    saved_path = writer.refresh_with_recovery(self.store.snapshot())
                except Exception as save_exc:
                    self.callback("error", {"message": f"Partial transcript could not be saved: {save_exc}"})
            guidance = " Complete local runtime setup before transcribing files." if isinstance(exc, ImportError) else ""
            self.callback("error", {"message": f"File transcription failed: {exc}{guidance}"})
        finally:
            self.callback("file_finished", {"outcome": outcome, "output_path": str(saved_path) if saved_path else None})
