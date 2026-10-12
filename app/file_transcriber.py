"""Bounded-memory offline transcription of saved media."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Iterator

from transcriber_engine import (
    AtomicTranscriptWriter, DuplicateSuppressor, EngineConfig, TranscriptStore,
    load_whisper_model, transcription_segment_is_usable,
)
from win_job import close_job, contain_process

# Whisper pads every window to 30 seconds, so saved recordings use near-full windows:
# less encoder work per audio minute and more context for the model.
FILE_WINDOW_SECONDS = 28.0
FILE_OVERLAP_SECONDS = 2.0
DECODER_MEMORY_LIMIT_BYTES = 2 * 1024**3
DECODER_IDLE_TIMEOUT_SECONDS = 120.0
DECODER_READ_BYTES = 256 * 1024
DECODER_STDERR_LIMIT = 64 * 1024
CREATE_NO_WINDOW = 0x08000000
DECODER_BOOTSTRAP = (
    "import json, sys; sys.path[:0] = json.loads(sys.argv[1]); "
    "import audio_decode_worker; raise SystemExit(audio_decode_worker.main(sys.argv[2:]))"
)


def pcm_windows(blocks: Iterable, seconds: float, overlap: float, cancelled: threading.Event):
    """Yield (start seconds, samples) 16 kHz windows from sample blocks, keeping only overlap plus one window."""
    import numpy as np

    size, keep = round(seconds * 16000), round(overlap * 16000)
    step = size - keep
    if size <= 0 or keep < 0 or step <= 0:
        raise ValueError("Audio chunk size must exceed overlap at 16 kHz.")
    # Blocks are collected in a list and joined only when a full window is ready,
    # instead of re-copying the whole buffer for every block.
    pending: list = []
    pending_size = 0
    offset = 0
    fresh = 0
    for samples in blocks:
        if cancelled.is_set():
            return
        pending.append(samples)
        pending_size += len(samples)
        fresh += len(samples)
        if pending_size < size:
            continue
        buffer = np.concatenate(pending)
        while len(buffer) >= size:
            if cancelled.is_set():
                return
            yield offset / 16000, buffer[:size].copy()
            buffer = buffer[step:]
            offset += step
            fresh = len(buffer) - keep
        pending = [buffer.copy()]
        pending_size = len(buffer)
    buffer = np.concatenate(pending) if pending else np.empty(0, dtype=np.float32)
    if fresh > 0 and len(buffer) and not cancelled.is_set():
        yield offset / 16000, buffer


def decoder_command(input_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--decode-worker", str(input_path)]
    search_path = [entry for entry in sys.path if entry and os.path.isdir(entry)]
    # -I: ignore PYTHON* variables and the user site folder; the app's own folders are passed explicitly.
    return [sys.executable, "-I", "-c", DECODER_BOOTSTRAP, json.dumps(search_path), str(input_path)]


class DecodedAudio:
    """Samples from a recording decoded by audio_decode_worker in a separate, contained process."""

    def __init__(self, input_path: Path, cancelled: threading.Event,
                 idle_timeout: float = DECODER_IDLE_TIMEOUT_SECONDS):
        self.cancelled = cancelled
        self.idle_timeout = idle_timeout
        self.duration: float | None = None
        self._job = None
        self._stderr = bytearray()
        self._blocks: queue.Queue = queue.Queue(maxsize=8)
        self.process = subprocess.Popen(
            decoder_command(input_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            self._job = contain_process(self.process, DECODER_MEMORY_LIMIT_BYTES)
            assert self.process.stdin is not None
            self.process.stdin.write(b"g")  # Contained now: let the decoder open the file.
            self.process.stdin.close()
        except Exception:
            self.close()
            raise
        threading.Thread(target=self._read_stdout, name="decoder-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="decoder-stderr", daemon=True).start()
        header = self._next_item()
        if header is None:
            self._raise_decoder_failure()
            return  # Cancelled before the decoder started.
        try:
            self.duration = json.loads(header.decode("utf-8"))["duration"]
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            self.close()
            raise RuntimeError("The audio decoder sent an unreadable header.") from exc

    def _read_stdout(self) -> None:
        stdout = self.process.stdout
        assert stdout is not None
        try:
            self._blocks.put(stdout.readline() or None)
            while True:
                data = stdout.read(DECODER_READ_BYTES)
                if not data:
                    break
                self._blocks.put(data)
        except (OSError, ValueError):
            pass
        finally:
            self._blocks.put(None)

    def _read_stderr(self) -> None:
        stderr = self.process.stderr
        assert stderr is not None
        try:
            for line in stderr:
                if len(self._stderr) < DECODER_STDERR_LIMIT:
                    self._stderr.extend(line[: DECODER_STDERR_LIMIT - len(self._stderr)])
        except (OSError, ValueError):
            pass

    def _next_item(self) -> bytes | None:
        waited = 0.0
        while True:
            if self.cancelled.is_set():
                self.close()
                return None
            try:
                return self._blocks.get(timeout=0.25)
            except queue.Empty:
                waited += 0.25
                if waited >= self.idle_timeout:
                    self.close()
                    raise RuntimeError("The audio decoder stopped responding and was stopped.")

    def _raise_decoder_failure(self) -> None:
        if self.cancelled.is_set():
            return
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        self.close()
        message = ""
        for line in bytes(self._stderr).decode("utf-8", errors="replace").splitlines():
            try:
                message = str(json.loads(line).get("error", "")) or message
            except (ValueError, AttributeError):
                continue
        if not message:
            message = f"The audio decoder stopped unexpectedly (exit code {code})."
        raise RuntimeError(message)

    def samples(self) -> Iterator:
        import numpy as np

        remainder = b""
        while True:
            data = self._next_item()
            if data is None:
                break
            data = remainder + data
            usable = len(data) - len(data) % 4
            remainder = data[usable:]
            if usable:
                yield np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)
        if self.cancelled.is_set():
            return
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            code = None
        if code != 0:
            self._raise_decoder_failure()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
        close_job(self._job)
        self._job = None
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def __enter__(self) -> "DecodedAudio":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class FileTranscriptionWorker:
    def __init__(self, config: EngineConfig, input_path: str, output_path: str, callback: Callable):
        self.config = replace(
            config, sample_rate=16000, capture_mode="mic", output_file=output_path,
            chunk_seconds=FILE_WINDOW_SECONDS, overlap_seconds=FILE_OVERLAP_SECONDS,
        )
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
            if self.input_path.resolve() == self.output_path.resolve():
                raise ValueError("Transcript output must differ from the source recording.")
            errors = self.config.validate()
            if errors:
                raise ValueError("\n".join(errors) + "\nComplete local model setup before transcribing files.")
            with DecodedAudio(self.input_path, self.cancelled) as decoded:
                total = decoded.duration
                self.callback("status", {"message": "Loading local Whisper model"})
                model = load_whisper_model(self.config)
                if not self.cancelled.is_set():
                    self.callback("status", {"message": "Transcribing saved recording"})
                duplicates = DuplicateSuppressor(self.config.overlap_seconds + 0.5)
                for index, (start, samples) in enumerate(pcm_windows(
                    decoded.samples(), self.config.chunk_seconds, self.config.overlap_seconds, self.cancelled
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
