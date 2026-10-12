"""Decode one saved recording to 16 kHz mono float32 PCM in a separate, contained process.

Meeting files come from other people, and FFmpeg parses them in native code. Decoding happens here, in a child
process that the GUI puts in a Windows job (memory cap, no child processes), with Python networking blocked and
FFmpeg limited to local files. A crash or hang in the decoder ends this process, not the app.

Protocol, on stdout: one JSON header line {"duration": seconds or null}, then raw little-endian float32 samples.
On failure: one JSON line {"error": message} on stderr and exit code 2.
The process waits for one byte on stdin before opening the file, so the GUI can contain it first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import BinaryIO, Iterator

SAMPLE_RATE = 16000
DECODE_EXIT_ERROR = 2


def checked_input_path(raw: str) -> Path:
    """Only an existing regular local file: no URLs, device names, pipes or folders."""
    if not raw or "://" in raw or raw.startswith(("\\\\.\\", "\\\\?\\", "//./", "//?/")):
        raise ValueError("Only a recording file on this computer can be transcribed.")
    path = Path(raw).resolve()
    if not path.is_file():
        raise ValueError(f"Recording not found: {path}")
    return path


def decoded_samples(container, cancelled=None) -> Iterator:
    """Resample the first audio stream to 16 kHz mono float32 arrays."""
    import av

    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)

    def stopped() -> bool:
        return cancelled is not None and cancelled.is_set()

    for frame in container.decode(stream):
        if stopped():
            return
        for resampled in resampler.resample(frame):
            yield resampled.to_ndarray().reshape(-1)
    if not stopped():
        for resampled in resampler.resample(None):
            yield resampled.to_ndarray().reshape(-1)


def decode_to(output: BinaryIO, raw_path: str) -> None:
    import av
    import numpy as np

    path = checked_input_path(raw_path)
    # protocol_whitelist=file: FFmpeg may not follow a playlist or reference to http, rtsp, etc.
    with av.open(str(path), options={"protocol_whitelist": "file"}) as container:
        if not container.streams.audio:
            raise ValueError("The selected recording contains no audio track.")
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = None
        output.write((json.dumps({"duration": duration}) + "\n").encode("utf-8"))
        for samples in decoded_samples(container):
            output.write(np.ascontiguousarray(samples, dtype="<f4").tobytes())
    output.flush()


def main(argv: list[str]) -> int:
    from offline_guard import enforce_offline

    enforce_offline()
    sys.stdin.buffer.read(1)  # Wait until the GUI has placed this process in its job.
    try:
        if len(argv) != 1:
            raise ValueError("Expected one recording path.")
        decode_to(sys.stdout.buffer, argv[0])
    except BrokenPipeError:
        return 0  # The GUI stopped reading (cancelled); nothing to report.
    except Exception as exc:
        sys.stderr.write(json.dumps({"error": str(exc) or type(exc).__name__}) + "\n")
        sys.stderr.flush()
        return DECODE_EXIT_ERROR
    return 0
