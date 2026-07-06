# Local Whisper Transcriber

Offline Windows desktop app for live meeting transcription with async speaker diarization.

## Files

- `meeting_transcriber_gui.py` - PySide6 GUI entrypoint.
- `transcriber_engine.py` - recording, chunk queues, faster-whisper transcription, pyannote diarization, speaker continuity, transcript writer.
- `offline_setup.md` - air-gapped install and model transfer guide.
- `requirements.txt` - pinned runtime packages.
- `tests/` - unit tests for core engine behavior.

## Run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links .\wheelhouse -r requirements.txt
python meeting_transcriber_gui.py
```

Set local model paths in the GUI before pressing `Record`.
