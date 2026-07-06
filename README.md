# Offline Meeting Transcriber

Windows desktop app for offline meeting transcription with speaker diarization. It records from the laptop microphone, shows live transcript text first, then updates speaker labels asynchronously when diarization catches up.

Built for Windows 11 Enterprise laptops with no NVIDIA GPU and no runtime internet access.

## What It Does

- Records microphone audio locally.
- Transcribes with `faster-whisper` on CPU using INT8 quantization.
- Runs `pyannote.audio` diarization from local model folders.
- Shows live transcript rows immediately as `Speaker ?`.
- Replaces pending labels later with `Speaker 1`, `Speaker 2`, etc.
- Lets users rename speakers in the GUI.
- Writes an updated transcript file continuously.
- Runs without cloud APIs during runtime.

## Portable App Status

The portable build bundles Python and Python package dependencies into:

```text
dist\OfflineMeetingTranscriber\_internal
```

AI model files are not bundled automatically. They must be copied into the portable app folder before transfer:

```text
dist\OfflineMeetingTranscriber\models\faster-whisper
dist\OfflineMeetingTranscriber\models\pyannote-pipeline
dist\OfflineMeetingTranscriber\models\pyannote-embedding
```

This keeps gated/licensed and large model files explicit.

## Target Layout

Final folder to copy to a corporate laptop:

```text
OfflineMeetingTranscriber/
  OfflineMeetingTranscriber.exe
  Run-OfflineMeetingTranscriber.bat
  config.json
  _internal/
  models/
    faster-whisper/
    pyannote-pipeline/
    pyannote-embedding/
  transcripts/
  offline_setup.md
```

Run:

```text
Run-OfflineMeetingTranscriber.bat
```

No admin account needed if Windows policy allows unsigned local executables and microphone access.

## Build Portable Windows Bundle

On an internet-connected Windows build machine:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-build.txt
.\build_portable.ps1 -Clean
```

Then copy model folders into:

```text
dist\OfflineMeetingTranscriber\models\
```

Move the whole `dist\OfflineMeetingTranscriber` folder to the target laptop.

For fully air-gapped build steps, see [PORTABLE_WINDOWS.md](PORTABLE_WINDOWS.md) and [offline_setup.md](offline_setup.md).

## Run From Source

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python meeting_transcriber_gui.py
```

Set model paths in GUI before pressing `Record`.

## Model Requirements

Recommended starting folders:

- `models\faster-whisper`: CTranslate2 faster-whisper model, such as `Systran/faster-whisper-small`.
- `models\pyannote-pipeline`: local pyannote diarization pipeline with `config.yaml`.
- `models\pyannote-embedding`: local pyannote embedding model.

Runtime must not download models. Pyannote `config.yaml` must reference local paths only.

## Source Files

- `meeting_transcriber_gui.py`: PySide6 desktop GUI.
- `transcriber_engine.py`: recorder, queues, transcription, diarization, speaker state, transcript writer.
- `app_config.py`: portable app path/config handling.
- `build_portable.ps1`: PyInstaller one-dir bundle build.
- `local_whisper_transcriber.spec`: PyInstaller packaging spec.
- `config.template.json`: portable config template.
- `tests/`: unit tests for engine and portable config behavior.

## Test

```powershell
python -B -m unittest discover -s tests -v
python -B -m py_compile app_config.py transcriber_engine.py meeting_transcriber_gui.py tests\test_app_config.py tests\test_engine_core.py local_whisper_transcriber.spec
```

Expected current result: 13 tests pass.

## Corporate Laptop Notes

- Windows microphone access must be enabled.
- If endpoint security blocks unsigned EXEs, IT must allow-list or sign `OfflineMeetingTranscriber.exe`.
- If diarization lags on CPU, increase chunk duration to 10-15 seconds or use smaller local models.
- Transcript output defaults to `transcripts\meeting_transcript.txt`.
