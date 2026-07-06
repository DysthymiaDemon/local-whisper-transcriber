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

## Double-Click Launch

On this machine, double-click:

```text
Open Offline Meeting Transcriber.pyw
```

Behavior:

- First launch opens a medium-sized setup window.
- Setup auto-starts and shows a task list, spinner, command text, and progress bars.
- Setup creates runtime files outside OneDrive under:

```text
C:\Users\<you>\Apps\OfflineMeetingTranscriber
```

- Setup creates `config.json`, `models\`, and `transcripts\` in that local app folder.
- Missing Python packages install with `pip --user`.
- Important setup messages stay visible for 5 seconds before moving on.
- When prerequisites finish, setup shows a 5-second launch countdown.
- Model folder buttons open/prepare local model locations.
- After setup is marked complete, future double-clicks open the GUI immediately.

If models are still missing, the GUI can open, but recording cannot start until these folders contain local model files:

```text
C:\Users\<you>\Apps\OfflineMeetingTranscriber\models\faster-whisper
C:\Users\<you>\Apps\OfflineMeetingTranscriber\models\pyannote-pipeline
C:\Users\<you>\Apps\OfflineMeetingTranscriber\models\pyannote-embedding
```

For a Windows laptop that does not have Python 3.12 installed, use:

```text
Open Offline Meeting Transcriber.cmd
```

That wrapper checks for Python 3.12. If missing, it attempts a current-user install with `winget install Python.Python.3.12 --scope user`, then starts the same setup GUI. If corporate policy blocks `.cmd` files or `winget`, use the portable EXE build below.

## Portable App Status

The portable build is the true no-Python option. It bundles Python and Python package dependencies into:

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

Portable EXE folder to copy to a corporate laptop:

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
py -3.12 -m venv .venv
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
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python meeting_transcriber_gui.py
```

Set model paths in GUI before pressing `Record`.

Source runtime data defaults to:

```text
C:\Users\<you>\Apps\OfflineMeetingTranscriber
```

Override for testing or custom install folders:

```powershell
$env:LOCAL_WHISPER_APP_ROOT = "D:\Apps\OfflineMeetingTranscriber"
```

## Model Requirements

Recommended starting folders:

- `models\faster-whisper`: CTranslate2 faster-whisper model, such as `Systran/faster-whisper-small`.
- `models\pyannote-pipeline`: local pyannote diarization pipeline with `config.yaml`.
- `models\pyannote-embedding`: local pyannote embedding model.

Runtime must not download models. Pyannote `config.yaml` must reference local paths only.

## Source Files

- `meeting_transcriber_gui.py`: PySide6 desktop GUI.
- `bootstrap_launcher.py`: first-run setup and direct GUI launch.
- `Open Offline Meeting Transcriber.pyw`: double-click launcher.
- `transcriber_engine.py`: recorder, queues, transcription, diarization, speaker state, transcript writer.
- `app_config.py`: portable app path/config handling.
- `build_portable.ps1`: PyInstaller one-dir bundle build.
- `local_whisper_transcriber.spec`: PyInstaller packaging spec.
- `config.template.json`: portable config template.
- `tests/`: unit tests for engine and portable config behavior.

## Test

```powershell
py -3.12 -B -m unittest discover -s tests -v
py -3.12 -B -m py_compile bootstrap_launcher.py app_config.py transcriber_engine.py meeting_transcriber_gui.py "Open Offline Meeting Transcriber.pyw" tests\test_bootstrap_launcher.py tests\test_app_config.py tests\test_engine_core.py local_whisper_transcriber.spec
```

Expected current result: 23 tests pass.

## Corporate Laptop Notes

- Windows microphone access must be enabled.
- Source `.pyw` launch requires Python 3.12. `Open Offline Meeting Transcriber.cmd` can install it if `winget` is available. Portable EXE does not require system Python.
- If endpoint security blocks unsigned EXEs, IT must allow-list or sign `OfflineMeetingTranscriber.exe`.
- If diarization lags on CPU, increase chunk duration to 10-15 seconds or use smaller local models.
- Transcript output defaults to `C:\Users\<you>\Apps\OfflineMeetingTranscriber\transcripts\meeting_transcript.txt` in source mode.
- Avoid placing model folders under OneDrive. Large model files can trigger sync errors and corporate cloud policy warnings.
