# Offline Meeting Transcriber

Windows desktop app for offline meeting transcription with speaker diarization. It records from the laptop microphone, shows live transcript text first, then updates speaker labels asynchronously when diarization catches up.

Built for Windows 11 Enterprise laptops with no NVIDIA GPU. First setup can use internet to download packages and default public models; recording/runtime stays local and offline after setup.

## Quick Start

### Option A: Copy One File

Use this when Python 3.12 already exists on the laptop.

1. Copy this file anywhere:

```text
Open Offline Meeting Transcriber.pyw
```

2. Double-click it.
3. It creates this folder beside itself:

```text
OfflineMeetingTranscriber\
```

4. First-run setup opens automatically.
5. Review what will be installed, then click `Install`.
6. Setup installs private Python packages and downloads default public models into:

```text
OfflineMeetingTranscriber\models\faster-whisper
OfflineMeetingTranscriber\models\speechbrain-ecapa
```

7. The setup window closes and launches the transcriber. Click `Record`.

### Option B: Python Missing

Use this if `.pyw` does not open.

```text
Open Offline Meeting Transcriber.cmd
```

This checks for Python 3.12 and tries a current-user install through `winget`. If corporate policy blocks `.cmd` or `winget`, use the portable EXE build.

### Option C: No Python Install Allowed

Use the portable build:

```text
dist\OfflineMeetingTranscriber\OfflineMeetingTranscriber.exe
```

or:

```text
dist\OfflineMeetingTranscriber\Run-OfflineMeetingTranscriber.bat
```

Portable EXE bundles Python and Python packages. If first setup has internet, it downloads default models; otherwise copy model folders into `models\` before use.

## What It Does

- Records microphone audio locally.
- Transcribes with `faster-whisper` on CPU using INT8 quantization.
- Runs default non-gated speaker diarization with SpeechBrain ECAPA embeddings and local clustering.
- Keeps pyannote as an optional advanced backend if local pyannote folders are configured.
- Shows live transcript rows immediately as `Speaker ?`.
- Replaces pending labels later with `Speaker 1`, `Speaker 2`, etc.
- Lets users rename speakers in the GUI.
- Writes an updated transcript file continuously.
- Uses internet only during first setup downloads; no cloud APIs are used during recording/runtime.

## Double-Click Launch

Copy or double-click this single file:

```text
Open Offline Meeting Transcriber.pyw
```

Behavior:

- If the file is copied to `X:\SomeFolder\`, it creates and uses:

```text
X:\SomeFolder\OfflineMeetingTranscriber
```

- If the file is already inside a folder named `OfflineMeetingTranscriber`, it uses that folder directly.
- First launch opens a medium-sized setup window.
- Before setup opens, a small startup window checks local folders, Python packages, and first-time setup status.
- Setup first shows what will be installed, then the `Install` button starts a task list, spinner, command text, and progress bars.
- Setup extracts the app source and resources into the install folder.
- Setup creates `config.json`, `models\`, and `transcripts\` there.
- Missing Python packages install into `OfflineMeetingTranscriber\.runtime\site-packages` unless the install folder is inside OneDrive.
- For OneDrive folders, Python packages install into `%LOCALAPPDATA%\OfflineMeetingTranscriberRuntime\site-packages` so the app does not create copied `python.exe` or `pythonw.exe` files.
- No Python packages are installed into global Python or user site-packages.
- If an older OneDrive install already has `OfflineMeetingTranscriber\.runtime`, setup moves it to `%LOCALAPPDATA%\OfflineMeetingTranscriberRuntime` or a `legacy-runtime` backup there.
- Important setup messages stay visible for 5 seconds before moving on.
- When prerequisites finish, setup shows a 5-second launch countdown.
- After setup closes, the small startup window appears again while the main transcriber loads.
- Setup downloads `Systran/faster-whisper-small` and `speechbrain/spkrec-ecapa-voxceleb` by default.
- Model folder buttons open/prepare local model locations for manual or optional advanced setup.
- After setup is marked complete, future double-clicks open the GUI immediately.

If default models are still missing and setup cannot download them, recording cannot start until these folders contain local model files:

```text
OfflineMeetingTranscriber\models\faster-whisper
OfflineMeetingTranscriber\models\speechbrain-ecapa
```

For a Windows laptop that does not have Python 3.12 installed, use:

```text
Open Offline Meeting Transcriber.cmd
```

That wrapper checks for Python 3.12. If missing, it attempts a current-user install with `winget install Python.Python.3.12 --scope user`, then starts the same setup GUI. If corporate policy blocks `.cmd` files or `winget`, use the portable EXE build below.

## How To Use The App

1. Open the app.
   A small startup window shows current loading status before the main window appears.
2. Select microphone from the dropdown.
3. Confirm model paths point to local folders:

```text
models\faster-whisper
models\speechbrain-ecapa
```

4. Confirm output file path, default:

```text
transcripts\meeting_transcript.txt
```

5. Click `Record`.
6. Speak into laptop microphone.
7. Watch live transcript table:

```text
Speaker ?: initial transcript text
Speaker 1: updated after diarization finishes
```

8. Use `Pause` to pause capture.
9. Use `Stop` to flush final chunk and save transcript.
10. Rename speakers from speaker panel; output file refreshes with new names.

Saved transcript format:

```text
[00:00:12] Speaker 1: Hello team.
[00:00:18] Speaker 2: Let's start with actions.
```

The console also prints live lines and later speaker updates.

## Error Logs

Setup and runtime errors are appended here:

```text
error_log.txt
```

The file is created beside `Open Offline Meeting Transcriber.pyw`, not inside `OfflineMeetingTranscriber\`. Use this file when setup fails, dependencies fail to install, or Record shows a model/configuration error. The installer also shows step details on mouseover in the left task list.

## First-Run Folder Layout

After opening copied `.pyw`, install folder contains:

```text
OfflineMeetingTranscriber/
  app/
  resources/
  config.json
  models/
    faster-whisper/
    speechbrain-ecapa/
    pyannote-pipeline/
    pyannote-embedding/
  transcripts/
error_log.txt   (created after an error)
```

Keep model folders and transcripts in this folder. Avoid OneDrive for large model files when possible.

The `.pyw` launcher still needs Python 3.12 to start. After that, app dependencies are loaded from a private `site-packages` folder. If the app is under OneDrive, dependencies are stored in `%LOCALAPPDATA%\OfflineMeetingTranscriberRuntime\site-packages` to avoid corporate OneDrive and group-policy blocks on copied `.exe` files.

## Portable App Status

The portable build is the true no-Python option. It bundles Python and Python package dependencies into:

```text
dist\OfflineMeetingTranscriber\_internal
```

AI model files are not bundled into the EXE. First setup can download default public models when internet is available. For air-gapped transfer, copy these folders into the portable app folder before transfer:

```text
dist\OfflineMeetingTranscriber\models\faster-whisper
dist\OfflineMeetingTranscriber\models\speechbrain-ecapa
```

Optional pyannote advanced mode still needs manually prepared local pyannote folders.

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
    speechbrain-ecapa/
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
python -m pip install -r .\resources\requirements-build.txt
.\packaging\build_portable.ps1 -Clean
```

Then either let first setup download default models or copy model folders into:

```text
dist\OfflineMeetingTranscriber\models\
```

Move the whole `dist\OfflineMeetingTranscriber` folder to the target laptop.

For fully air-gapped build steps, see [PORTABLE_WINDOWS.md](docs/PORTABLE_WINDOWS.md) and [offline_setup.md](docs/offline_setup.md).

## Run From Source

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\resources\requirements.txt
python .\app\meeting_transcriber_gui.py
```

Set model paths in GUI before pressing `Record`, or run the first-time setup flow to download defaults.

Source runtime data defaults to:

```text
C:\Users\<you>\Apps\OfflineMeetingTranscriber
```

Override for testing or custom install folders:

```powershell
$env:LOCAL_WHISPER_APP_ROOT = "D:\Apps\OfflineMeetingTranscriber"
```

## Model Requirements

Default one-touch folders:

- `models\faster-whisper`: CTranslate2 faster-whisper model downloaded from `Systran/faster-whisper-small`; must contain `model.bin`.
- `models\speechbrain-ecapa`: SpeechBrain ECAPA speaker embedding model downloaded from `speechbrain/spkrec-ecapa-voxceleb`; must contain `hyperparams.yaml` plus `embedding_model.ckpt` or `model.ckpt`.

Optional advanced pyannote mode:

- Requires installing `pyannote.audio` into the app runtime or source environment.
- `models\pyannote-pipeline`: local pyannote diarization pipeline; must contain `config.yaml`.
- `models\pyannote-embedding`: local pyannote embedding model; must contain `config.yaml` plus `pytorch_model.bin` or `model.safetensors`.

Runtime recording does not download models. Pyannote `config.yaml` must reference local paths only.

## Source Files

- `Open Offline Meeting Transcriber.pyw`: self-extracting double-click launcher.
- `app/`: runtime source for setup, GUI, engine, and config handling.
- `resources/`: pip requirement files and config template.
- `packaging/`: PyInstaller build script/spec, portable BAT, standalone launcher generator.
- `docs/`: offline setup and portable Windows build docs.
- `tests/`: unit tests for engine and portable config behavior.

## Test

```powershell
py -3.12 -B -m unittest discover -s tests -v
py -3.12 -B -m py_compile app\app_config.py app\bootstrap_launcher.py app\meeting_transcriber_gui.py app\transcriber_engine.py packaging\build_standalone_pyw.py "Open Offline Meeting Transcriber.pyw" tests\test_bootstrap_launcher.py tests\test_app_config.py tests\test_engine_core.py tests\test_standalone_launcher.py packaging\local_whisper_transcriber.spec
```

Expected current result: 62 tests pass.

## Corporate Laptop Notes

- Windows microphone access must be enabled.
- Source `.pyw` launch requires Python 3.12. `Open Offline Meeting Transcriber.cmd` can install it if `winget` is available. Portable EXE does not require system Python.
- If endpoint security blocks unsigned EXEs, IT must allow-list or sign `OfflineMeetingTranscriber.exe`.
- First setup uses Windows certificate trust via `truststore` for Hugging Face downloads. If corporate HTTPS inspection still causes `CERTIFICATE_VERIFY_FAILED`, IT can place `company-ca.pem`, `corporate-ca.pem`, or `ca-bundle.pem` beside `Open Offline Meeting Transcriber.pyw`, or set `LOCAL_WHISPER_CA_BUNDLE`.
- If diarization lags on CPU, increase chunk duration to 10-15 seconds or use smaller local models.
- Transcript output defaults to `<install folder>\transcripts\meeting_transcript.txt` when launched from the standalone `.pyw`.
- Avoid placing model folders under OneDrive. Large model files can trigger sync errors and corporate cloud policy warnings.
