# Portable Windows Build

Goal: produce a folder that runs on Windows 11 Enterprise without admin rights. User opens `Run-OfflineMeetingTranscriber.bat` or `OfflineMeetingTranscriber.exe`; bundled Python runtime and Python packages run from the folder.

Use PyInstaller `onedir`, not `onefile`. Torch, pyannote, Qt, CTranslate2, and model files are large; one-file extraction is slow and often blocked by corporate endpoint tools.

## Folder Layout

Final bundle:

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

No admin install needed on target laptop. Microphone access still depends on Windows privacy policy:

```text
Settings > Privacy & security > Microphone
Microphone access: On
Let desktop apps access your microphone: On
```

## Build On Internet-Connected Windows Machine

Use same Windows architecture as target laptop. Python 3.12 64-bit recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\resources\requirements-build.txt
python -m PyInstaller --clean --noconfirm .\packaging\local_whisper_transcriber.spec
```

Or use script:

```powershell
.\packaging\build_portable.ps1 -Clean
```

## Add Models

Copy local model folders into:

```text
dist\OfflineMeetingTranscriber\models\faster-whisper
dist\OfflineMeetingTranscriber\models\pyannote-pipeline
dist\OfflineMeetingTranscriber\models\pyannote-embedding
```

`config.json` defaults to these relative folders. Edit only if model folders use different names.
Keep model folders beside the EXE as shown above, not inside `_internal`.

## Offline Wheelhouse Build

For reproducible corporate build machine:

```powershell
mkdir wheelhouse
py -3.12 -m pip download -r .\resources\requirements-build.txt -d wheelhouse
.\packaging\build_portable.ps1 -Clean -Wheelhouse .\wheelhouse
```

For CPU-only PyTorch wheels:

```powershell
py -3.12 -m pip download torch==2.11.0 torchaudio==2.11.0 `
  --index-url https://download.pytorch.org/whl/cpu `
  -d wheelhouse
```

## Transfer To Corporate Laptop

Copy whole `dist\OfflineMeetingTranscriber` folder. Do not copy only `.exe`; `_internal`, `models`, and `config.json` are required.

Run:

```text
Run-OfflineMeetingTranscriber.bat
```

Output transcript writes to:

```text
OfflineMeetingTranscriber\transcripts\meeting_transcript.txt
```

## Notes

- Runtime sets Hugging Face/Transformers offline environment flags.
- No admin rights required if corporate policy allows running unsigned local executables.
- If endpoint security blocks unsigned EXEs, IT must allow-list or code-sign `OfflineMeetingTranscriber.exe`.
- If mic level stays zero, check Windows microphone privacy and device selection.
