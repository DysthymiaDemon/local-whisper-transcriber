# Portable Windows build

The portable build is a folder that runs on Windows 11 Enterprise without Python installed and without admin rights. It bundles the Python runtime and all packages. The user starts `OfflineMeetingTranscriber.exe`.

It uses PyInstaller `onedir`, not `onefile`. Qt and CTranslate2 are large, and corporate endpoint tools often slow down or block one-file extraction.

Managed laptops block `.cmd` and `.bat` files, so the bundle has no batch launcher. The EXE switches itself to offline mode once setup is complete.

## Folder layout

```text
OfflineMeetingTranscriber/
  OfflineMeetingTranscriber.exe
  config.json
  _internal/
  models/
    faster-whisper/
  transcripts/
  offline_setup.md
```

## Build on an internet-connected Windows machine

Use the same Windows architecture as the target laptop and 64-bit Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\resources\requirements-build.txt
.\packaging\build_portable.ps1 -Clean
```

To build from a local wheelhouse:

```powershell
mkdir wheelhouse
py -3.12 -m pip download -r .\resources\requirements-build.txt -d wheelhouse
.\packaging\build_portable.ps1 -Clean -Wheelhouse .\wheelhouse
```

## Add the model

The first launch can download the default model if the laptop is online during setup. For an air-gapped laptop, copy a complete model folder into the bundle before transfer (see [offline_setup.md](offline_setup.md#3-download-the-speech-model-on-an-internet-connected-machine)):

```text
dist\OfflineMeetingTranscriber\models\faster-whisper
```

Keep the model folder beside the EXE, not inside `_internal`.

## Transfer to the laptop

Copy the whole `dist\OfflineMeetingTranscriber` folder, not just the `.exe`. The `_internal`, `models` and `config.json` items are all required.

## Notes

- If endpoint security blocks unsigned EXEs, IT must sign or allow-list `OfflineMeetingTranscriber.exe`.
- If the microphone level stays at zero, check Windows microphone privacy settings and the selected device.
