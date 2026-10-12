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
  resources/        (hash locks and model_manifest.json)
  transcripts/
  offline_setup.md
```

## Build on an internet-connected Windows machine

Use the same Windows architecture as the target laptop and 64-bit Python 3.12. The build script creates its own virtual environment. It installs only the exact wheels in `resources\requirements.lock.txt` and `resources\requirements-build.lock.txt`, with `--require-hashes --only-binary=:all: --no-deps`, the same rules setup uses.

```powershell
.\packaging\build_portable.ps1 -Clean
```

To build from a local wheelhouse, download the same locked wheels first:

```powershell
mkdir wheelhouse
py -3.12 -m pip download --require-hashes --only-binary=:all: --no-deps `
  -r .\resources\requirements.lock.txt -r .\resources\requirements-build.lock.txt -d wheelhouse
.\packaging\build_portable.ps1 -Clean -Wheelhouse .\wheelhouse
```

The script copies `resources\` into the bundle. The app needs `model_manifest.json` there to download the model and to check model files every time the model starts.

## Add the model

The first launch can download the default model if the laptop is online during setup. For an air-gapped laptop, copy a complete model folder into the bundle before transfer (see [offline_setup.md](offline_setup.md#3-download-the-speech-model-on-an-internet-connected-machine)):

```text
dist\OfflineMeetingTranscriber\models\faster-whisper
```

Keep the model folder beside the EXE, not inside `_internal`.

## Transfer to the laptop

Copy the whole `dist\OfflineMeetingTranscriber` folder, not just the `.exe`. The `_internal`, `models`, `resources` and `config.json` items are all required. A model copied in by hand needs a `MODEL_SOURCE.json` marker (see offline_setup.md) to be recognised as the approved model and checked on each start.

## Notes

- If endpoint security blocks unsigned EXEs, IT must sign or allow-list `OfflineMeetingTranscriber.exe`.
- The bundle's audio decoder is the same EXE started with `--decode-worker`. Firewall or allow-list rules for the EXE cover it too.
- If the microphone level stays at zero, check Windows microphone privacy settings and the selected device.
