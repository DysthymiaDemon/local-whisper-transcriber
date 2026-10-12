# Meeting Scribe

Meeting Scribe is a Windows desktop app that transcribes meetings on your own laptop. It can record a meeting live from the microphone, the system audio, or both, and it can transcribe a meeting recording you already have.

It is built for managed corporate Windows 11 laptops:

- The internet is used only while setup installs Python packages and downloads the speech model.
- After setup, the app never uses the network. Audio, transcripts and models stay on the laptop.
- Everything you launch is plain Python (`.pyw`). The app does not need `.cmd`, `.bat` or PowerShell scripts, which managed laptops usually block.
- No admin rights are needed.

## Quick start

You need Python 3.12 (64-bit). On a managed laptop, install it from the company software portal.

1. Copy `Open Offline Meeting Transcriber.pyw` to a folder of your choice. If you received the file from someone else rather than from this repository, check it first (see [Checking the launcher](#checking-the-launcher)).
2. Double-click it. It creates an `OfflineMeetingTranscriber` folder next to itself.
3. The setup window lists what it will install. Click **Install**.
4. Setup installs the Python packages and downloads the speech model: 828 MB on Intel GPU laptops, 1.5 GB otherwise. This is the only step that uses the internet.
5. When setup finishes, Meeting Scribe opens. Later launches open it straight away.

## Using the app

The home screen has three buttons:

- **Record and Transcribe This Meeting** opens the recording window. Choose the capture mode and microphone, then click **Record**. Text appears as the meeting goes on. **Pause** pauses capture. **Stop** finishes the last chunk and saves the transcript. **Copy** copies the transcript to the clipboard.
- **Transcribe Existing Meeting** asks for a recording (`.mp4`, `.m4a`, `.mp3`, `.wav`, `.webm`, `.mkv`, `.mov`, `.aac` or `.wma`) and where to save the text. The recording is decoded locally and never changed. **Stop** saves what has been transcribed so far.
- **Close This App** closes Meeting Scribe. If a transcription is running, it finishes safely first.

Each recording is saved to its own file, so earlier meetings are never overwritten:

```text
OfflineMeetingTranscriber\transcripts\meeting_transcript_2026-10-12_093015.txt
```

Transcript lines look like this:

```text
[00:00:12] Hello team.
[00:00:18] Let's start with actions.
```

### Capture modes

- **Microphone**: your own microphone.
- **System audio (loopback)**: what the laptop plays, for example the other people in a Teams call. This needs a WASAPI output device.
- **Mixed**: microphone and system audio together.

### Advanced settings

The **Advanced** section in the recording window holds the model folder and the transcript file name. Most people never need to change them.

## Speech engine

Meeting Scribe picks the speech engine for the laptop when setup runs.

| Laptop | Engine | Model | Download |
|---|---|---|---|
| Intel GPU (Iris Xe, Arc) | OpenVINO on the GPU | [`OpenVINO/whisper-large-v3-turbo-int8-ov`](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov) | 828 MB model plus OpenVINO packages |
| Anything else | faster-whisper on the CPU | [`distil-whisper/distil-large-v3.5-ct2`](https://huggingface.co/distil-whisper/distil-large-v3.5) | 1.5 GB model |

Results on the target i5-1345U laptop with the 15-minute Security Data Platform recording:

| | Intel GPU (large-v3-turbo) | CPU (distil-large-v3.5) |
|---|---|---|
| Whole recording | 107 s | 346 s |
| Live chunk | about 2.5 s per 8 s chunk | about 9 s per 15 s chunk |
| Live chunk size | 8 s with 1 s overlap | 15 s with 1.5 s overlap |
| Text delay while recording | about 13 s | about 15 to 25 s |

The GPU engine was also more accurate on that meeting, with better punctuation and names. It leaves the CPU free for Teams.

Whisper pads every chunk to 30 seconds, so a chunk costs about the same whatever its length. That is why the CPU engine needs longer chunks to keep up.

### Start-up

The app loads the speech model in the background as soon as it opens. The home screen shows "Speech model ready" when it is done, so Record starts transcribing at once.

The first start on the GPU compiles the model, which takes about 30 seconds. The compiled model is cached in `models\openvino-whisper\.ov_cache`, so later starts take a few seconds.

If the GPU driver fails, the OpenVINO engine runs the same model on the CPU, and the home screen says so.

To force the CPU engine on an Intel laptop, set the environment variable `LOCAL_WHISPER_FORCE_CPU=1` before setup.

### Required model files

Setup checks that the model folder is complete. The app never loads a partial folder, because faster-whisper would otherwise download a missing tokenizer from the internet.

- **CPU model:** `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json` or `vocabulary.txt`, and `preprocessor_config.json`.
- **GPU model:** the OpenVINO encoder, decoder, tokenizer and detokenizer `.xml` and `.bin` files.

### Upgrading an existing install

When you launch the new version on an existing install, setup can show one of two prompts:

- **Use your Intel GPU?** appears on Intel laptops still on the CPU engine.
- **Upgrade speech model?** appears on other laptops still on the older `Systran/faster-whisper-small.en`.

Both prompts offer the same two choices:

- **Install** downloads the new engine or model once, then switches. If a model download fails, the old model is restored.
- **Use current model** opens the app with what you have, fully offline. Setup asks again on the next launch.

If you set your own model folder in `config.json`, it is never replaced.

## Network policy

| Phase | Network use |
|---|---|
| Setup, after you click **Install** | `pip` downloads the exact wheels listed in `resources\requirements.lock.txt`, plus `resources\requirements-openvino.lock.txt` on Intel GPU laptops. Hugging Face downloads the speech model at a pinned commit, with no login token. |
| Setup window open, before **Install** | None. |
| Running the app | None. |

How the app stays offline after setup:

- It forces `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` and `HF_HUB_DISABLE_TELEMETRY` to `1`, even if they are set differently on the laptop.
- It loads models with `local_files_only=True` and only from a complete local folder.
- It blocks Python network connections to anything other than this laptop (loopback). An attempt fails with "Network access is disabled after setup".

- It decodes saved recordings in a separate process that cannot start other programs, is capped at 2 GB of memory, and only lets FFmpeg read local files.

The Python block cannot stop code in native libraries. If IT wants an operating-system guarantee, an outbound Windows Firewall rule for `pythonw.exe` can block everything after setup.

### What setup checks

- **Packages:** every package, including dependencies, is pinned to one version and one SHA-256 hash in the `.lock.txt` files. pip installs with `--require-hashes --only-binary=:all: --no-deps`, so it refuses a changed file, anything built from source, and any package not in the lock. OpenVINO's telemetry package (`openvino-telemetry`) is deliberately left out. Only OpenVINO's model converter uses it, and this app never runs the converter. Setup also deletes it from older installs.
- **Models:** `resources\model_manifest.json` pins each model to a Hugging Face commit and lists the SHA-256 of every file. Setup downloads that commit and refuses a download whose files do not match. A model folder you set yourself under **Advanced** is not checked, and the log says so.

### Checking the launcher

The launcher's checksum changes with every release:

`Open Offline Meeting Transcriber.pyw` SHA-256: `1a47793a45dfe27c17723dda30208e34525859465e9c5de5a117b3bcb9bd5550`

To check a copy, run this in a terminal in its folder, and compare the result with the line above as published in this repository:

```text
certutil -hashfile "Open Offline Meeting Transcriber.pyw" SHA256
```

The launcher also checks its own embedded files. That check only catches damage, not a deliberately replaced launcher, so get the file from the repository or another channel your IT team approves.

### Corporate HTTPS inspection

Setup trusts the Windows certificate store through `truststore`. If downloads still fail with `CERTIFICATE_VERIFY_FAILED`, IT can place `company-ca.pem`, `corporate-ca.pem` or `ca-bundle.pem` next to `Open Offline Meeting Transcriber.pyw`, or set `LOCAL_WHISPER_CA_BUNDLE`.

## Where files go

```text
<folder with the .pyw>\
  Open Offline Meeting Transcriber.pyw
  error_log.txt              (only after an error)
  install_log.txt            (setup progress)
  OfflineMeetingTranscriber\
    app\                     (extracted app source)
    resources\
    config.json
    models\faster-whisper\
    transcripts\
```

Python packages go to `%LOCALAPPDATA%\OfflineMeetingTranscriberRuntime\site-packages` when the folder is inside OneDrive. Otherwise they go to `OfflineMeetingTranscriber\.runtime\site-packages`. Nothing is installed into the global Python.

When something fails, check `error_log.txt`. It records the error and the file paths involved, but never transcript text.

## Portable build (no Python on the laptop)

A PyInstaller build bundles Python and the packages into one folder. See [docs/PORTABLE_WINDOWS.md](docs/PORTABLE_WINDOWS.md). Endpoint security often blocks unsigned EXEs, so IT may need to sign or allow-list `OfflineMeetingTranscriber.exe`.

## Development

Run from source:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\resources\requirements.txt
python .\app\meeting_transcriber_gui.py
```

Source runtime data defaults to `C:\Users\<you>\Apps\OfflineMeetingTranscriber`. To use another folder, set `LOCAL_WHISPER_APP_ROOT`.

Source layout:

- `app/`: setup, GUI, theme, offline guard, engine and config code.
- `resources/`: requirements, their hash locks, the model manifest and the config template.
- `packaging/`: the standalone `.pyw` generator, the lock generators and the PyInstaller build.
- `tests/`: unit tests.
- `docs/threat-model.md`: the security threat model and what has been fixed.

The `.pyw` launcher embeds `app/` and `resources/`. After changing anything there, regenerate it and commit it. This also updates the checksum in this README:

```powershell
py -3.12 packaging\build_standalone_pyw.py
```

After changing `resources\requirements.txt` or `resources\requirements-openvino.txt`, regenerate the hash locks on a Windows machine with Python 3.12, review the diff, then rebuild the launcher:

```powershell
py -3.12 packaging\lock_requirements.py
```

To move to a newer model revision, regenerate the model manifest (this needs the network), review the diff, then rebuild the launcher:

```powershell
py -3.12 packaging\lock_models.py
```

Run the tests:

```powershell
py -3.12 -B -m unittest discover -s tests -v
```
