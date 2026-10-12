# Meeting Scribe

Meeting Scribe is a Windows desktop app that transcribes meetings on your own laptop. It can record a meeting live from the microphone, the system audio, or both, and it can transcribe a meeting recording you already have.

It is built for managed corporate Windows 11 laptops:

- The internet is used only while setup installs Python packages and downloads the speech model.
- After setup, the app never uses the network. Audio, transcripts and models stay on the laptop.
- Everything you launch is plain Python (`.pyw`). The app does not need `.cmd`, `.bat` or PowerShell scripts, which managed laptops usually block.
- No admin rights are needed.

## Quick start

You need Python 3.12 (64-bit). On a managed laptop, install it from the company software portal.

1. Copy `Open Offline Meeting Transcriber.pyw` to a folder of your choice.
2. Double-click it. It creates an `OfflineMeetingTranscriber` folder next to itself.
3. The setup window lists what it will install. Click **Install**.
4. Setup installs the Python packages and downloads the speech model (about 1.5 GB). This is the only step that uses the internet.
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

## Speech model

The default model is [`distil-whisper/distil-large-v3.5-ct2`](https://huggingface.co/distil-whisper/distil-large-v3.5). It is English-only, MIT-licensed and about 1.5 GB. It is as accurate as Whisper large-v3 (7.1% average word error rate on short-form audio) but much faster, so it runs on a laptop CPU.

The model folder must contain:

- `model.bin`
- `config.json`
- `tokenizer.json`
- `vocabulary.json` or `vocabulary.txt`
- `preprocessor_config.json` (required for this model, which uses 128 mel bins)

The app refuses to load an incomplete folder, because faster-whisper would otherwise try to download a missing tokenizer from the internet.

Live recording works in 15-second chunks with 1.5 seconds of overlap, so text appears about 15 to 25 seconds behind speech. Whisper pads every chunk to 30 seconds, so each chunk takes about 9 seconds on an i5-1345U laptop CPU, whatever its length. With 10-second chunks the app could not keep up; 15-second chunks leave about 4 seconds of headroom. Saved recordings use 28-second windows. A 15-minute meeting takes about 6 minutes to transcribe.

### Upgrading from the older model

Earlier versions used `Systran/faster-whisper-small.en`. When you launch the new version, setup offers **Upgrade speech model?**:

- **Install** downloads the new model once and replaces the old one. If anything fails, the old model is restored.
- **Use current model** opens the app with the model you already have. Setup asks again on the next launch.

If you set your own model folder in `config.json`, it is never replaced.

## Network policy

| Phase | Network use |
|---|---|
| Setup, after you click **Install** | `pip` downloads the packages in `resources\requirements.txt`. Hugging Face downloads the speech model with no login token. |
| Setup window open, before **Install** | None. |
| Running the app | None. |

How the app stays offline after setup:

- It forces `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` and `HF_HUB_DISABLE_TELEMETRY` to `1`, even if they are set differently on the laptop.
- It loads models with `local_files_only=True` and only from a complete local folder.
- It blocks Python network connections to anything other than this laptop (loopback). An attempt fails with "Network access is disabled after setup".

The Python block cannot stop code in native libraries. If IT wants an operating-system guarantee, an outbound Windows Firewall rule for `pythonw.exe` can block everything after setup.

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

## GPU trial

`Open Offline Meeting Transcriber GPU Trial.pyw` is a separate launcher for testing GPU acceleration. It uses its own `OfflineMeetingTranscriberGpuTrial` folder.

- **Intel GPUs** use OpenVINO GenAI 2026.1 or newer with the 828 MB `OpenVINO/whisper-large-v3-turbo-int8-ov` model.
- **NVIDIA GPUs** use the default model through CUDA, if CUDA and cuDNN are installed.
- **AMD GPUs** only install the DirectML runtime for probing. The transcription engine does not use it yet.

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
- `resources/`: pinned requirements and the config template.
- `packaging/`: the standalone `.pyw` generator and the PyInstaller build.
- `tests/`: unit tests.

The `.pyw` launchers embed `app/` and `resources/`. After changing anything there, regenerate both launchers and commit them:

```powershell
py -3.12 packaging\build_standalone_pyw.py
```

Run the tests:

```powershell
py -3.12 -B -m unittest discover -s tests -v
```
