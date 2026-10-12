# Offline and air-gapped setup

Target machine: HP EliteBook 830 G10, Intel Core i5-1345U, 32 GB RAM, Intel Iris Xe, no CUDA.

Normally the first-run setup downloads everything it needs after you click **Install**. Use this guide when the laptop cannot reach PyPI or Hugging Face, even during setup.

The app never uses the network after setup. For details, see the Network policy section in the [README](../README.md#network-policy).

## 1. Python

Use 64-bit Python 3.12 from the company software portal. The `.pyw` launcher needs it to start.

## 2. Build a wheelhouse on an internet-connected machine

Use the same Windows architecture and Python version as the laptop.

```powershell
mkdir wheelhouse
py -3.12 -m pip download -r .\resources\requirements.txt -d wheelhouse
```

On the laptop, install the wheels into the app's private package folder. This is the folder setup would use:

```powershell
py -3.12 -m pip install --no-index --find-links wheelhouse `
  --target "$env:LOCALAPPDATA\OfflineMeetingTranscriberRuntime\site-packages" `
  -r .\resources\requirements.txt
```

If the install folder is not inside OneDrive, use `OfflineMeetingTranscriber\.runtime\site-packages` as the target instead.

## 3. Download the speech model on an internet-connected machine

```powershell
py -3.12 -m pip install huggingface_hub
huggingface-cli download distil-whisper/distil-large-v3.5-ct2 `
  --local-dir faster-whisper `
  --include "model.bin" "config.json" "tokenizer.json" "vocabulary.*" "preprocessor_config.json"
```

Copy the `faster-whisper` folder to the laptop:

```text
<install folder>\OfflineMeetingTranscriber\models\faster-whisper
```

The folder must contain `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json` (or `vocabulary.txt`) and `preprocessor_config.json`. The app will not load a folder with missing files.

Add a `MODEL_SOURCE.json` file so setup does not offer to download the model again:

```json
{
  "repo_id": "distil-whisper/distil-large-v3.5-ct2"
}
```

## 4. Microphone check

```powershell
py -3.12 -c "import sounddevice as sd; print(sd.query_devices())"
```

Windows must allow desktop apps to use the microphone:

```text
Settings > Privacy & security > Microphone
Microphone access: On
Let desktop apps access your microphone: On
```

## 5. Other models

Any CTranslate2 faster-whisper model folder works. Set its path under **Advanced > Model folder** in the recording window, or as `whisper_model_dir` in `config.json`. Setup never replaces a custom model folder.

- Lower latency: `Systran/faster-whisper-small.en` (the previous default, less accurate).
- Multilingual meetings: a large-v3-turbo CTranslate2 conversion. Also set `"language": null` in `config.json`.
