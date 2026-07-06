# Offline Setup

Target machine: HP EliteBook 830 G10, Intel Core i5-1345U, 32GB RAM, Intel Iris Xe, no CUDA.

Runtime is offline-only. `transcriber_engine.py` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`; model paths must point to local folders.

## 1. Prepare Python

Use 64-bit Python 3.12. The source launcher targets Python 3.12, and the no-Python wrapper can install it with `winget` when policy allows.

On offline laptop:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If corporate image only provides another Python version, build wheelhouse with same Python version and architecture.

## 2. Build Wheelhouse On Internet Machine

Use same OS/Python architecture where possible.

```powershell
mkdir wheelhouse
py -3.12 -m pip download -r .\resources\requirements.txt -d wheelhouse
```

For CPU-only PyTorch wheels, use official CPU index if default PyPI resolver selects unsuitable wheels:

```powershell
py -3.12 -m pip download torch==2.11.0 torchaudio==2.11.0 `
  --index-url https://download.pytorch.org/whl/cpu `
  -d wheelhouse
```

Copy `wheelhouse`, `resources`, `app`, launchers, and model folders to offline laptop.

Install offline:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links .\wheelhouse -r .\resources\requirements.txt
```

## 3. Audio Driver Notes

`sounddevice` uses PortAudio. Windows wheels usually include needed binary support, but enterprise images may block microphone access.

Check Windows settings:

```text
Settings > Privacy & security > Microphone
Microphone access: On
Let desktop apps access your microphone: On
```

Test devices:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 4. Download Faster-Whisper Model

Use CTranslate2/faster-whisper model folders, not OpenAI `.pt` files.

Recommended CPU start:

```powershell
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-small `
  --local-dir models\faster-whisper `
  --local-dir-use-symlinks False
```

For lower latency, try `Systran/faster-distil-small.en` if English-only. For better accuracy but slower CPU, try `Systran/faster-whisper-medium`.

Copy folder to offline laptop, example:

```text
<install folder>\models\faster-whisper
```

The first-run setup GUI creates this local install folder beside the standalone `.pyw`, or in the selected portable app folder.

## 5. Download Pyannote Models

Pyannote models may be gated. Accept model terms and use Hugging Face token only on internet-connected machine. Do not copy token to offline laptop.

Download community diarization pipeline:

```powershell
pip install huggingface_hub
huggingface-cli download pyannote/speaker-diarization-community-1 `
  --local-dir models\pyannote-pipeline `
  --local-dir-use-symlinks False `
  --token <HF_TOKEN>
```

Download embedding model:

```powershell
huggingface-cli download pyannote/embedding `
  --local-dir models\pyannote-embedding `
  --local-dir-use-symlinks False `
  --token <HF_TOKEN>
```

If downloaded pipeline `config.yaml` references remote model IDs, edit it on connected machine before transfer so every referenced model uses a local path. Example intent:

```yaml
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
params:
  segmentation: <install folder>/models/pyannote-pipeline/segmentation
  embedding: <install folder>/models/pyannote-embedding
  clustering:
    method: centroid
```

Exact config keys vary by pyannote release. Open the downloaded `config.yaml`, find model references such as `pyannote/...` or `speechbrain/...`, download those folders, and replace each reference with local absolute path. Run this check on offline laptop:

```powershell
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained(r'<install folder>\\models\\pyannote-pipeline'); print('pyannote local load ok')"
```

No line in offline config should require Hugging Face repo lookup.

## 6. Run App

```powershell
.\.venv\Scripts\Activate.ps1
python .\app\meeting_transcriber_gui.py
```

In GUI:

- Select microphone.
- Set local Whisper folder.
- Set local pyannote pipeline folder.
- Set local pyannote embedding folder.
- Choose output `.txt`.
- Press `Record`.

Transcript appears immediately as `Speaker ?`. Diarization may lag on CPU; rows update later to `Speaker 1`, `Speaker 2`, etc. Output file refreshes atomically with latest labels.

## 7. Console Mode Behavior

GUI app also prints live transcript lines to console:

```text
[00:00:12] Speaker ?: We should start with project status.
[speaker update] [00:00:12] Speaker 1: We should start with project status.
```

## 8. Troubleshooting

`Whisper model path does not exist`: set GUI path to local CTranslate2 model folder containing model files.

`Pyannote model load failed`: config still points to remote repo or missing local model file.

No microphone level: confirm Windows microphone privacy settings and correct input device.

Very slow diarization: increase chunk size to 10-15 seconds or use smaller pyannote pipeline if available locally.
