# Transcription Hallucination Findings

## Summary

Observed issue in `C:\Users\p1360560\Test`: recorded audio produced mixed English and Malay/Indonesian-like transcript lines, despite user not speaking Malay.

Most likely root cause:

- `faster-whisper-small` is multilingual.
- App passes `language=None`, so Whisper auto-detects language per chunk.
- Input audio quality is poor/noisy or near-silent.
- VAD path is unreliable in current Test runtime because `onnxruntime` fails to import.
- Whisper auto-detect drifts on low-confidence chunks, causing hallucinated mixed-language text.

## Evidence

- Config in `C:\Users\p1360560\Test\OfflineMeetingTranscriber\config.json` has no language override.
- Code passes `language=self.config.language`; default is `None`.
- Test transcript contains multiple Malay/Indonesian-like lines, including `Saya`, `Untuk`, `Jadi`, and `Kalau`.
- Earlier mic probes showed default mic callbacks arriving, but signal was near silence:
  - `max_rms ~0.000015`
  - `max_abs ~0.000031`
- Direct runtime import check showed:
  - `onnxruntime IMPORT_ERROR DLL load failed while importing onnxruntime_pybind11_state`
- Code currently uses `vad_filter=True`, which depends on a working VAD backend.

## Why This Happens

Whisper is not translating intentionally. It is hallucinating under weak/noisy input.

With `language=None`, Whisper attempts language detection. On short, noisy, or low-confidence audio chunks, detection can drift to another language. Once it drifts, decoded text can look like Malay/Indonesian even if the user spoke English.

The diarization output also created many speakers for short/noisy segments. That supports the same diagnosis: audio/chunk quality is poor enough that downstream models are unstable.

## Recommended Fix

1. Pin transcription language to English by default:
   - Set `EngineConfig.language = "en"` or pass `language="en"` from GUI/config.

2. Handle broken VAD explicitly:
   - Detect `onnxruntime` import/load failure before recording starts.
   - Either disable VAD with a clear warning or fix the missing DLL dependency.

3. Filter low-confidence/no-speech outputs:
   - Drop segments when Whisper reports weak probability or empty/no-speech-like output.
   - Log `No speech detected` instead of adding hallucinated rows.

4. Improve input device defaults:
   - Prefer explicit WASAPI microphone device when available.
   - Avoid generic MME `Default input` when it produces near-silent audio.

## Current Status

No code fix applied in this findings document. Diagnosis only.
