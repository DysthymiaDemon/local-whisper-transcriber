# OpenVINO Whisper Large V3 Turbo INT8 Implementation Plan

**Goal:** Replace Intel GPU Trial default with `OpenVINO/whisper-large-v3-turbo-int8-ov`.

## Decisions

- Keep model path `models/openvino-whisper` and replace previous Small FP16 model in place.
- Stage and validate new model before replacement.
- Use `models/.openvino-whisper.rollback` during replacement; restore it on failure.
- Require OpenVINO, OpenVINO GenAI, and OpenVINO Tokenizers 2026.1 or newer.
- Use 10-second chunks with 1-second overlap for Intel GPU Trial only.
- Migrate untouched Intel defaults; preserve custom model paths and chunk settings.
- Preserve CPU and NVIDIA behavior.

## Implementation

1. Update Intel backend repository, package floors, INT8 metadata, chunk values, setup text, and report text.
2. Validate OpenVINO XML/BIN files and exact turbo architecture signature from `config.json`.
3. Make runtime checks reject installed OpenVINO packages below 2026.1.
4. Replace stale model through verified staging and rollback-safe folder swap.
5. Update README and embedded model-folder guide.
6. Regenerate both standalone launchers.

## Verification

- Test turbo, Small FP16, malformed, and incomplete model signatures.
- Test older, equal, and newer OpenVINO package versions.
- Test successful replacement, copy failure restoration, and existing rollback-folder abort.
- Test fresh and migrated Intel config plus custom-config preservation.
- Run full unittest suite, `py_compile`, payload-current checks, and `git diff --check`.
- Run final Intel Iris Xe recording acceptance when compatible runtime/model download is available.
