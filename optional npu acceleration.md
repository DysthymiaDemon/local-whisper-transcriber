# Optional NPU Acceleration

## Summary

Newer HP laptops with NPUs may accelerate transcription, but not through the current app stack.

Current app:

- `faster-whisper` + CTranslate2
- CPU path works now
- CUDA path is possible for NVIDIA GPUs
- Intel/Qualcomm NPU path is not supported by this backend

For Intel Core Ultra HP laptops:

- NPU can be used through OpenVINO.
- OpenVINO supports Intel Core Ultra NPU devices.
- OpenVINO release notes mention Whisper pipeline support on NPU.
- App needs a separate OpenVINO backend and OpenVINO-compatible model.

## Constraints

- CTranslate2 model folders cannot run directly on NPU.
- OpenVINO path needs separate packages such as `openvino` and possibly `openvino-genai`.
- OpenVINO path needs separate model export/download.
- NPU is not guaranteed faster for short realtime chunks.
- NPU may still help by reducing CPU load and power use.
- Benchmark must decide final backend.

## Backend Strategy

Always install and keep CPU backend as baseline.

Install-time detection:

1. Detect NVIDIA CUDA.
2. If CUDA available, probe `faster-whisper` with `device="cuda"` and `compute_type="int8_float16"` or `float16`.
3. Detect Intel NPU through `openvino.Core().available_devices` containing `NPU`.
4. If NPU available, download/export OpenVINO Whisper model.
5. Run a 10-20 second benchmark on available backends.
6. Save fastest stable backend.

Saved config options:

- `backend: faster_whisper_cpu`
- `backend: faster_whisper_cuda`
- `backend: openvino_npu`

## Install-Time Probe Plan

CPU:

- Always available.
- Current default.
- Use `faster-whisper` with `device="cpu"` and `compute_type="int8"`.

NVIDIA CUDA:

- Probe CTranslate2 CUDA device count.
- Attempt model load with CUDA.
- Attempt one small transcription benchmark.
- Fall back to CPU on any failure.

Intel NPU:

- Import OpenVINO.
- Check `openvino.Core().available_devices`.
- Require `NPU` in available devices.
- Verify compatible OpenVINO Whisper model exists.
- Attempt one small transcription benchmark.
- Fall back to CPU on any failure.

Qualcomm NPU:

- Separate path from Intel NPU.
- Likely ONNX Runtime QNN Execution Provider.
- Do not mix with Intel OpenVINO plan.
- Treat as future backend after Intel/NVIDIA paths are stable.

## Benchmark Plan

Use same local audio sample across backends.

Measure:

- model load success
- first transcription latency
- steady-state transcription latency
- realtime factor
- memory use where easy to read
- error/fallback reason

Decision:

- Prefer backend that stays below realtime for current chunk settings.
- Prefer CPU if NPU is unstable or only marginally faster.
- Record chosen backend and benchmark result in install log.

## UI Plan

Installer:

- Show detected acceleration:
  - `CPU`
  - `NVIDIA CUDA`
  - `Intel NPU`
- Show benchmark status.
- Show chosen backend.
- If Intel NPU exists but not selected, show reason.

Main app:

- Show `Acceleration: CPU`, `Acceleration: NVIDIA CUDA`, or `Acceleration: Intel NPU`.
- Keep backend read-only unless manual override is added later.

## Implementation Notes

- Keep current CPU backend as fallback.
- Add backend abstraction before adding OpenVINO code.
- Do not break offline/local runtime.
- Do not require admin rights.
- Do not require user to install drivers manually during setup.
- If required runtime is missing, log clear reason and use CPU.

## Sources

- OpenVINO NPU device support: https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO NPU detection via `available_devices`: https://docs.openvino.ai/2024/notebooks/hello-npu-with-output.html
- OpenVINO release notes mention Whisper on NPU: https://docs.openvino.ai/releasenotes
- Qualcomm NPU path: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
