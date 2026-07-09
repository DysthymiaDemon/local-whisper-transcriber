from __future__ import annotations

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "Open Offline Meeting Transcriber.pyw"
GPU_TRIAL_OUTPUT_FILE = PROJECT_ROOT / "Open Offline Meeting Transcriber GPU Trial.pyw"
PAYLOAD_DIRS = ("app", "resources")
APP_FOLDER_NAME = "OfflineMeetingTranscriber"
GPU_TRIAL_APP_FOLDER_NAME = "OfflineMeetingTranscriberGpuTrial"
GPU_TRIAL_RUNTIME_FOLDER_NAME = "OfflineMeetingTranscriberGpuTrialRuntime"


def _iter_payload_files(root: Path = PROJECT_ROOT) -> list[Path]:
    files: list[Path] = []
    for folder_name in PAYLOAD_DIRS:
        folder = root / folder_name
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    return files


def build_payload(root: Path = PROJECT_ROOT) -> dict[str, object]:
    payload_files = {}
    for path in _iter_payload_files(root):
        rel_path = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        payload_files[rel_path] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    return {"version": 1, "files": payload_files}


def encode_payload(payload: dict[str, object]) -> tuple[str, str]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode("ascii")
    wrapped = "\n".join(textwrap.wrap(encoded, 96))
    return wrapped, digest


def render_launcher(
    payload: dict[str, object],
    app_folder_name: str = APP_FOLDER_NAME,
    gpu_trial: bool = False,
) -> str:
    encoded, digest = encode_payload(payload)
    gpu_trial_env = (
        f'    os.environ["LOCAL_WHISPER_GPU_TRIAL"] = "1"\n'
        f'    os.environ["LOCAL_WHISPER_RUNTIME_APP_FOLDER_NAME"] = "{GPU_TRIAL_RUNTIME_FOLDER_NAME}"\n'
        if gpu_trial
        else ""
    )
    return f'''#! python3.12
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path


APP_FOLDER_NAME = "{app_folder_name}"
PAYLOAD_SHA256 = "{digest}"
PAYLOAD_B64 = """
{encoded}
""".strip()


def default_install_root(launcher_path: Path) -> Path:
    folder = launcher_path.resolve().parent
    if folder.name.lower() == APP_FOLDER_NAME.lower():
        return folder
    return folder / APP_FOLDER_NAME


def decode_payload() -> dict[str, object]:
    compressed = base64.b64decode(PAYLOAD_B64.encode("ascii"))
    actual = hashlib.sha256(compressed).hexdigest()
    if actual != PAYLOAD_SHA256:
        raise RuntimeError("Launcher payload checksum mismatch.")
    return json.loads(zlib.decompress(compressed).decode("utf-8"))


def write_payload_file(root: Path, rel_path: str, encoded: str, expected_sha: str) -> None:
    target = root / Path(rel_path)
    raw = base64.b64decode(encoded.encode("ascii"))
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"Embedded file checksum mismatch: {{rel_path}}")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def ensure_runtime_folders(root: Path) -> None:
    for model in ("faster-whisper",):
        (root / "models" / model).mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    template_path = root / "resources" / "config.template.json"
    if not config_path.exists() and template_path.exists():
        config_path.write_bytes(template_path.read_bytes())


def initialize_app(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = decode_payload()
    files = payload.get("files", {{}})
    if not isinstance(files, dict):
        raise RuntimeError("Launcher payload missing files.")
    for rel_path, file_info in files.items():
        if not isinstance(file_info, dict):
            raise RuntimeError(f"Invalid payload entry: {{rel_path}}")
        write_payload_file(root, rel_path, str(file_info["data"]), str(file_info["sha256"]))
    ensure_runtime_folders(root)
    (root / ".launcher_payload.sha256").write_text(PAYLOAD_SHA256 + "\\n", encoding="utf-8")
    return root


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--install-root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    launcher_path = Path(__file__).resolve()
    install_root = Path(args.install_root).expanduser() if args.install_root else default_install_root(launcher_path)
    install_root = initialize_app(install_root)

    os.environ["LOCAL_WHISPER_APP_ROOT"] = str(install_root)
    os.environ["LOCAL_WHISPER_SOURCE_ROOT"] = str(install_root)
    os.environ["LOCAL_WHISPER_LOG_ROOT"] = str(launcher_path.parent)
{gpu_trial_env}

    app_dir = str(install_root / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    if args.init_only:
        print(install_root)
        return 0

    from bootstrap_launcher import run_bootstrap

    return run_bootstrap()


if __name__ == "__main__":
    raise SystemExit(main())
'''


def generate(root: Path = PROJECT_ROOT) -> str:
    return render_launcher(build_payload(root))


def generate_gpu_trial(root: Path = PROJECT_ROOT) -> str:
    return render_launcher(
        build_payload(root),
        app_folder_name=GPU_TRIAL_APP_FOLDER_NAME,
        gpu_trial=True,
    )


def write_launcher(root: Path = PROJECT_ROOT, output_file: Path | None = None) -> Path:
    target = output_file or root / OUTPUT_FILE.name
    target.write_text(generate(root), encoding="utf-8", newline="\n")
    return target


def write_gpu_trial_launcher(root: Path = PROJECT_ROOT, output_file: Path | None = None) -> Path:
    target = output_file or root / GPU_TRIAL_OUTPUT_FILE.name
    target.write_text(generate_gpu_trial(root), encoding="utf-8", newline="\n")
    return target


def main() -> int:
    target = write_launcher()
    gpu_target = write_gpu_trial_launcher()
    print(f"Wrote {target}")
    print(f"Wrote {gpu_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
