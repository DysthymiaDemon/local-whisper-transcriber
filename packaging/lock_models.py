"""Pin the speech models setup downloads to one Hugging Face commit and record every file's SHA-256.

Run on a developer machine (network needed) when moving to a newer model revision:

    py -3.12 packaging/lock_models.py

It writes resources/model_manifest.json. Setup downloads exactly that commit and refuses any file whose
SHA-256 differs, so a later change on the Hub (or in transit) cannot swap the model. Review the diff before
committing: the manifest records what the Hub served today, which is only as trustworthy as that source.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "resources" / "model_manifest.json"
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import bootstrap_launcher  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_repo(spec, download_dir: Path) -> dict[str, object]:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(spec.repo_id, files_metadata=True, token=False)
    siblings = list(
        filter_repo_objects(
            info.siblings or [],
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
            key=lambda sibling: sibling.rfilename,
        )
    )
    files: dict[str, str] = {}
    for sibling in sorted(siblings, key=lambda item: item.rfilename):
        lfs = getattr(sibling, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if not sha256:
            # Small files are stored in git, not LFS, so the Hub has no SHA-256 for them: hash the real bytes.
            local = hf_hub_download(
                spec.repo_id, sibling.rfilename, revision=info.sha, token=False, local_dir=str(download_dir)
            )
            sha256 = file_sha256(Path(local))
        files[sibling.rfilename] = sha256
    if not files:
        raise SystemExit(f"No files matched for {spec.repo_id}")
    return {"revision": info.sha, "files": files}


def main() -> int:
    specs = bootstrap_launcher.DEFAULT_MODEL_DOWNLOADS + bootstrap_launcher.OPENVINO_MODEL_DOWNLOADS
    manifest: dict[str, object] = {}
    with bootstrap_launcher.online_huggingface_download_env(PROJECT_ROOT):
        for spec in specs:
            with tempfile.TemporaryDirectory(prefix="ml") as tmp:
                manifest[spec.repo_id] = lock_repo(spec, Path(tmp))
            entry = manifest[spec.repo_id]
            print(f"{spec.repo_id} @ {entry['revision']}: {len(entry['files'])} files")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
