"""Approved speech-model revisions and file digests, shared by setup and the speech engine.

resources/model_manifest.json (written by packaging/lock_models.py) pins each setup model to a Hugging Face
commit and records the SHA-256 of every file. Setup checks downloads against it, and the engine checks a
setup-managed model folder again every time it loads the model, so files changed after setup are refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_SOURCE_FILE = "MODEL_SOURCE.json"
MODEL_MANIFEST_FILE = "model_manifest.json"


class ModelIntegrityError(RuntimeError):
    """A setup-managed model folder no longer matches the approved file digests."""


def read_model_source_details(folder: Path) -> tuple[str | None, str | None]:
    """(repo_id, revision) from the MODEL_SOURCE.json marker; revision is None for markers written before pinning."""
    try:
        raw = json.loads((Path(folder) / MODEL_SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(raw, dict):
        return None, None
    repo_id = raw.get("repo_id")
    revision = raw.get("revision")
    return (repo_id if isinstance(repo_id, str) else None, revision if isinstance(revision, str) else None)


def write_model_source(folder: Path, repo_id: str, revision: str = "") -> None:
    marker: dict[str, str] = {"repo_id": repo_id}
    if revision:
        marker["revision"] = revision
    (Path(folder) / MODEL_SOURCE_FILE).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def default_resources_dir() -> Path:
    from app_config import source_root

    base = source_root()
    candidate = base / "resources"
    return candidate if candidate.is_dir() else base


def load_model_manifest(resources_dir: Path | None = None) -> dict[str, Any]:
    path = Path(resources_dir or default_resources_dir()) / MODEL_MANIFEST_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Model manifest is not a JSON object: {path}")
    return raw


def manifest_entry(manifest: dict[str, Any], repo_id: str) -> dict[str, Any]:
    entry = manifest.get(repo_id)
    files = entry.get("files") if isinstance(entry, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("revision"), str) or not isinstance(files, dict) or not files:
        raise RuntimeError(f"No approved revision for {repo_id} in {MODEL_MANIFEST_FILE}; refusing to use it.")
    return entry


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(folder: Path, entry: dict[str, Any]) -> list[str]:
    """Problems found comparing the files in folder with an approved entry; empty when every file matches."""
    problems: list[str] = []
    for rel_path, expected in sorted(entry["files"].items()):
        path = Path(folder) / rel_path
        if not path.is_file():
            problems.append(f"missing: {rel_path}")
        elif file_sha256(path) != str(expected).lower():
            problems.append(f"changed: {rel_path}")
    return problems


def verify_managed_model(folder: Path | str, resources_dir: Path | None = None) -> str | None:
    """Re-check a setup-managed model folder before it is loaded.

    Returns the approved repo id when every file matches, or None for a folder setup does not manage (no marker,
    or a model that is not in the manifest, such as a custom folder chosen under Advanced). Raises
    ModelIntegrityError when a managed folder's files differ from the approved digests.
    """
    repo_id, _revision = read_model_source_details(Path(folder))
    if repo_id is None:
        return None
    try:
        manifest = load_model_manifest(resources_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelIntegrityError(f"The approved model list could not be read, so the model was not loaded: {exc}")
    if repo_id not in manifest:
        return None
    problems = verify_files(Path(folder), manifest_entry(manifest, repo_id))
    if problems:
        shown = "; ".join(problems[:5])
        raise ModelIntegrityError(
            f"Speech model files changed since setup ({shown}). Run setup to restore the approved model."
        )
    return repo_id
