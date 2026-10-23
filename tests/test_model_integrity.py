import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import model_integrity  # noqa: E402
import transcriber_engine  # noqa: E402
from model_integrity import (  # noqa: E402
    MODEL_SOURCE_FILE,
    SETUP_MODEL_REPOS,
    ModelIntegrityError,
    managed_model_repo,
    verify_managed_model,
    write_model_source,
)
from transcriber_engine import EngineConfig, load_whisper_model, verified_model_repo  # noqa: E402

FILES = {"model.bin": b"weights", "config.json": b"{}", "tokenizer.json": b"{}", "vocabulary.json": b"[]"}
CPU_REPO = SETUP_MODEL_REPOS["faster-whisper"]
GPU_REPO = SETUP_MODEL_REPOS["openvino-whisper"]
REVISION = "a" * 40


def write_model(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in FILES.items():
        (folder / name).write_bytes(data)


class ManagedModelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # The default folders setup manages, found by location under the app root.
        self.model = self.root / "models" / "faster-whisper"
        self.gpu_model = self.root / "models" / "openvino-whisper"
        self.custom = self.root / "custom-whisper"
        for folder in (self.model, self.gpu_model, self.custom):
            write_model(folder)
        self.resources = self.root / "resources"
        self.resources.mkdir()
        entry = {"revision": REVISION, "files": {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}}
        manifest = {CPU_REPO: entry, GPU_REPO: entry}
        (self.resources / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for patcher in (
            patch.object(model_integrity, "default_resources_dir", return_value=self.resources),
            patch.dict(os.environ, {"LOCAL_WHISPER_APP_ROOT": str(self.root)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        transcriber_engine.clear_whisper_model_cache()
        self.addCleanup(transcriber_engine.clear_whisper_model_cache)

    def test_setup_folders_are_managed_by_location(self):
        self.assertEqual(managed_model_repo(self.model), CPU_REPO)
        self.assertEqual(managed_model_repo(self.gpu_model), GPU_REPO)
        self.assertIsNone(managed_model_repo(self.custom))

    def test_custom_folder_is_not_checked(self):
        self.assertIsNone(verify_managed_model(self.custom))
        write_model_source(self.custom, "someone/else")
        self.assertIsNone(verify_managed_model(self.custom))

    def test_matching_managed_folder_passes(self):
        write_model_source(self.model, CPU_REPO, REVISION)
        self.assertEqual(verify_managed_model(self.model), CPU_REPO)

    def test_default_folder_without_marker_is_refused(self):
        with self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
            verify_managed_model(self.model)
        (self.model / MODEL_SOURCE_FILE).write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
            verify_managed_model(self.model)

    def test_default_folder_with_unknown_model_marker_is_refused(self):
        write_model_source(self.model, "someone/else", REVISION)
        with self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
            verify_managed_model(self.model)

    def test_default_folder_with_the_other_models_marker_is_refused(self):
        write_model_source(self.gpu_model, CPU_REPO, REVISION)
        with self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
            verify_managed_model(self.gpu_model)
        write_model_source(self.model, GPU_REPO, REVISION)
        with self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
            verify_managed_model(self.model)

    def test_default_folder_with_another_or_no_revision_is_refused(self):
        for revision in ("b" * 40, ""):
            write_model_source(self.model, CPU_REPO, revision)
            with self.subTest(revision=revision), self.assertRaisesRegex(ModelIntegrityError, "no setup record"):
                verify_managed_model(self.model)

    def test_changed_or_missing_file_is_refused(self):
        write_model_source(self.model, CPU_REPO, REVISION)
        (self.model / "model.bin").write_bytes(b"weightz")
        with self.assertRaisesRegex(ModelIntegrityError, "changed: model.bin"):
            verify_managed_model(self.model)
        (self.model / "config.json").unlink()
        with self.assertRaisesRegex(ModelIntegrityError, "missing: config.json"):
            verify_managed_model(self.model)

    def test_managed_folder_without_readable_manifest_is_refused(self):
        write_model_source(self.model, CPU_REPO, REVISION)
        (self.resources / "model_manifest.json").unlink()
        with self.assertRaises(ModelIntegrityError):
            verify_managed_model(self.model)

    def test_default_folder_without_marker_is_refused_even_without_a_manifest_entry(self):
        (self.resources / "model_manifest.json").write_text(json.dumps({}), encoding="utf-8")
        with self.assertRaises(ModelIntegrityError):
            verify_managed_model(self.model)

    def fake_whisper(self, calls):
        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                calls.append(args)

        return patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)})

    def test_engine_checks_on_every_fresh_load_but_not_for_a_cached_model(self):
        write_model_source(self.model, CPU_REPO, REVISION)
        config = EngineConfig(whisper_model_dir=str(self.model))
        calls = []
        with self.fake_whisper(calls), patch.object(
            model_integrity, "verify_files", wraps=model_integrity.verify_files
        ) as verify:
            load_whisper_model(config)
            load_whisper_model(config)
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(verified_model_repo(config), CPU_REPO)
            transcriber_engine.unload_whisper_model(config)
            self.assertIsNone(verified_model_repo(config))
            load_whisper_model(config)
            self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(calls), 2)

    def test_tampered_model_is_never_constructed(self):
        write_model_source(self.model, CPU_REPO, REVISION)
        (self.model / "tokenizer.json").write_bytes(b"{\"swapped\": true}")
        calls = []
        with self.fake_whisper(calls):
            with self.assertRaises(ModelIntegrityError):
                load_whisper_model(EngineConfig(whisper_model_dir=str(self.model)))
        self.assertEqual(calls, [])
        self.assertFalse(transcriber_engine.whisper_model_is_loaded(EngineConfig(whisper_model_dir=str(self.model))))

    def test_default_model_without_marker_is_never_constructed(self):
        calls = []
        with self.fake_whisper(calls):
            with self.assertRaisesRegex(ModelIntegrityError, "Run setup"):
                load_whisper_model(EngineConfig(whisper_model_dir=str(self.model)))
        self.assertEqual(calls, [])
        self.assertFalse(transcriber_engine.whisper_model_is_loaded(EngineConfig(whisper_model_dir=str(self.model))))

    def test_custom_model_without_marker_loads_as_unverified(self):
        config = EngineConfig(whisper_model_dir=str(self.custom))
        calls = []
        with self.fake_whisper(calls):
            load_whisper_model(config)
        self.assertEqual(len(calls), 1)
        self.assertTrue(transcriber_engine.whisper_model_is_loaded(config))
        self.assertIsNone(verified_model_repo(config))


if __name__ == "__main__":
    unittest.main()
