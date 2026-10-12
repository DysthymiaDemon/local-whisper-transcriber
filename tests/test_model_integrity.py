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
from model_integrity import ModelIntegrityError, verify_managed_model, write_model_source  # noqa: E402
from transcriber_engine import EngineConfig, load_whisper_model  # noqa: E402

FILES = {"model.bin": b"weights", "config.json": b"{}", "tokenizer.json": b"{}", "vocabulary.json": b"[]"}


class ManagedModelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        for name, data in FILES.items():
            (self.model / name).write_bytes(data)
        self.resources = self.root / "resources"
        self.resources.mkdir()
        entry = {"revision": "a" * 40, "files": {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}}
        (self.resources / "model_manifest.json").write_text(json.dumps({"org/model": entry}), encoding="utf-8")
        patcher = patch.object(model_integrity, "default_resources_dir", return_value=self.resources)
        patcher.start()
        self.addCleanup(patcher.stop)
        transcriber_engine.clear_whisper_model_cache()
        self.addCleanup(transcriber_engine.clear_whisper_model_cache)

    def test_unmanaged_folder_is_not_checked(self):
        self.assertIsNone(verify_managed_model(self.model))
        write_model_source(self.model, "someone/else")
        self.assertIsNone(verify_managed_model(self.model))

    def test_matching_managed_folder_passes(self):
        write_model_source(self.model, "org/model", "a" * 40)
        self.assertEqual(verify_managed_model(self.model), "org/model")

    def test_changed_or_missing_file_is_refused(self):
        write_model_source(self.model, "org/model", "a" * 40)
        (self.model / "model.bin").write_bytes(b"weightz")
        with self.assertRaisesRegex(ModelIntegrityError, "changed: model.bin"):
            verify_managed_model(self.model)
        (self.model / "config.json").unlink()
        with self.assertRaisesRegex(ModelIntegrityError, "missing: config.json"):
            verify_managed_model(self.model)

    def test_managed_folder_without_readable_manifest_is_refused(self):
        write_model_source(self.model, "org/model", "a" * 40)
        (self.resources / "model_manifest.json").unlink()
        with self.assertRaises(ModelIntegrityError):
            verify_managed_model(self.model)

    def fake_whisper(self, calls):
        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                calls.append(args)

        return patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)})

    def test_engine_checks_on_every_fresh_load_but_not_for_a_cached_model(self):
        write_model_source(self.model, "org/model", "a" * 40)
        config = EngineConfig(whisper_model_dir=str(self.model))
        calls = []
        with self.fake_whisper(calls), patch.dict(os.environ, {}, clear=False), patch.object(
            model_integrity, "verify_files", wraps=model_integrity.verify_files
        ) as verify:
            load_whisper_model(config)
            load_whisper_model(config)
            self.assertEqual(verify.call_count, 1)
            transcriber_engine.unload_whisper_model(config)
            load_whisper_model(config)
            self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(calls), 2)

    def test_tampered_model_is_never_constructed(self):
        write_model_source(self.model, "org/model", "a" * 40)
        (self.model / "tokenizer.json").write_bytes(b"{\"swapped\": true}")
        calls = []
        with self.fake_whisper(calls), patch.dict(os.environ, {}, clear=False):
            with self.assertRaises(ModelIntegrityError):
                load_whisper_model(EngineConfig(whisper_model_dir=str(self.model)))
        self.assertEqual(calls, [])
        self.assertFalse(transcriber_engine.whisper_model_is_loaded(EngineConfig(whisper_model_dir=str(self.model))))


if __name__ == "__main__":
    unittest.main()
