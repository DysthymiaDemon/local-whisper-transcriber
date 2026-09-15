import base64
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "packaging" / "build_standalone_pyw.py"
LAUNCHER_PATH = PROJECT_ROOT / "Open Offline Meeting Transcriber.pyw"
GPU_TRIAL_LAUNCHER_PATH = PROJECT_ROOT / "Open Offline Meeting Transcriber GPU Trial.pyw"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_standalone_pyw", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandaloneLauncherTests(unittest.TestCase):
    def test_payload_contains_required_app_and_resource_files(self):
        generator = load_generator()

        payload = generator.build_payload(PROJECT_ROOT)
        files = payload["files"]

        self.assertIn("app/bootstrap_launcher.py", files)
        self.assertIn("app/meeting_transcriber_gui.py", files)
        self.assertIn("app/transcriber_engine.py", files)
        self.assertIn("app/ui_helpers.py", files)
        self.assertIn("resources/requirements.txt", files)
        self.assertIn("resources/config.template.json", files)

    def test_committed_launcher_payload_is_current(self):
        generator = load_generator()

        expected = generator.generate(PROJECT_ROOT)
        actual = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertEqual(actual, expected)

    def test_committed_gpu_trial_launcher_payload_is_current(self):
        generator = load_generator()

        expected = generator.generate_gpu_trial(PROJECT_ROOT)
        actual = GPU_TRIAL_LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertEqual(actual, expected)

    def test_launcher_sets_log_root_to_copied_file_folder(self):
        content = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertIn('os.environ["LOCAL_WHISPER_LOG_ROOT"] = str(launcher_path.parent)', content)

    def test_gpu_trial_launcher_uses_separate_folder_and_env_flag(self):
        content = GPU_TRIAL_LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertIn('APP_FOLDER_NAME = "OfflineMeetingTranscriberGpuTrial"', content)
        self.assertIn('os.environ["LOCAL_WHISPER_GPU_TRIAL"] = "1"', content)
        self.assertIn('os.environ["LOCAL_WHISPER_RUNTIME_APP_FOLDER_NAME"] = "OfflineMeetingTranscriberGpuTrialRuntime"', content)
        self.assertNotIn('APP_FOLDER_NAME = "OfflineMeetingTranscriber"\nPAYLOAD', content)

    def test_gpu_trial_launcher_embeds_large_v3_turbo_int8_model(self):
        namespace = {"__name__": "embedded_gpu_trial_launcher"}
        exec(GPU_TRIAL_LAUNCHER_PATH.read_text(encoding="utf-8"), namespace)
        payload = namespace["decode_payload"]()
        encoded = payload["files"]["app/bootstrap_launcher.py"]["data"]
        content = base64.b64decode(encoded).decode("utf-8")

        self.assertIn("OpenVINO/whisper-large-v3-turbo-int8-ov", content)
        self.assertIn('"compute_type": "int8"', content)
        self.assertIn('"chunk_seconds": 10.0', content)
        self.assertNotIn("OpenVINO/whisper-small-fp16-ov", content)

    def test_launcher_does_not_force_huggingface_offline_before_bootstrap(self):
        content = LAUNCHER_PATH.read_text(encoding="utf-8")
        before_import = content.split("from bootstrap_launcher import run_bootstrap", maxsplit=1)[0]

        self.assertNotIn("HF_HUB_OFFLINE", before_import)
        self.assertNotIn("TRANSFORMERS_OFFLINE", before_import)
        self.assertNotIn("HF_DATASETS_OFFLINE", before_import)

    def test_runtime_launch_still_forces_huggingface_offline(self):
        content = (PROJECT_ROOT / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('os.environ.setdefault("HF_HUB_OFFLINE", "1")', content)
        self.assertIn('os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")', content)
        self.assertIn('os.environ.setdefault("HF_DATASETS_OFFLINE", "1")', content)

    def test_copy_buttons_show_temporary_copied_feedback(self):
        content = (PROJECT_ROOT / "app" / "meeting_transcriber_gui.py").read_text(encoding="utf-8")

        self.assertIn("self.copy_button.clicked.connect(lambda: self._copy_transcript(self.copy_button))", content)
        self.assertIn(
            "self.copy_footer_button.clicked.connect(lambda: self._copy_transcript(self.copy_footer_button))",
            content,
        )
        self.assertIn('button.setText("✓ Copied!")', content)
        self.assertIn('button.setStyleSheet("QPushButton { color: #2e7d32; }")', content)
        self.assertIn("QTimer.singleShot(1500, restore)", content)
        self.assertIn("self._copy_feedback_tokens", content)

    def test_gui_keeps_transcription_draining_on_stop_and_close(self):
        content = (PROJECT_ROOT / "app" / "meeting_transcriber_gui.py").read_text(encoding="utf-8")

        self.assertIn('self.status_label.setText("Finishing transcription")', content)
        self.assertIn('self._append_log("Finishing transcription before closing. Wait for Stopped.")', content)
        self.assertIn("event.ignore()", content)
        self.assertNotIn("engine.stop()\n        event.accept()", content)

    def test_init_only_extracts_beside_copied_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            copied_launcher = folder / LAUNCHER_PATH.name
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            app_root = folder / "OfflineMeetingTranscriber"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), app_root.resolve())
            self.assertTrue((app_root / "app" / "bootstrap_launcher.py").is_file())
            self.assertTrue((app_root / "resources" / "requirements.txt").is_file())
            self.assertTrue((app_root / "config.json").is_file())
            self.assertTrue((app_root / "models" / "faster-whisper").is_dir())
            self.assertFalse((app_root / "models" / "speechbrain-ecapa").exists())
            self.assertTrue((app_root / "transcripts").is_dir())

    def test_gpu_trial_init_only_extracts_beside_copied_launcher_to_trial_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            copied_launcher = folder / GPU_TRIAL_LAUNCHER_PATH.name
            shutil.copy2(GPU_TRIAL_LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            app_root = folder / "OfflineMeetingTranscriberGpuTrial"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), app_root.resolve())
            self.assertTrue((app_root / "app" / "bootstrap_launcher.py").is_file())
            self.assertTrue((app_root / "resources" / "requirements.txt").is_file())
            self.assertTrue((app_root / "config.json").is_file())

    def test_init_only_uses_current_folder_when_named_like_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp) / "OfflineMeetingTranscriber"
            app_root.mkdir()
            copied_launcher = app_root / LAUNCHER_PATH.name
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), app_root.resolve())
            self.assertTrue((app_root / "app" / "bootstrap_launcher.py").is_file())

    def test_init_only_honors_install_root_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            copied_launcher = folder / LAUNCHER_PATH.name
            target_root = folder / "custom-install"
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only", "--install-root", str(target_root)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), target_root.resolve())
            self.assertTrue((target_root / "resources" / "config.template.json").is_file())


if __name__ == "__main__":
    unittest.main()
