import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from bootstrap_launcher import (
    BOOTSTRAP_STEPS,
    PipProgressParser,
    PipInstallProgressTracker,
    PipProgressEvent,
    REQUIRED_IMPORTS,
    SETUP_MARKER,
    BootstrapStatus,
    StepDiagnostic,
    StepState,
    build_step_tooltip,
    ensure_portable_layout,
    local_runtime_dir,
    local_venv_dir,
    missing_imports,
    missing_runtime_imports,
    model_folder_status,
    needs_setup,
    write_setup_error_log,
)


class BootstrapLauncherTests(unittest.TestCase):
    def test_missing_imports_reports_unavailable_modules(self):
        result = missing_imports(
            {
                "ok": "json",
                "missing": "definitely_missing_local_transcriber_module",
            }
        )

        self.assertEqual(result, ["missing"])

    def test_required_imports_cover_runtime_stacks(self):
        self.assertIn("PySide6", REQUIRED_IMPORTS)
        self.assertIn("sounddevice", REQUIRED_IMPORTS)
        self.assertIn("faster-whisper", REQUIRED_IMPORTS)
        self.assertIn("pyannote.audio", REQUIRED_IMPORTS)
        self.assertIn("torch", REQUIRED_IMPORTS)

    def test_bootstrap_steps_cover_setup_flow(self):
        self.assertEqual(BOOTSTRAP_STEPS[0], "Checking Python runtime")
        self.assertIn("Installing Python packages", BOOTSTRAP_STEPS)
        self.assertEqual(BOOTSTRAP_STEPS[-1], "Launching transcriber")

    def test_ensure_portable_layout_creates_config_and_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config_path = ensure_portable_layout(root)

            self.assertTrue(config_path.exists())
            self.assertTrue((root / ".runtime").is_dir())
            self.assertTrue((root / "models" / "faster-whisper").is_dir())
            self.assertTrue((root / "models" / "pyannote-pipeline").is_dir())
            self.assertTrue((root / "models" / "pyannote-embedding").is_dir())
            self.assertTrue((root / "transcripts").is_dir())
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["whisper_model_dir"], "models/faster-whisper")
        self.assertEqual(config["output_file"], "transcripts/meeting_transcript.txt")

    def test_local_runtime_paths_stay_under_install_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(local_runtime_dir(root), root / ".runtime")
            self.assertEqual(local_venv_dir(root), root / ".runtime" / "venv")

    def test_missing_runtime_imports_requires_local_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(missing_runtime_imports(root, {"json": "json"}), ["json"])

    def test_ensure_portable_layout_uses_external_template_root(self):
        with tempfile.TemporaryDirectory() as runtime_tmp, tempfile.TemporaryDirectory() as source_tmp:
            runtime_root = Path(runtime_tmp)
            source_root = Path(source_tmp)
            (source_root / "config.template.json").write_text(
                json.dumps({"chunk_seconds": 10, "output_file": "transcripts/custom.txt"}),
                encoding="utf-8",
            )

            config_path = ensure_portable_layout(runtime_root, source_root)
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["chunk_seconds"], 10)
        self.assertEqual(config["output_file"], "transcripts/custom.txt")

    def test_model_folder_status_requires_nonempty_model_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            (root / "models" / "faster-whisper" / "model.bin").write_bytes(b"model")
            (root / "models" / "pyannote-pipeline" / "config.yaml").write_text("pipeline", encoding="utf-8")

            status = model_folder_status(root)

        self.assertEqual(status["faster-whisper"], BootstrapStatus.READY)
        self.assertEqual(status["pyannote-pipeline"], BootstrapStatus.READY)
        self.assertEqual(status["pyannote-embedding"], BootstrapStatus.MISSING)

    def test_needs_setup_uses_marker_after_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)

            self.assertTrue(needs_setup(root, {"json": "json"}))
            (root / SETUP_MARKER).write_text("complete\n", encoding="utf-8")

            with patch("bootstrap_launcher.local_venv_python", return_value=Path(sys.executable)):
                self.assertFalse(needs_setup(root, {"json": "json"}))

    def test_pip_progress_parser_reports_download_and_install_progress(self):
        parser = PipProgressParser()

        collect = parser.parse("Collecting PySide6==6.11.1")
        download = parser.parse("Downloading pyside6-6.11.1-cp310-abi3-win_amd64.whl (578 kB)")
        installing = parser.parse("Installing collected packages: PySide6, sounddevice, torch")
        installed_one = parser.parse("Successfully installed PySide6-6.11.1 sounddevice-0.5.5 torch-2.11.0")

        self.assertEqual(collect.phase, "Resolving package")
        self.assertEqual(collect.detail, "PySide6==6.11.1")
        self.assertEqual(download.phase, "Downloading package")
        self.assertIn("pyside6", download.detail)
        self.assertEqual(installing.phase, "Installing packages")
        self.assertEqual(installing.progress_percent, 0)
        self.assertEqual(installed_one.phase, "Install complete")
        self.assertEqual(installed_one.progress_percent, 100)

    def test_pip_progress_parser_counts_already_satisfied(self):
        parser = PipProgressParser()

        event = parser.parse("Requirement already satisfied: numpy==2.4.2 in c:\\site-packages")

        self.assertEqual(event.phase, "Already installed")
        self.assertEqual(event.detail, "numpy==2.4.2")
        self.assertEqual(event.progress_percent, 100)

    def test_step_tooltip_includes_error_command_and_last_output(self):
        diagnostic = StepDiagnostic(
            name="Installing Python packages",
            state=StepState.ERROR,
            detail="pip failed",
            command="python -m pip install",
            last_output="ERROR: no matching distribution",
            error="exit code 1",
            start_time=100.0,
            end_time=145.0,
        )

        tooltip = build_step_tooltip(diagnostic, now=150.0)

        self.assertIn("Installing Python packages", tooltip)
        self.assertIn("Status: error", tooltip)
        self.assertIn("Elapsed: 45s", tooltip)
        self.assertIn("Command: python -m pip install", tooltip)
        self.assertIn("Last output: ERROR: no matching distribution", tooltip)
        self.assertIn("Error: exit code 1", tooltip)

    def test_pip_progress_tracker_reports_elapsed_and_eta(self):
        tracker = PipInstallProgressTracker(start_time=100.0)

        tracker.record(PipProgressEvent("Resolving package", "PySide6"), now=110.0)
        tracker.record(PipProgressEvent("Downloading package", "PySide6 wheel"), now=120.0)
        progress, detail = tracker.record(PipProgressEvent("Running pip", "Installing"), now=130.0)

        self.assertGreater(progress, 0)
        self.assertIn("Elapsed 30s", detail)
        self.assertIn("ETA", detail)

    def test_write_setup_error_log_records_command_and_recent_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_setup_error_log(
                Path(tmp),
                "Installing Python packages",
                "pip failed with exit code 1",
                "python -m pip install -r requirements.txt",
                ["line one", "line two"],
            )
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("context: Installing Python packages", content)
        self.assertIn("pip failed with exit code 1", content)
        self.assertIn("python -m pip install -r requirements.txt", content)
        self.assertIn("line two", content)


if __name__ == "__main__":
    unittest.main()
