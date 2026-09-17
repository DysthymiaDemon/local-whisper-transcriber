import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from file_transcriber import FileTranscriptionWorker, audio_chunks
from transcriber_engine import EngineConfig


def make_audio(path, seconds=2.2, rate=48000):
    import numpy as np
    samples = (np.sin(np.arange(round(rate * seconds)) * 440 * 2 * np.pi / rate) * 10000).astype("int16")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.column_stack((samples, samples)).tobytes())


class FileTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "meeting.wav"
        make_audio(self.source)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.bin").write_bytes(b"model")
        self.config = EngineConfig(whisper_model_dir=str(self.model), chunk_seconds=1, overlap_seconds=.2,
                                   capture_mode="mixed", system_device_index=None)
        self.events = []
        self.worker = FileTranscriptionWorker(self.config, str(self.source), str(self.root / "out.txt"),
                                               lambda e, p: self.events.append((e, p)))

    def fake_model(self):
        return SimpleNamespace(transcribe=lambda samples, **kwargs: (
            [SimpleNamespace(text="Meeting discussion", start=0, end=len(samples)/16000)], None))

    def test_stereo_resampling_overlap_and_tail(self):
        import av
        with av.open(str(self.source)) as container:
            chunks = list(audio_chunks(container, 1, .2, threading.Event()))
        self.assertEqual([round(start, 1) for start, _ in chunks], [0, .8, 1.6])
        self.assertEqual([len(samples) for _, samples in chunks], [16000, 16000, 9600])
        self.assertTrue(all(samples.ndim == 1 for _, samples in chunks))

    def test_exact_window_does_not_repeat_overlap(self):
        import av
        make_audio(self.source, seconds=1)
        with av.open(str(self.source)) as container:
            self.assertEqual(len(list(audio_chunks(container, 1, .2, threading.Event()))), 1)

    def test_completes_without_capture_devices_and_saves(self):
        with patch("file_transcriber.load_whisper_model", return_value=self.fake_model()):
            self.worker.run()
        self.assertEqual(self.events[-1][1]["outcome"], "completed")
        self.assertIn("Meeting discussion", (self.root / "out.txt").read_text())
        progress = [p for e, p in self.events if e == "file_progress"]
        self.assertAlmostEqual(progress[-1]["processed"], 2.2)
        self.assertEqual(self.config.capture_mode, "mixed")

    def test_cancel_finishes_current_chunk(self):
        def transcribe(samples, **kwargs):
            self.worker.cancel()
            return [SimpleNamespace(text="Finished current chunk", start=0, end=1)], None
        with patch("file_transcriber.load_whisper_model", return_value=SimpleNamespace(transcribe=transcribe)):
            self.worker.run()
        self.assertEqual(self.events[-1][1]["outcome"], "cancelled")
        self.assertEqual(len([e for e, _ in self.events if e == "file_progress"]), 1)
        self.assertIn("Finished current chunk", (self.root / "out.txt").read_text())

    def test_model_failure_and_corrupt_input_finish(self):
        with patch("file_transcriber.load_whisper_model", side_effect=RuntimeError("bad model")):
            self.worker.run()
        self.assertEqual(self.events[-1][1]["outcome"], "failed")
        self.source.write_bytes(b"not media")
        self.worker.run()
        self.assertEqual(self.events[-1][1]["outcome"], "failed")

    def test_missing_audio(self):
        container = SimpleNamespace(streams=SimpleNamespace(audio=[]))
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = container
        with patch("av.open", return_value=context):
            self.worker.run()
        self.assertTrue(any("no audio track" in p.get("message", "") for _, p in self.events))
        self.assertEqual(self.events[-1][1]["outcome"], "failed")

    def test_write_error_recovers_partial(self):
        with patch("file_transcriber.load_whisper_model", return_value=self.fake_model()), patch(
            "file_transcriber.AtomicTranscriptWriter.refresh", side_effect=OSError("locked")
        ):
            self.worker.run()
        result = self.events[-1][1]
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("recovery", result["output_path"])
        self.assertIn("Meeting discussion", Path(result["output_path"]).read_text())

    def test_source_cannot_be_output(self):
        original = self.source.read_bytes()
        self.worker.output_path = self.source
        self.worker.run()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.events[-1][1]["outcome"], "failed")

    def test_model_uses_local_only_and_offline_environment(self):
        import os
        from transcriber_engine import load_whisper_model
        constructor = Mock()
        with patch.dict(os.environ, {}, clear=False), patch.dict(
            sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=constructor)}
        ):
            load_whisper_model(self.config)
            self.assertTrue(constructor.call_args.kwargs["local_files_only"])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_missing_model_does_not_load_or_download(self):
        self.worker.config = replace(self.worker.config, whisper_model_dir=str(self.root / "missing"))
        with patch("file_transcriber.load_whisper_model") as loader:
            self.worker.run()
            loader.assert_not_called()
        self.assertEqual(self.events[-1][1]["outcome"], "failed")
        self.assertTrue(any("setup" in p.get("message", "") for _, p in self.events))


if __name__ == "__main__":
    unittest.main()
