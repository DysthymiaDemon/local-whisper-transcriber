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
import file_transcriber
from file_transcriber import DecodedAudio, FileTranscriptionWorker, pcm_windows
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
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
            (self.model / name).write_bytes(b"model")
        self.config = EngineConfig(whisper_model_dir=str(self.model), chunk_seconds=1, overlap_seconds=.2,
                                   capture_mode="mixed", system_device_index=None)
        self.events = []
        self.worker = FileTranscriptionWorker(self.config, str(self.source), str(self.root / "out.txt"),
                                               lambda e, p: self.events.append((e, p)))

    def fake_model(self):
        return SimpleNamespace(transcribe=lambda samples, **kwargs: (
            [SimpleNamespace(text="Meeting discussion", start=0, end=len(samples)/16000)], None))

    def decoded_windows(self, seconds=1, overlap=.2):
        cancelled = threading.Event()
        with DecodedAudio(self.source, cancelled) as decoded:
            return list(pcm_windows(decoded.samples(), seconds, overlap, cancelled))

    def test_stereo_resampling_overlap_and_tail(self):
        chunks = self.decoded_windows()
        self.assertEqual([round(start, 1) for start, _ in chunks], [0, .8, 1.6])
        self.assertEqual([len(samples) for _, samples in chunks], [16000, 16000, 9600])
        self.assertTrue(all(samples.ndim == 1 for _, samples in chunks))

    def test_exact_window_does_not_repeat_overlap(self):
        make_audio(self.source, seconds=1)
        self.assertEqual(len(self.decoded_windows()), 1)

    def test_decoder_runs_in_a_separate_process_and_reports_duration(self):
        import os

        cancelled = threading.Event()
        with DecodedAudio(self.source, cancelled) as decoded:
            samples = sum(len(block) for block in decoded.samples())
            child_pid = decoded.process.pid
            duration = decoded.duration
        self.assertNotEqual(child_pid, os.getpid())
        self.assertEqual(decoded.process.returncode, 0)
        self.assertAlmostEqual(duration, 2.2, places=2)
        self.assertEqual(samples, round(2.2 * 16000))

    def test_cancel_stops_the_decoder_process(self):
        make_audio(self.source, seconds=120)
        cancelled = threading.Event()
        with DecodedAudio(self.source, cancelled) as decoded:
            for _ in decoded.samples():
                cancelled.set()
        self.assertIsNotNone(decoded.process.poll())

    def fake_decoder(self, code):
        return patch.object(file_transcriber, "decoder_command",
                            return_value=[sys.executable, "-I", "-c", code])

    def test_decoder_crash_is_reported_not_raised_in_the_app(self):
        with self.fake_decoder("import sys; sys.stdin.buffer.read(1); sys.exit(3)"):
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                DecodedAudio(self.source, threading.Event())

    def test_decoder_error_message_is_passed_on(self):
        code = ("import sys, json; sys.stdin.buffer.read(1); "
                "sys.stderr.write(json.dumps({'error': 'Invalid data found'}) + chr(10)); sys.exit(2)")
        with self.fake_decoder(code):
            with self.assertRaisesRegex(RuntimeError, "Invalid data found"):
                DecodedAudio(self.source, threading.Event())

    def test_silent_decoder_is_stopped_after_the_idle_timeout(self):
        code = ("import sys, time; sys.stdin.buffer.read(1); "
                "sys.stdout.write('{\"duration\": 1}' + chr(10)); sys.stdout.flush(); time.sleep(30)")
        with self.fake_decoder(code):
            decoded = DecodedAudio(self.source, threading.Event(), idle_timeout=0.5)
            with self.assertRaisesRegex(RuntimeError, "stopped responding"):
                list(decoded.samples())
        decoded.process.wait(timeout=10)
        self.assertIsNotNone(decoded.process.poll())

    @unittest.skipUnless(sys.platform == "win32", "Windows job objects")
    def test_decoder_job_blocks_child_processes(self):
        code = ("import subprocess, sys; sys.stdin.buffer.read(1)\n"
                "try:\n    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n"
                "except OSError:\n    sys.exit(0)\n"
                "sys.exit(5)")
        with self.fake_decoder(code):
            with self.assertRaisesRegex(RuntimeError, "exit code 0"):
                # Exit 0 without a header means the spawn was refused inside the job.
                DecodedAudio(self.source, threading.Event())

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
        import io
        import audio_decode_worker
        container = SimpleNamespace(streams=SimpleNamespace(audio=[]))
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = container
        with patch("av.open", return_value=context) as opened:
            with self.assertRaisesRegex(ValueError, "no audio track"):
                audio_decode_worker.decode_to(io.BytesIO(), str(self.source))
        self.assertEqual(opened.call_args.kwargs["options"], {"protocol_whitelist": "file"})

    def test_decoder_only_opens_existing_local_files(self):
        import audio_decode_worker
        for raw in ("http://example.com/a.mp3", "rtsp://host/stream", "\\\\.\\pipe\\x", str(self.root),
                    str(self.root / "missing.wav"), ""):
            with self.subTest(path=raw), self.assertRaises(ValueError):
                audio_decode_worker.checked_input_path(raw)
        self.assertEqual(audio_decode_worker.checked_input_path(str(self.source)), self.source.resolve())

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

    def test_saved_recordings_use_near_full_whisper_windows(self):
        from file_transcriber import FILE_OVERLAP_SECONDS, FILE_WINDOW_SECONDS

        self.assertEqual((FILE_WINDOW_SECONDS, FILE_OVERLAP_SECONDS), (28.0, 2.0))
        self.assertEqual(self.worker.config.chunk_seconds, 28.0)
        self.assertEqual(self.worker.config.overlap_seconds, 2.0)

    def test_missing_model_does_not_load_or_download(self):
        self.worker.config = replace(self.worker.config, whisper_model_dir=str(self.root / "missing"))
        with patch("file_transcriber.load_whisper_model") as loader:
            self.worker.run()
            loader.assert_not_called()
        self.assertEqual(self.events[-1][1]["outcome"], "failed")
        self.assertTrue(any("setup" in p.get("message", "") for _, p in self.events))


if __name__ == "__main__":
    unittest.main()
