import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from transcriber_engine import (
    AtomicTranscriptWriter,
    AudioChunk,
    DiarizationTurn,
    DuplicateSuppressor,
    EngineConfig,
    InputBlockBuffer,
    MeetingTranscriberEngine,
    SpeakerRegistry,
    TranscriptRow,
    TranscriptStore,
    cluster_local_embeddings,
    extract_row_audio_window,
    microphone_health_message,
    preferred_input_sample_rate,
    resample_audio,
    rms_to_meter_percent,
)


class SpeakerRegistryTests(unittest.TestCase):
    def test_same_embedding_reuses_existing_speaker(self):
        registry = SpeakerRegistry(match_threshold=0.70)

        first = registry.assign([1.0, 0.0, 0.0], confidence=0.95)
        second = registry.assign([0.98, 0.02, 0.0], confidence=0.95)

        self.assertEqual(first.key, "speaker_1")
        self.assertEqual(second.key, "speaker_1")
        self.assertEqual(registry.display_name("speaker_1"), "Speaker 1")

    def test_different_embedding_creates_new_speaker(self):
        registry = SpeakerRegistry(match_threshold=0.70)

        first = registry.assign([1.0, 0.0, 0.0], confidence=0.95)
        second = registry.assign([0.0, 1.0, 0.0], confidence=0.95)

        self.assertEqual(first.key, "speaker_1")
        self.assertEqual(second.key, "speaker_2")

    def test_low_confidence_keeps_unknown_speaker(self):
        registry = SpeakerRegistry(match_threshold=0.70)

        identity = registry.assign([1.0, 0.0, 0.0], confidence=0.20)

        self.assertIsNone(identity)


class TranscriptStoreTests(unittest.TestCase):
    def test_transcript_row_updates_when_diarization_arrives_later(self):
        registry = SpeakerRegistry(match_threshold=0.70)
        store = TranscriptStore(registry)

        row = store.add_transcript(
            chunk_index=4,
            start=12.0,
            end=15.0,
            text="We need to close the action items.",
        )
        self.assertEqual(row.speaker_label, "Speaker ?")

        updates = store.apply_diarization(
            chunk_index=4,
            turns=[
                DiarizationTurn(
                    start=11.5,
                    end=15.5,
                    local_label="LOCAL_A",
                    embedding=[1.0, 0.0, 0.0],
                    confidence=0.95,
                )
            ],
        )

        self.assertEqual(len(updates), 1)
        self.assertEqual(store.rows[0].id, row.id)
        self.assertEqual(store.rows[0].speaker_label, "Speaker 1")
        self.assertEqual(store.rows[0].text, "We need to close the action items.")

    def test_low_overlap_diarization_does_not_force_wrong_label(self):
        registry = SpeakerRegistry(match_threshold=0.70)
        store = TranscriptStore(registry, min_overlap_ratio=0.50)
        store.add_transcript(chunk_index=1, start=10.0, end=20.0, text="Long segment")

        updates = store.apply_diarization(
            chunk_index=1,
            turns=[
                DiarizationTurn(
                    start=10.0,
                    end=11.0,
                    local_label="LOCAL_A",
                    embedding=[1.0, 0.0, 0.0],
                    confidence=0.95,
                )
            ],
        )

        self.assertEqual(updates, [])
        self.assertEqual(store.rows[0].speaker_label, "Speaker ?")

    def test_rename_speaker_updates_existing_rows(self):
        registry = SpeakerRegistry(match_threshold=0.70)
        store = TranscriptStore(registry)
        store.add_transcript(chunk_index=1, start=0.0, end=2.0, text="Hello.")
        store.apply_diarization(
            chunk_index=1,
            turns=[
                DiarizationTurn(
                    start=0.0,
                    end=2.0,
                    local_label="LOCAL_A",
                    embedding=[1.0, 0.0, 0.0],
                    confidence=0.95,
                )
            ],
        )

        store.rename_speaker("speaker_1", "Alice")

        self.assertEqual(store.rows[0].speaker_label, "Alice")


class DuplicateSuppressorTests(unittest.TestCase):
    def test_duplicate_text_inside_overlap_window_is_suppressed(self):
        suppressor = DuplicateSuppressor(window_seconds=2.0)

        self.assertFalse(suppressor.is_duplicate(10.0, "Confirm the deadline."))
        self.assertTrue(suppressor.is_duplicate(11.5, " confirm  the deadline "))
        self.assertFalse(suppressor.is_duplicate(13.1, "Confirm the deadline."))


class LocalDiarizationHelperTests(unittest.TestCase):
    def test_cluster_local_embeddings_groups_similar_vectors(self):
        labels = cluster_local_embeddings(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 1.0, 0.0],
            ],
            distance_threshold=0.20,
        )

        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_short_transcript_segment_is_padded_for_embedding(self):
        import numpy as np

        chunk = AudioChunk(
            index=1,
            start_time=0.0,
            samples=np.ones(16000, dtype=np.float32),
            sample_rate=16000,
        )
        row = TranscriptRow(
            id="row_1",
            chunk_index=1,
            start=0.10,
            end=0.20,
            text="Yes.",
        )

        window = extract_row_audio_window(chunk, row, target_seconds=1.0)

        self.assertEqual(len(window), 16000)


class AtomicTranscriptWriterTests(unittest.TestCase):
    def test_refresh_writes_latest_labels_atomically(self):
        registry = SpeakerRegistry()
        store = TranscriptStore(registry)
        row = store.add_transcript(chunk_index=1, start=0.0, end=2.0, text="Hello team.")
        row.speaker_key = "speaker_1"
        row.speaker_label = "Speaker 1"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transcript.txt")
            writer = AtomicTranscriptWriter(path)
            writer.refresh(store.rows)

            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("[00:00:00] Speaker 1: Hello team.", content)


class InputBlockBufferTests(unittest.TestCase):
    def test_callback_block_is_copied_before_queueing(self):
        buffer = InputBlockBuffer(max_blocks=2)
        block = [[0.1], [0.2], [0.3]]

        buffer.push_from_callback(block)
        block[0][0] = 9.9

        queued = buffer.pop()
        self.assertEqual(queued[0][0], 0.1)


class MicrophoneDiagnosticsTests(unittest.TestCase):
    def test_rms_to_meter_percent_makes_real_microphone_levels_visible(self):
        self.assertEqual(rms_to_meter_percent(0.0), 0)
        self.assertEqual(rms_to_meter_percent(0.00001), 0)
        self.assertGreaterEqual(rms_to_meter_percent(0.001), 5)
        self.assertGreaterEqual(rms_to_meter_percent(0.01), 50)
        self.assertEqual(rms_to_meter_percent(1.0), 100)

    def test_microphone_health_reports_no_callbacks(self):
        message = microphone_health_message(block_count=0, peak_rms=0.0, chunk_count=0)

        self.assertIn("No microphone audio was received", message)
        self.assertIn("microphone permission", message)

    def test_microphone_health_reports_silent_input(self):
        message = microphone_health_message(block_count=10, peak_rms=0.00002, chunk_count=0)

        self.assertIn("Microphone input looks silent", message)
        self.assertIn("choose a different input device", message)

    def test_microphone_health_accepts_audible_input(self):
        self.assertIsNone(microphone_health_message(block_count=10, peak_rms=0.01, chunk_count=1))

    def test_engine_stop_emits_diagnostic_when_recording_was_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))
            engine._audio_block_count = 10
            engine._audio_peak_rms = 0.00002
            engine._audio_chunk_count = 0

            engine.stop()

        messages = [payload["message"] for event_type, payload in events if event_type == "error"]
        self.assertTrue(any("Microphone input looks silent" in message for message in messages))

    def test_preferred_input_sample_rate_uses_target_when_supported(self):
        class FakeSoundDevice:
            @staticmethod
            def check_input_settings(device=None, channels=None, samplerate=None, dtype=None):
                self.assertEqual(samplerate, 16000)

        self.assertEqual(preferred_input_sample_rate(FakeSoundDevice, None, 16000), 16000)

    def test_preferred_input_sample_rate_falls_back_to_device_default(self):
        class FakeSoundDevice:
            @staticmethod
            def check_input_settings(device=None, channels=None, samplerate=None, dtype=None):
                raise RuntimeError("unsupported")

            @staticmethod
            def query_devices(device=None, kind=None):
                return {"default_samplerate": 48000}

        self.assertEqual(preferred_input_sample_rate(FakeSoundDevice, 9, 16000), 48000)

    def test_resample_audio_changes_length_for_model_sample_rate(self):
        import numpy as np

        source = np.ones(48000, dtype=np.float32)
        resampled = resample_audio(source, 48000, 16000)

        self.assertEqual(len(resampled), 16000)
        self.assertEqual(resampled.dtype, np.float32)


class LocalSpeakerModelLoadTests(unittest.TestCase):
    def test_local_speaker_model_uses_copy_strategy_to_avoid_windows_symlinks(self):
        calls = {}
        copy_strategy = object()

        class FakeLocalStrategy:
            COPY = copy_strategy

        class FakeEncoderClassifier:
            @staticmethod
            def from_hparams(**kwargs):
                calls["kwargs"] = kwargs
                return "classifier"

        fake_speechbrain = types.ModuleType("speechbrain")
        fake_speechbrain.__path__ = []
        fake_inference = types.ModuleType("speechbrain.inference")
        fake_inference.__path__ = []
        fake_speaker = types.ModuleType("speechbrain.inference.speaker")
        fake_speaker.EncoderClassifier = FakeEncoderClassifier
        fake_utils = types.ModuleType("speechbrain.utils")
        fake_utils.__path__ = []
        fake_fetching = types.ModuleType("speechbrain.utils.fetching")
        fake_fetching.LocalStrategy = FakeLocalStrategy
        fake_speechbrain.inference = fake_inference
        fake_speechbrain.utils = fake_utils
        fake_inference.speaker = fake_speaker
        fake_utils.fetching = fake_fetching
        fake_modules = {
            "speechbrain": fake_speechbrain,
            "speechbrain.inference": fake_inference,
            "speechbrain.inference.speaker": fake_speaker,
            "speechbrain.utils": fake_utils,
            "speechbrain.utils.fetching": fake_fetching,
        }
        previous_modules = {name: sys.modules.get(name) for name in fake_modules}
        try:
            sys.modules.update(fake_modules)
            config = EngineConfig(speaker_embedding_model_dir=r"C:\models\speechbrain-ecapa")
            result = MeetingTranscriberEngine(config)._load_local_speaker_model()
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result, "classifier")
        self.assertEqual(calls["kwargs"]["source"], r"C:\models\speechbrain-ecapa")
        self.assertEqual(calls["kwargs"]["savedir"], r"C:\models\speechbrain-ecapa")
        self.assertEqual(calls["kwargs"]["run_opts"], {"device": "cpu"})
        self.assertIs(calls["kwargs"]["local_strategy"], copy_strategy)


class EngineConfigTests(unittest.TestCase):
    def test_missing_model_paths_return_clear_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EngineConfig(
                whisper_model_dir=os.path.join(tmp, "missing-whisper"),
                speaker_embedding_model_dir=os.path.join(tmp, "missing-speaker"),
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = config.validate()

        self.assertIn("Whisper model path does not exist", "\n".join(errors))
        self.assertIn("Speaker embedding model path does not exist", "\n".join(errors))
        self.assertNotIn("Pyannote pipeline path does not exist", "\n".join(errors))

    def test_empty_model_folders_return_incomplete_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            speaker = os.path.join(tmp, "speaker")
            os.mkdir(whisper)
            os.mkdir(speaker)
            config = EngineConfig(
                whisper_model_dir=whisper,
                speaker_embedding_model_dir=speaker,
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = "\n".join(config.validate())

        self.assertIn("Whisper model is incomplete: expected model.bin", errors)
        self.assertIn("Speaker embedding model is incomplete", errors)
        self.assertNotIn("Pyannote", errors)

    def test_minimal_required_model_files_pass_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            speaker = os.path.join(tmp, "speaker")
            os.mkdir(whisper)
            os.mkdir(speaker)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            open(os.path.join(speaker, "hyperparams.yaml"), "w", encoding="utf-8").close()
            open(os.path.join(speaker, "embedding_model.ckpt"), "wb").close()
            config = EngineConfig(
                whisper_model_dir=whisper,
                speaker_embedding_model_dir=speaker,
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = config.validate()

        self.assertEqual(errors, [])

    def test_pyannote_backend_validates_pyannote_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            pipeline = os.path.join(tmp, "pipeline")
            embedding = os.path.join(tmp, "embedding")
            os.mkdir(whisper)
            os.mkdir(pipeline)
            os.mkdir(embedding)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            open(os.path.join(pipeline, "config.yaml"), "w", encoding="utf-8").close()
            open(os.path.join(embedding, "config.yaml"), "w", encoding="utf-8").close()
            open(os.path.join(embedding, "pytorch_model.bin"), "wb").close()
            config = EngineConfig(
                diarization_backend="pyannote",
                whisper_model_dir=whisper,
                pyannote_pipeline_dir=pipeline,
                pyannote_embedding_model_dir=embedding,
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = config.validate()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
