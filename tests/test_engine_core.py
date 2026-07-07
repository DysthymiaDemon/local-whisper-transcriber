import os
import sys
import tempfile
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
    SpeakerRegistry,
    TranscriptRow,
    TranscriptStore,
    cluster_local_embeddings,
    extract_row_audio_window,
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
