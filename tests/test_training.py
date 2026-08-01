"""Tests for data splitting, threshold tuning, and augmentation contracts."""

import unittest

import numpy as np

from audio_pipeline import augment_audio
from dataset_tools import Recording
from train_model import grouped_split, select_threshold


class TrainingTests(unittest.TestCase):
    def test_grouped_split_keeps_speakers_isolated(self):
        recordings = [
            Recording(None, label, f"speaker-{speaker}", f"{speaker}-{index}")
            for speaker in range(10)
            for index, label in enumerate(("Normal", "Distress"))
        ]
        left, right = grouped_split(recordings, test_size=0.2, random_state=42)
        left_speakers = {recording.speaker_id for recording in left}
        right_speakers = {recording.speaker_id for recording in right}
        self.assertFalse(left_speakers & right_speakers)

    def test_threshold_selection_respects_precision_floor(self):
        labels = np.asarray(["Distress", "Distress", "Normal", "Normal"])
        probabilities = np.asarray([0.9, 0.6, 0.55, 0.1])
        threshold, _precision, _recall, _thresholds = select_threshold(
            labels, probabilities, minimum_precision=0.75
        )
        predictions = probabilities >= threshold
        true_positives = np.sum(predictions & (labels == "Distress"))
        predicted_positives = np.sum(predictions)
        self.assertGreaterEqual(true_positives / predicted_positives, 0.75)

    def test_augmentation_is_deterministic_and_non_destructive(self):
        audio = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
        original = audio.copy()
        first = augment_audio(audio, 22050, np.random.default_rng(7))
        second = augment_audio(audio, 22050, np.random.default_rng(7))
        np.testing.assert_array_equal(audio, original)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, original))


if __name__ == "__main__":
    unittest.main()
