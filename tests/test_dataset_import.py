"""Integration tests for safe, repeatable custom dataset imports."""

import csv
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from prepare_dataset import import_recordings


def write_wav(file_path):
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        frames = [
            struct.pack("<h", int(5000 * math.sin(2 * math.pi * 220 * index / 8000)))
            for index in range(8000)
        ]
        wav_file.writeframes(b"".join(frames))


class DatasetImportTests(unittest.TestCase):
    def test_import_is_validated_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "dataset"
            source_dir = root / "source"
            source_dir.mkdir()
            audio_path = source_dir / "sample.wav"
            write_wav(audio_path)
            manifest_path = source_dir / "samples.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as manifest:
                writer = csv.writer(manifest)
                writer.writerow(["file", "label", "speaker_id"])
                writer.writerow(["sample.wav", "Distress", "speaker-01"])

            output_dir = dataset_dir / "custom"
            with patch("prepare_dataset.DATASET_DIR", dataset_dir):
                first_count, output_manifest = import_recordings(
                    manifest_path, output_dir
                )
                second_count, _ = import_recordings(manifest_path, output_dir)

            self.assertEqual(first_count, 1)
            self.assertEqual(second_count, 0)
            with output_manifest.open(encoding="utf-8", newline="") as manifest:
                self.assertEqual(len(list(csv.DictReader(manifest))), 1)


if __name__ == "__main__":
    unittest.main()
