"""Integration tests for the public Flask API contract."""

import math
import struct
import unittest
import wave
from io import BytesIO

from api.index import app
from audio_pipeline import MAX_UPLOAD_BYTES


def make_wav():
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        samples = (
            int(8000 * math.sin(2 * math.pi * 440 * index / 22050))
            for index in range(22050)
        )
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    output.seek(0)
    return output


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_missing_audio_is_rejected(self):
        self.assertEqual(self.client.post("/api/predict").status_code, 400)

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post(
            "/api/predict", data={"audio": (BytesIO(b"text"), "audio.txt")}
        )
        self.assertEqual(response.status_code, 415)

    def test_empty_audio_is_rejected(self):
        response = self.client.post(
            "/api/predict", data={"audio": (BytesIO(), "audio.wav")}
        )
        self.assertEqual(response.status_code, 400)

    def test_malformed_audio_is_rejected(self):
        response = self.client.post(
            "/api/predict", data={"audio": (BytesIO(b"not audio"), "audio.wav")}
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_audio_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            data={"audio": (BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1)), "audio.wav")},
        )
        self.assertEqual(response.status_code, 413)

    def test_valid_audio_is_predicted(self):
        response = self.client.post(
            "/api/predict", data={"audio": (make_wav(), "audio.wav")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], {"Normal", "Distress"})


if __name__ == "__main__":
    unittest.main()
