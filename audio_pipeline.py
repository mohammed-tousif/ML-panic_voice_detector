"""Shared audio feature extraction used by training and inference."""

from pathlib import Path
from typing import Union

import librosa
import numpy as np


AUDIO_DURATION_SECONDS = 3
MFCC_COUNT = 40
DISTRESS_THRESHOLD = 0.45
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
)


def extract_features(file_path: Union[str, Path]) -> np.ndarray:
    """Return the model's fixed-size MFCC feature vector for an audio file."""
    audio, sample_rate = load_audio(file_path)
    return extract_features_from_audio(audio, sample_rate)


def load_audio(file_path: Union[str, Path]):
    """Decode the configured leading audio window as mono samples."""
    try:
        audio, sample_rate = librosa.load(
            file_path, duration=AUDIO_DURATION_SECONDS, mono=True
        )
    except Exception as error:
        raise ValueError("The file is not valid decodable audio.") from error
    if audio.size == 0:
        raise ValueError("The audio file contains no decodable samples.")
    return audio, sample_rate


def extract_features_from_audio(audio, sample_rate) -> np.ndarray:
    """Extract the exact feature vector expected by the classifier."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCC_COUNT)
    features = np.mean(mfcc.T, axis=0)
    if features.shape != (MFCC_COUNT,) or not np.all(np.isfinite(features)):
        raise ValueError("The audio file could not produce valid model features.")
    return features


def augment_audio(audio, sample_rate, random_generator) -> np.ndarray:
    """Create a conservative training-only variant of an audio signal."""
    augmented = np.asarray(audio, dtype=np.float32).copy()

    gain = random_generator.uniform(0.75, 1.25)
    augmented *= gain

    signal_power = float(np.mean(np.square(augmented)))
    if signal_power > 0:
        signal_to_noise_db = random_generator.uniform(22.0, 35.0)
        noise_power = signal_power / (10 ** (signal_to_noise_db / 10))
        noise = random_generator.normal(0.0, np.sqrt(noise_power), augmented.shape)
        augmented += noise.astype(np.float32)

    max_shift = min(len(augmented) // 10, int(sample_rate * 0.15))
    if max_shift:
        shift = int(random_generator.integers(-max_shift, max_shift + 1))
        augmented = np.roll(augmented, shift)
        if shift > 0:
            augmented[:shift] = 0
        elif shift < 0:
            augmented[shift:] = 0

    return np.clip(augmented, -1.0, 1.0)
