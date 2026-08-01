"""Versioned model artifact loading shared by API and CLI inference."""

from pathlib import Path

import joblib

from audio_pipeline import DISTRESS_THRESHOLD, MFCC_COUNT


def load_model_artifact(model_path):
    artifact = joblib.load(Path(model_path))
    if not isinstance(artifact, dict) or "model" not in artifact:
        return artifact, DISTRESS_THRESHOLD

    feature_config = artifact.get("feature_config", {})
    if feature_config.get("mfcc_count", MFCC_COUNT) != MFCC_COUNT:
        raise RuntimeError("The model artifact expects a different MFCC feature count.")
    return artifact["model"], float(
        artifact.get("distress_threshold", DISTRESS_THRESHOLD)
    )
