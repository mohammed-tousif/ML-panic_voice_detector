from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from audio_pipeline import (
    AUDIO_DURATION_SECONDS,
    MFCC_COUNT,
    augment_audio,
    extract_features_from_audio,
    load_audio,
)
from dataset_tools import discover_recordings

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "model.pkl"
METRICS_PATH = PROJECT_DIR / "training_metrics.json"
RANDOM_STATE = 42
MIN_THRESHOLD_PRECISION = 0.50
AUGMENTATIONS_PER_RECORDING = 1


def build_feature_matrix(recordings, augmentation_count=0, random_state=RANDOM_STATE):
    """Extract originals plus deterministic training-only augmentations."""
    random_generator = np.random.default_rng(random_state)
    features = []
    labels = []
    for recording in recordings:
        audio, sample_rate = load_audio(recording.path)
        features.append(extract_features_from_audio(audio, sample_rate))
        labels.append(recording.label)
        for _ in range(augmentation_count):
            augmented = augment_audio(audio, sample_rate, random_generator)
            features.append(extract_features_from_audio(augmented, sample_rate))
            labels.append(recording.label)
    return np.asarray(features), np.asarray(labels)


def grouped_split(recordings, test_size, random_state):
    labels = np.asarray([recording.label for recording in recordings])
    speakers = np.asarray([recording.speaker_id for recording in recordings])
    indices = np.arange(len(recordings))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    left_indices, right_indices = next(splitter.split(indices, labels, speakers))
    left = [recordings[index] for index in left_indices]
    right = [recordings[index] for index in right_indices]
    return left, right


def make_classifier(random_state=RANDOM_STATE):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )


def distress_probabilities(model, features):
    distress_index = model.classes_.tolist().index("Distress")
    return model.predict_proba(features)[:, distress_index]


def select_threshold(labels, probabilities, minimum_precision=MIN_THRESHOLD_PRECISION):
    binary_labels = np.asarray(labels) == "Distress"
    precision, recall, thresholds = precision_recall_curve(binary_labels, probabilities)
    candidates = [
        (float(recall[index]), float(precision[index]), float(threshold))
        for index, threshold in enumerate(thresholds)
        if precision[index] >= minimum_precision
    ]
    if not candidates:
        return 0.5, precision, recall, thresholds
    best_recall, best_precision, best_threshold = max(
        candidates, key=lambda candidate: (candidate[0], candidate[1], candidate[2])
    )
    print(
        f"Selected threshold: {best_threshold:.3f} "
        f"(validation precision={best_precision:.3f}, recall={best_recall:.3f})"
    )
    return best_threshold, precision, recall, thresholds


def speaker_ids(recordings):
    return sorted({recording.speaker_id for recording in recordings})


def build_metrics(context):
    test_labels = context["test_labels"]
    test_predictions = context["test_predictions"]
    return {
        "recording_count": len(context["recordings"]),
        "augmentation_count_per_development_recording": AUGMENTATIONS_PER_RECORDING,
        "split": {
            "train_speakers": speaker_ids(context["train"]),
            "validation_speakers": speaker_ids(context["validation"]),
            "test_speakers": speaker_ids(context["test"]),
        },
        "selected_threshold": context["threshold"],
        "minimum_validation_precision": MIN_THRESHOLD_PRECISION,
        "test": {
            "accuracy": accuracy_score(test_labels, test_predictions),
            "distress_precision": precision_score(
                test_labels, test_predictions, pos_label="Distress", zero_division=0
            ),
            "distress_recall": recall_score(
                test_labels, test_predictions, pos_label="Distress", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                test_labels, test_predictions, labels=["Normal", "Distress"]
            ).tolist(),
            "classification_report": classification_report(
                test_labels, test_predictions, output_dict=True
            ),
        },
        "validation_precision_recall_curve": {
            "precision": context["precision_curve"].tolist(),
            "recall": context["recall_curve"].tolist(),
            "thresholds": context["curve_thresholds"].tolist(),
        },
    }


def save_outputs(model, threshold, metrics):
    artifact = {
        "artifact_version": 2,
        "model": model,
        "distress_threshold": threshold,
        "feature_config": {
            "duration_seconds": AUDIO_DURATION_SECONDS,
            "mfcc_count": MFCC_COUNT,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    recordings = discover_recordings(DATASET_PATH)
    development, test = grouped_split(recordings, test_size=0.2, random_state=RANDOM_STATE)
    train, validation = grouped_split(
        development, test_size=0.2, random_state=RANDOM_STATE + 1
    )

    train_features, train_labels = build_feature_matrix(
        train, AUGMENTATIONS_PER_RECORDING, RANDOM_STATE
    )
    validation_features, validation_labels = build_feature_matrix(validation)
    selection_model = make_classifier()
    selection_model.fit(train_features, train_labels)
    validation_probabilities = distress_probabilities(selection_model, validation_features)
    threshold, precision_curve, recall_curve, curve_thresholds = select_threshold(
        validation_labels, validation_probabilities
    )

    development_features, development_labels = build_feature_matrix(
        development, AUGMENTATIONS_PER_RECORDING, RANDOM_STATE + 2
    )
    test_features, test_labels = build_feature_matrix(test)
    model = make_classifier()
    model.fit(development_features, development_labels)
    test_probabilities = distress_probabilities(model, test_features)
    test_predictions = np.where(test_probabilities >= threshold, "Distress", "Normal")

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Unique recordings: {len(recordings)}")
    print(f"Development speakers: {speaker_ids(development)}")
    print(f"Test speakers: {speaker_ids(test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(test_labels, test_predictions))

    metrics_context = {
        "recordings": recordings,
        "train": train,
        "validation": validation,
        "test": test,
        "threshold": threshold,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "curve_thresholds": curve_thresholds,
    }
    save_outputs(model, threshold, build_metrics(metrics_context))
    print("Model saved successfully!")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
