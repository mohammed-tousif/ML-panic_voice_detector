"""Dataset discovery and validation for RAVDESS and custom recordings."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from audio_pipeline import SUPPORTED_AUDIO_EXTENSIONS


VALID_LABELS = frozenset({"Normal", "Distress"})


@dataclass(frozen=True)
class Recording:
    path: Path
    label: str
    speaker_id: str
    digest: str


def file_digest(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest_row(row, row_number, manifest_path):
    manifest_dir = manifest_path.parent.resolve()
    file_path = (manifest_dir / row["file"]).resolve()
    if not file_path.is_relative_to(manifest_dir):
        raise ValueError(f"Unsafe file path on row {row_number}: {manifest_path}")
    if not file_path.is_file():
        raise ValueError(f"Missing audio on row {row_number}: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio on row {row_number}: {file_path}")
    label = row["label"].strip().title()
    speaker_id = row["speaker_id"].strip()
    if label not in VALID_LABELS or not speaker_id:
        raise ValueError(f"Invalid label or speaker on row {row_number}")
    return Recording(file_path, label, f"custom:{speaker_id}", file_digest(file_path))


def read_custom_manifests(dataset_path):
    recordings = []
    manifested_paths = set()
    for manifest_path in sorted(dataset_path.rglob("labels.csv")):
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            required = {"file", "label", "speaker_id"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {manifest_path}")
            for row_number, row in enumerate(reader, start=2):
                recording = parse_manifest_row(row, row_number, manifest_path)
                recordings.append(recording)
                manifested_paths.add(recording.path)
    return recordings, manifested_paths


def ravdess_recording(file_path):
    parts = file_path.stem.split("-")
    valid_emotions = {f"{code:02d}" for code in range(1, 9)}
    if len(parts) != 7 or parts[2] not in valid_emotions:
        return None
    label = "Distress" if parts[2] in {"05", "06"} else "Normal"
    return Recording(
        file_path.resolve(), label, f"ravdess:{parts[6]}", file_digest(file_path)
    )


def deduplicate_recordings(recordings):
    unique_recordings = {}
    for recording in recordings:
        existing = unique_recordings.get(recording.digest)
        if existing and existing.label != recording.label:
            raise ValueError(f"Conflicting labels for duplicate audio: {recording.path}")
        unique_recordings.setdefault(recording.digest, recording)
    return list(unique_recordings.values())


def discover_recordings(dataset_path):
    """Return content-deduplicated recordings with validated labels and speakers."""
    dataset_path = Path(dataset_path).resolve()
    recordings, manifested_paths = read_custom_manifests(dataset_path)

    for file_path in sorted(dataset_path.rglob("*.wav")):
        resolved_path = file_path.resolve()
        if resolved_path in manifested_paths:
            continue
        recording = ravdess_recording(file_path)
        if recording:
            recordings.append(recording)

    unique_recordings = deduplicate_recordings(recordings)
    if not unique_recordings:
        raise RuntimeError(f"No valid recordings were found under {dataset_path}")
    return unique_recordings
