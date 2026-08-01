"""Validate and import labeled recordings into the project's dataset."""

import argparse
import csv
import re
import shutil
from pathlib import Path

from audio_pipeline import SUPPORTED_AUDIO_EXTENSIONS, load_audio
from dataset_tools import VALID_LABELS, file_digest


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"


def safe_speaker_id(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise ValueError("speaker_id must contain letters or numbers")
    return cleaned


def read_source_manifest(manifest_path):
    source_dir = manifest_path.parent.resolve()
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"file", "label", "speaker_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest columns must be: file,label,speaker_id")
        rows = list(reader)

    validated = []
    for row_number, row in enumerate(rows, start=2):
        source_path = (source_dir / row["file"]).resolve()
        if not source_path.is_relative_to(source_dir) or not source_path.is_file():
            raise ValueError(f"Invalid source file on manifest row {row_number}")
        if source_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported format on manifest row {row_number}")
        label = row["label"].strip().title()
        if label not in VALID_LABELS:
            raise ValueError(f"Label must be Normal or Distress on row {row_number}")
        load_audio(source_path)
        validated.append((source_path, label, safe_speaker_id(row["speaker_id"])))
    return validated


def read_existing_imports(manifest_path):
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = csv.DictReader(manifest_file)
        return {row["sha256"]: row for row in rows}


def import_recordings(manifest_path, output_dir):
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(DATASET_DIR.resolve()):
        raise ValueError("Output directory must be inside the project dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_output = output_dir / "labels.csv"
    imported = read_existing_imports(manifest_output)
    added_count = 0
    for source_path, label, speaker_id in read_source_manifest(manifest_path):
        digest = file_digest(source_path)
        existing = imported.get(digest)
        if existing and existing["label"] != label:
            raise ValueError(f"Conflicting labels for duplicate file: {source_path}")
        if existing:
            continue
        filename = f"{speaker_id}_{digest[:16]}{source_path.suffix.lower()}"
        shutil.copy2(source_path, output_dir / filename)
        imported[digest] = {
            "file": filename,
            "label": label,
            "speaker_id": speaker_id,
            "sha256": digest,
        }
        added_count += 1

    with manifest_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["file", "label", "speaker_id", "sha256"]
        )
        writer.writeheader()
        writer.writerows(imported.values())
    return added_count, manifest_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with file,label,speaker_id")
    parser.add_argument(
        "--output", type=Path, default=DATASET_DIR / "custom", help="Import directory"
    )
    arguments = parser.parse_args()
    count, output_manifest = import_recordings(arguments.manifest, arguments.output)
    print(f"Imported {count} new unique recordings into {output_manifest.parent}")
    print(f"Training manifest: {output_manifest}")


if __name__ == "__main__":
    main()
