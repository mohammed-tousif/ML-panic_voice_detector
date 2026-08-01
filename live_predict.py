import argparse
from pathlib import Path

from audio_pipeline import SUPPORTED_AUDIO_EXTENSIONS, extract_features
from model_artifact import load_model_artifact

# Load trained model
PROJECT_DIR = Path(__file__).resolve().parent
model, distress_threshold = load_model_artifact(PROJECT_DIR / "model.pkl")

def first_audio_file(directory):
    files = sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    return files[0] if files else None


def extension_match(path):
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = Path(f"{path}{extension}")
        if candidate.is_file():
            return candidate
    return None


def prefix_match(path):
    directory = path.parent if str(path.parent) else Path.cwd()
    if not directory.is_dir():
        return None
    matches = sorted(
        candidate
        for candidate in directory.glob(f"{path.name}*")
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    return matches[0] if matches else None


def resolve_audio_path(raw_path):
    """Resolve a user-provided path, directory, or extensionless audio name."""
    if not raw_path:
        raise FileNotFoundError("No path provided.")

    path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        match = first_audio_file(path)
        if match:
            return match
        raise FileNotFoundError(f"No audio files found in directory: {path}")

    match = extension_match(path) or prefix_match(path)
    if match:
        return match
    raise FileNotFoundError(f"Audio file not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Classify an audio recording.")
    parser.add_argument("audio", help="Audio file, directory, or extensionless path")
    arguments = parser.parse_args()
    try:
        file_path = resolve_audio_path(arguments.audio)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return 1

    features = extract_features(file_path).reshape(1, -1)
    distress_index = model.classes_.tolist().index("Distress")
    distress_probability = model.predict_proba(features)[0][distress_index]
    prediction = "Distress" if distress_probability >= distress_threshold else "Normal"
    print("\nPrediction result:")
    print("Distress pattern detected" if prediction == "Distress" else "Normal voice detected")
    print(f"Distress score: {distress_probability:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
