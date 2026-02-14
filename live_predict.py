import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pickle
import os
import sys

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Audio recording settings
DURATION = 3  # seconds
SAMPLE_RATE = 22050

def record_audio(filename="live_audio.wav"):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("✅ Recording finished.")
    return filename

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


def resolve_audio_path(path):
    """Return a valid audio file path.
    - Expands user, strips quotes.
    - If path exists and is a directory, pick first audio file inside.
    - If path doesn't exist, try common extensions (.wav, .flac, .mp3, .ogg).
    - If still not found, try to match files starting with the given basename.
    Exits with an error message if no file found.
    """
    if not path:
        print("No path provided.")
        sys.exit(1)

    path = os.path.expanduser(path).strip().strip('"').strip("'")

    if os.path.exists(path):
        if os.path.isdir(path):
            for ext in ('.wav', '.flac', '.mp3', '.ogg'):
                files = [f for f in os.listdir(path) if f.lower().endswith(ext)]
                if files:
                    return os.path.join(path, files[0])
            print(f"No audio files found in directory: {path}")
            sys.exit(1)
        return path

    # Try adding common extensions
    for ext in ('.wav', '.flac', '.mp3', '.ogg'):
        candidate = path + ext
        if os.path.exists(candidate):
            return candidate

    # Try matching files that start with the provided basename in the same directory
    dirname = os.path.dirname(path) or os.getcwd()
    base = os.path.basename(path)
    try:
        matches = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.startswith(base)]
        if matches:
            return matches[0]
    except FileNotFoundError:
        pass

    print(f"Error: audio file not found: {path}")
    sys.exit(1)


file_path = input("Enter dataset audio file path:")
file_path = resolve_audio_path(file_path)

# Extract features
features = extract_features(file_path)
features = features.reshape(1, -1)

# Predict
prediction = model.predict(features)

print("\n🔍 Prediction Result:")

if prediction[0] == "Distress":
    print("⚠️ EMERGENCY SITUATION DETECTED")
else:
    print("😊 Normal Voice Detected")

# Optional: delete recorded file only if it is the temporary recorder output
# This prevents accidental deletion of dataset or user files.
if os.path.exists(file_path):
    if os.path.basename(file_path) == "live_audio.wav":
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Warning: failed to delete {file_path}: {e}")
