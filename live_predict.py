import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pickle
import os

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


file_path = input("Enter dataset audio file path: ")

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

# Optional: delete recorded file
if os.path.exists(file_path):
    os.remove(file_path)
