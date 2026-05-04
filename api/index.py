from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load model from project root (one level above api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Serve frontend
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)


def preprocess_audio(audio):
    """Normalize amplitude and apply pre-emphasis to reduce mic-vs-studio mismatch."""
    # 1. Peak-normalize to -3 dBFS so mic level differences don't affect MFCCs
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.707  # -3 dBFS
    # 2. Pre-emphasis filter (standard in speech processing)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    return audio


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, mono=True)
    audio = preprocess_audio(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    fname = file.filename or "upload.wav"
    suffix = ".wav"
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        if fname.lower().endswith(ext):
            suffix = ext
            break

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        features = extract_features(tmp_path)
        features = features.reshape(1, -1)
        proba = model.predict_proba(features)[0]
        classes = model.classes_.tolist()

        distress_conf = float(proba[classes.index('Distress')])
        normal_conf   = float(proba[classes.index('Normal')])

        # Threshold: 45% — safe for real mic audio after normalization
        DISTRESS_THRESHOLD = 0.45
        is_distress = distress_conf >= DISTRESS_THRESHOLD
        label = 'Distress' if is_distress else 'Normal'

        return jsonify({
            "prediction": label,
            "confidence": round(distress_conf * 100, 1),
            "normal_confidence": round(normal_conf * 100, 1),
            "is_distress": is_distress
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
