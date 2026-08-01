import os
import sys
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from audio_pipeline import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_AUDIO_EXTENSIONS,
    extract_features,
)
from model_artifact import load_model_artifact

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Load model from project root (one level above api/)
# Serve frontend
@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


model, distress_threshold = load_model_artifact(BASE_DIR / "model.pkl")


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify({"error": "Audio files must be 10 MiB or smaller."}), 413


def classify_features(features):
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_.tolist()
    distress_confidence = float(probabilities[classes.index("Distress")])
    normal_confidence = float(probabilities[classes.index("Normal")])
    is_distress = distress_confidence >= distress_threshold
    return jsonify(
        {
            "prediction": "Distress" if is_distress else "Normal",
            "confidence": round(distress_confidence * 100, 1),
            "normal_confidence": round(normal_confidence * 100, 1),
            "is_distress": is_distress,
            "threshold": round(distress_threshold * 100, 1),
        }
    )


def save_upload(upload, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        upload.save(temporary_file.name)
        return Path(temporary_file.name)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    upload = request.files["audio"]
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        return jsonify({"error": "Unsupported audio format."}), 415

    temporary_path = None
    try:
        temporary_path = save_upload(upload, suffix)
        if temporary_path.stat().st_size == 0:
            return jsonify({"error": "The uploaded audio file is empty."}), 400
        return classify_features(extract_features(temporary_path).reshape(1, -1))

    except (ValueError, EOFError):
        return jsonify({"error": "The uploaded file is not valid decodable audio."}), 422
    except Exception:
        app.logger.exception("Audio prediction failed")
        return jsonify({"error": "Audio processing failed."}), 500

    finally:
        if temporary_path and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
