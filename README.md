# PanicSense AI — Voice Distress Detector

A machine learning project that analyzes voice audio and detects whether the speaker is in **distress/panic** or speaking in a **normal** tone. Built using the RAVDESS Emotional Speech Dataset and a Random Forest classifier.

---

## How It Works

### 1. Dataset
The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. It contains 24 professional actors (12 male, 12 female) performing 8 emotions:

| Code | Emotion | Label in Model |
|------|---------|----------------|
| 01 | Neutral | Normal |
| 02 | Calm | Normal |
| 03 | Happy | Normal |
| 04 | Sad | Normal |
| **05** | **Angry** | **Distress** |
| **06** | **Fearful** | **Distress** |
| 07 | Disgust | Normal |
| 08 | Surprised | Normal |

Filenames follow the format: `03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav`

### 2. Feature Extraction
For each audio file, the model extracts **40 MFCC (Mel-Frequency Cepstral Coefficients)** features using `librosa`. These are averaged over time to produce a single 40-dimensional feature vector per clip.

```
Audio (.wav) → librosa.load() → MFCC (40 coefficients) → Mean over time → Feature vector (shape: 40,)
```

### 3. Model Training
A **Random Forest Classifier** (100 estimators) is trained on these features with an 80/20 train-test split. The trained model is saved as `model.pkl`.

### 4. Prediction Pipeline
```
Input audio → Feature extraction (MFCC) → model.predict() → "Distress" or "Normal"
```
The model uses a **30% sensitivity threshold** — if the Distress probability exceeds 30%, the audio is flagged (instead of the default 50%), making it more sensitive to real-world emotional speech.

### 5. Web Interface
- **Live Recording**: Browser captures mic audio via `MediaRecorder API`, converts it to WAV using `AudioContext`, then sends it to the Flask backend.
- **File Upload**: User provides a `.wav/.mp3/.flac/.ogg` file, sent directly to the backend.
- **Backend** (`api/index.py`): Flask API receives audio, extracts MFCCs, runs prediction, and returns JSON with `prediction`, `confidence`, and `normal_confidence`.
- **Frontend** (`index.html`): Shows result as 🚨 Emergency, ⚠️ Borderline, or ✅ Normal with animated confidence bars.

---

## Project Structure

```
ML-panic_voice_detector/
├── api/
│   └── index.py          # Flask backend API
├── dataset/
│   ├── Actor_01/         # RAVDESS audio files (Actor 01–24)
│   └── ...
├── index.html            # Web frontend (UI)
├── live_predict.py       # CLI prediction script (file or mic)
├── train_model.py        # Model training script
├── model.pkl             # Pre-trained Random Forest model
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel deployment config
└── .vercelignore         # Files excluded from deployment
```

---

## How to Execute

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Retrain the Model

Only needed if you want to retrain from scratch. The `model.pkl` is already included.

```bash
python train_model.py
```

### Step 3 — Run the Web App (Recommended)

```bash
python api/index.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

You will see two options:
- 🎙️ **Record Voice** — Click the mic button to start recording, click again to stop and analyze.
- 📁 **Upload Audio** — Drag & drop or browse for a `.wav`, `.mp3`, `.flac`, or `.ogg` file.

### Step 4 — (Alternative) Run the CLI Script

If you prefer the command line without the web interface:

```bash
python live_predict.py
```

It will prompt you to enter a file path:
```
Enter dataset audio file path: dataset/Actor_01/03-01-06-01-02-01-01.wav
```

To switch to **live microphone mode**, open `live_predict.py` and replace lines 76–77:
```python
# Change this:
file_path = input("Enter dataset audio file path:")
file_path = resolve_audio_path(file_path)

# To this:
file_path = record_audio()
```

---

## Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login and deploy
vercel
```

Or connect your GitHub repository to [vercel.com](https://vercel.com) and import the project directly. Vercel will auto-detect the Python API and static frontend.

> **Note:** The `dataset/` folder and training scripts are excluded from deployment via `.vercelignore` to keep the bundle small.

---

## Limitations

- Trained on **acted emotional speech** (RAVDESS), not real-world panic recordings. Real screaming may score lower than expected because the acoustic profile differs from studio-performed fear/anger.
- Best accuracy is achieved with **clear, close-mic audio** similar to the training data.
- Single-shot prediction — analyzes only the first 3 seconds of audio.
