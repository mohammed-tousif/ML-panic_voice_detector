# PanicSense AI — Voice Distress Pattern Detector

**Live application:** [https://panic-voice-detector.vercel.app](https://panic-voice-detector.vercel.app)

PanicSense AI is an experimental machine-learning web application that analyzes a short voice recording and estimates whether its acoustic pattern is closer to **distress** or **normal speech**. A user can record from the browser or upload an audio file, and the application returns a distress score with a clear visual result.

The project was created to explore whether classical audio features and machine learning can recognize emotional stress patterns in speech. It demonstrates the complete lifecycle of an audio ML system: collecting and validating labeled data, preventing speaker leakage, extracting features, training and evaluating a model, serving predictions through an API, and presenting results in an accessible web interface.

> **Important:** This is a research and educational prototype trained primarily on acted emotional speech. It is not a medical device, emergency service, or replacement for human judgment. Never use its prediction as the sole basis for an emergency decision.

## What a User Can Do

- Open the [live web application](https://panic-voice-detector.vercel.app) on a phone or computer.
- Record a voice sample using the device microphone.
- Upload a `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, or `.webm` audio file up to 10 MiB.
- Receive a normal, possible-distress, or strong-distress result.
- View separate distress and normal confidence scores.

Only the first three seconds of each recording are analyzed. Uploaded files are stored temporarily for processing and deleted immediately after the request finishes.

## End-to-End Workflow

1. The browser records audio or accepts a file upload.
2. The Flask API validates the file type and 10 MiB size limit.
3. `librosa` decodes up to three seconds of mono audio.
4. The shared pipeline extracts 40 mean MFCC features.
5. The Random Forest calculates probabilities for `Distress` and `Normal`.
6. The API compares the distress probability with the threshold stored inside the trained model artifact.
7. The frontend displays the result and both confidence scores.

The training workflow is kept separate from inference. It validates labels, removes duplicate audio by SHA-256 content, isolates speakers across train/validation/test splits, applies augmentation only to development data, tunes the threshold on validation speakers, and reports final metrics on untouched test speakers.

---

## Technical Overview

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
A **Random Forest Classifier** (300 estimators) is trained on deduplicated recordings. Speakers are isolated across train, validation, and test sets. Conservative gain, noise, and time-shift augmentation is applied only to development recordings. The validation speakers select a distress threshold subject to a minimum precision target; the untouched test speakers provide the final metrics. The versioned model and tuned threshold are saved together in `model.pkl`.

### 4. Prediction Pipeline
```
Input audio → Feature extraction (MFCC) → model.predict() → "Distress" or "Normal"
```
The web app uses the threshold saved with the trained model. It is tuned on validation speakers to improve distress recall while maintaining at least 50% validation precision. It is not a calibrated emergency-risk score.

### Current Model Results

The checked-in artifact was trained on **1,439 content-unique recordings** and uses a **25.8% distress threshold** selected from validation speakers.

| Untouched-speaker metric | Result |
|---|---:|
| Accuracy | 68.3% |
| Distress precision | 43% |
| Distress recall | 61% |

The lower threshold raises distress recall from the previous 39% to 61%, at the cost of more false alarms. Exact speaker splits, the confusion matrix, classification report, and precision-recall curve are stored in `training_metrics.json`.

### 5. Web Interface
- **Live Recording**: Browser captures mic audio via `MediaRecorder API`, converts it to WAV using `AudioContext`, then sends it to the Flask backend.
- **File Upload**: User provides a supported audio file up to 10 MiB, sent directly to the backend.
- **Shared pipeline** (`audio_pipeline.py`): Defines the audio duration, supported formats, upload limit, threshold, and MFCC extraction used throughout the project.
- **Backend** (`api/index.py`): Flask API validates uploads, extracts MFCCs, runs prediction, and returns JSON with `prediction`, `confidence`, and `normal_confidence`.
- **Frontend** (`index.html`): Shows normal, possible-distress, or strong-distress patterns with animated confidence bars.

---

## Project Structure

```
ML-panic_voice_detector/
├── api/
│   └── index.py          # Flask backend API
├── audio_pipeline.py     # Shared features and training-only augmentation
├── dataset_tools.py      # Manifest validation and content deduplication
├── prepare_dataset.py    # Safe custom recording importer
├── dataset/
│   ├── Actor_01/         # RAVDESS audio files (Actor 01–24)
│   └── ...
├── index.html            # Web frontend (UI)
├── live_predict.py       # File-based CLI prediction script
├── model_artifact.py     # Versioned model and threshold loading
├── train_model.py        # Model training script
├── model.pkl             # Pre-trained Random Forest model
├── training_metrics.json # Split, threshold, test metrics, and PR curve
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

### Add Real Recordings

Create a CSV using [sample_manifest.example.csv](sample_manifest.example.csv):

```csv
file,label,speaker_id
recordings/person_01_panic.wav,Distress,person_01
recordings/person_01_normal.wav,Normal,person_01
```

Paths must be relative to the CSV and labels must be `Normal` or `Distress`. Every person needs a stable, non-identifying `speaker_id`; this is what prevents speaker leakage.

```bash
python prepare_dataset.py path/to/samples.csv --output dataset/custom
python train_model.py
```

The importer decodes every file, rejects unsafe paths and invalid labels, deduplicates content with SHA-256, and creates `dataset/custom/labels.csv`. Do not place augmented copies in the manifest—the trainer creates augmentation only after splitting speakers.

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
python live_predict.py dataset/Actor_01/03-01-06-01-02-01-01.wav
```

Pass an audio file, a directory, or an extensionless audio path as the argument. Live microphone recording is available through the web interface.

### Run Tests

```bash
python -m unittest discover -s tests -v
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

The prediction endpoint accepts `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, and `.webm` audio up to **10 MiB**. Invalid, empty, or oversized uploads are rejected before prediction.

---

## Limitations

- Trained on **acted emotional speech** (RAVDESS), not real-world panic recordings. Real screaming may score lower than expected because the acoustic profile differs from studio-performed fear/anger.
- Best accuracy is achieved with **clear, close-mic audio** similar to the training data.
- Single-shot prediction — analyzes only the first 3 seconds of audio.
- Review `training_metrics.json`, especially distress precision and recall, after every retraining run.
- This is an experimental emotion classifier, not a validated medical, safety, or emergency-response system. Do not use it as the sole basis for emergency decisions.
