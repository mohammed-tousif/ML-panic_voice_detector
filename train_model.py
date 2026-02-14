import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

DATASET_PATH = "dataset"

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            emotion_code = file.split("-")[2]

            if emotion_code in ["05", "06"]:  # Angry (05), Fear (06)
                label = "Distress"
            else:
                label = "Normal"

            mfcc_features = extract_features(file_path)
            features.append(mfcc_features)
            labels.append(label)


X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully!")
