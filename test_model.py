
import os
import librosa
import numpy as np
import joblib

# ----------------- USER INPUT HERE -----------------
PATH_TO_INPUT = "./test_sample.wav"  # 👈 change this to your .wav file path
# ---------------------------------------------------

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    return np.hstack((mfcc_mean, chroma_mean, contrast_mean, zcr_mean))

# Load model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # 👈 make sure you save it earlier!

# Extract features from file
features = extract_features(PATH_TO_INPUT).reshape(1, -1)

# Predict
predicted_class_index = model.predict(features)[0]
predicted_proba = model.predict_proba(features)[0]
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]  # 👈 decode

confidence = predicted_proba[predicted_class_index]

# Output
print(f"\n🔊 Input file: {PATH_TO_INPUT}")
print(f"🧠 Predicted emotion: **{predicted_class_label.upper()}**")
print(f"📈 Confidence: {confidence:.2f}")
