
import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import altair as alt
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Speech Emotion Classifier", page_icon="🎧", layout="centered")
st.title("🎧 Speech Emotion Recognition")
st.write("Upload a short audio clip and I’ll predict the emotion it conveys.")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained scikit‑learn model and label encoder once per session."""
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, label_encoder = load_artifacts()

def extract_features(file_path: str) -> np.ndarray:
    """Replicates the pipeline you used in *test.py* (MFCC + chroma + contrast + ZCR)."""
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

uploaded_file = st.file_uploader("📂 Select an audio file (wav / mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

   
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Analysing …"):
        features = extract_features(tmp_path).reshape(1, -1)
        pred_idx = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        emotion = label_encoder.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx]

    st.success(f"🧠 Predicted emotion: **{emotion.capitalize()}**  ({confidence:.2%} confident)")

   
    df = pd.DataFrame({"Emotion": label_encoder.classes_, "Probability": proba})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X("Emotion", sort="-y"), y="Probability:Q", tooltip=["Emotion", "Probability"])
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ & Streamlit · scikit‑learn MLPClassifier")
