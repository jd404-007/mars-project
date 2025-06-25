# Emotion Classification on RAVDESS


A complete, end‑to‑end pipeline that classifies speech/audio clips into **eight emotion categories** using a Multi‑Layer Perceptron (MLP) and hand‑crafted acoustic features. The project is built on the RAVDESS dataset and reaches nearly **79 % accuracy / F1 score  on a held‑out validation set.**

> **Classes:** angry · calm · disgust · fearful · happy · neutral · sad · surprised

---

## 1. Project Motivation

Emotion recognition from speech is foundational for empathetic voice assistants, mental‑health monitoring, and human–robot interaction. While deep end‑to‑end models dominate recent research, many real‑world projects still need *light‑weight*, *inspectable* solutions that run on CPUs and work with limited data.\
This repo demonstrates that carefully engineered **MFCC + chroma + spectral** features, balanced with **SMOTE** and tuned with **GridSearchCV**, can deliver competitive performance with modest compute.

---

## 2. Repository Structure

```
├── final_result.ipynb            # Full training & evaluation notebook
├── preprocessing.ipynb           # Data preprocessing file
├── test_model.py                 # Simple CLI to run inference on new data
├── model.pkl                     # Trained MLP pipeline (scaler ► SMOTE ► MLP)
├── label_encoder.pkl             # Fitted LabelEncoder for emotion labels
├── ravdess_features.csv          # Pre‑extracted features (61 cols + label)
├── test_sample.wav               # Audio sample 
├── demo.mp4                      # 2‑minute screencast of project
└── README.md                     # Readme with requirements and instructions

```

---

## 3. Data & Pre‑processing

| Step                            | Details                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Dataset**                     | [RAVDESS](https://zenodo.org/record/1188976) – 24 actors × 8 emotions, both speech and song modalities. |
| **Audio loading**               | `librosa.load(sr=22050)` – mono, 22 kHz.                                                                |
| **Feature extraction**          | – 40‑dim MFCC mean– 12‑dim chroma mean– 7‑dim spectral‑contrast mean– 1‑dim zero‑crossing‑rate mean.    |
| Total = **61 features / clip**. |                                                                                                         |
| **Label parsing**               | Emotion code parsed from filename (`03‑01‑05‑01‑... → angry`).                                          |
| **Output**                      | Saved to `ravdess_features.csv` (2 452 rows × 61 features + label).                                     |

```python
# snippet
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40); mfcc_mean = mfcc.mean(axis=1)
chroma = librosa.feature.chroma_stft(y, sr).mean(axis=1)
contrast = librosa.feature.spectral_contrast(y, sr).mean(axis=1)
zcr = librosa.feature.zero_crossing_rate(y).mean()
features = np.hstack([mfcc_mean, chroma, contrast, zcr])
```

---

## 4. Model Pipeline

```
StandardScaler ► SMOTE ► MLPClassifier
```

| Component  | Purpose                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| **Scaler** | Z‑score normalisation critical for MLP convergence.                                                    |
| **SMOTE**  | Synthetic minority oversampling to balance the 8 emotion classes.                                      |
| **MLP**    | Two hidden layers **(256, 64)**, ReLU, `alpha=1e‑4`, early‑stopping, tuned with 3‑fold `GridSearchCV`. |

```python
param_grid = {
  'hidden_layer_sizes': [(256, 64), (128, 64), (64,)],
  'activation': ['relu', 'tanh'],
  'alpha': [1e‑4, 1e‑3],
  'learning_rate': ['constant', 'adaptive']
}
```

---

## 5. Performance

| Metric (validation, 491 clips) | Score    |
| ------------------------------ | -------- |
| **Accuracy**                   | **0.79** |
| **Weighted F1**                | **0.79** |
| **Macro F1**                   | 0.77     |

**Per‑class recall**

| angry | calm | disgust | fearful | happy | neutral | sad  | surprised |
| ----- | ---- | ------- | ------- | ----- | ------- | ---- | --------- |
| 85 %  | 83 % | 72 %    | 77 %    | 79 %  | 76 %    | 77 % | 72 %      |

> Confusion matrix and loss‑curve plots are available in the notebook.

---

## 6. Quick Start

```bash
# Clone and install
$ git clone https://github.com/your‑handle/Emotion-Classification-MLP.git
$ cd Emotion-Classification-MLP
$ pip install -r requirements.txt

# Run the notebook (recommended)
$ jupyter notebook emotion_classification.ipynb

# Or test the saved model from CLI
$ python test_model.py --wav path/to/audio.wav
```

`test_model.py` loads **model.pkl**, extracts the same 61‑dim feature vector, scales it, and outputs the predicted emotion.

---

## 7. Demo Video

A 2‑minute walkthrough of training, evaluating and using the model: [`demo.mp4`](demo.mp4)

---

## 8. Requirements

numpy
pandas
librosa
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib


