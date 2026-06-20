# 🎧 Music Genre & Mood Classification AI

## Overview

Music Genre & Mood Classification AI is a Machine Learning project that automatically predicts the **genre** and **mood** of an uploaded music file using audio signal processing and classification algorithms.

The system extracts meaningful audio features such as:

* MFCC (Mel Frequency Cepstral Coefficients)
* Chroma Features
* Spectral Contrast
* Tonnetz Features
* Zero Crossing Rate
* Root Mean Square Energy
* Tempo

These features are processed through a trained Random Forest classifier to predict the music genre. Based on the predicted genre, an appropriate mood is assigned.

The project also provides a modern Streamlit web interface that allows users to upload audio files and visualize:

* Genre Prediction
* Mood Prediction
* Waveform
* Spectrogram

---

# Features

✅ Music Genre Classification

✅ Music Mood Prediction

✅ Audio Feature Extraction using Librosa

✅ Random Forest Machine Learning Model

✅ PCA-Based Feature Reduction

✅ Interactive Streamlit Web Interface

✅ Waveform Visualization

✅ Spectrogram Visualization

✅ Confidence Score Display

---

# Dataset

This project uses the GTZAN Music Genre Dataset.

Dataset Download:

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Alternative Source:

http://marsyas.info/downloads/datasets.html

---

## Dataset Structure

Place the downloaded dataset inside:

```text
data/
└── genres_original/
    ├── blues/
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
```

---

# Technologies Used

### Programming Language

* Python 3.11+

### Libraries

* Streamlit
* Librosa
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Joblib

### Machine Learning

* Random Forest Classifier
* PCA (Principal Component Analysis)
* StandardScaler

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/music-genre-mood-classification-ai.git

cd music-genre-mood-classification-ai
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Generate Features

```bash
python src/feature_extraction.py --dataset_dir data/genres_original --out data/features.csv
```

---

# Train Model

```bash
python src/train_model.py --features data/features.csv --out models/rf_genre.pkl
```

---

# Run Application

```bash
streamlit run app/streamlit_app.py
```

The application will open automatically in your browser.

---

# Project Structure

```text
Music Genre-Mood Classification AI
│
├── app
│   └── streamlit_app.py
│
├── data
│   ├── features.csv
│   └── genres_original
│
├── models
│   └── rf_genre.pkl
│
├── src
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── predict.py
│   └── train_model.py
│
├── screenshots
│   └── output.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Output Screenshot

After uploading an audio file, the application predicts the genre and mood and displays waveform and spectrogram visualizations.

Replace the image below with your own screenshot.

```markdown
![Application Output](screenshots/output.png)
```

Example:

<p align="center">
  <img src="screenshots/output.png" width="900">
</p>

---

# Sample Prediction

| Feature    | Result |
| ---------- | ------ |
| Genre      | Pop    |
| Mood       | Happy  |
| Confidence | 92%    |

---

# Machine Learning Pipeline

1. Audio File Upload
2. Feature Extraction
3. Feature Scaling
4. PCA Dimensionality Reduction
5. Random Forest Classification
6. Genre Prediction
7. Mood Mapping
8. Result Visualization

---

# Future Enhancements

* Deep Learning (CNN + Spectrogram Images)
* LSTM-Based Audio Analysis
* Real-Time Music Classification
* Spotify API Integration
* Multi-Genre Detection
* Advanced Mood Analysis
* Deployment on Streamlit Cloud

---

# Author

**King Aarya**

Third-Year Artificial Intelligence & Machine Learning Engineering Student

Project: Music Genre & Mood Classification AI

---

# License

This project is developed for educational and research purposes.
