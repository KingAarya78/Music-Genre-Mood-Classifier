import joblib
import pandas as pd
import numpy as np
import librosa
from src.feature_extraction import extract_features

# 🎵 Map genres to moods
MOOD_MAP = {
    'pop': 'Happy',
    'classical': 'Calm',
    'jazz': 'Calm',
    'rock': 'Energetic',
    'blues': 'Sad',
    'reggae': 'Relaxed',
    'hiphop': 'Energetic',
    'metal': 'Aggressive',
    'country': 'Nostalgic',
    'disco': 'Party'
}

def predict(file_path, model_path='models/rf_genre.pkl'):
    """
    Predicts the genre and mood of an audio file using the trained model.
    Returns: dict with genre, mood, and confidence score.
    """
    try:
        # 🎯 Load trained model pipeline (includes scaler, PCA, classifier)
        pipeline = joblib.load(model_path)
    except Exception as e:
        return {'genre': 'Error', 'mood': 'Error', 'proba': 0.0, 'error': f'Model not found: {e}'}

    try:
        # 🧠 Extract features
        feats = extract_features(file_path)
        df = pd.DataFrame([feats])

        # 🧩 Predict genre
        pred = pipeline.predict(df)[0]

        # 🔢 Confidence (probability of predicted class, if supported)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df)[0]
            pred_index = list(pipeline.classes_).index(pred)
            confidence = float(probs[pred_index])
        else:
            confidence = 1.0  # fallback

        # 💫 Map mood from genre
        mood = MOOD_MAP.get(pred.lower(), 'Neutral')

        return {'genre': pred, 'mood': mood, 'proba': confidence}

    except Exception as e:
        return {'genre': 'Error', 'mood': 'Error', 'proba': 0.0, 'error': str(e)}

# 🔹 CLI test mode
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Predict music genre and mood from an audio file.")
    parser.add_argument('--file', type=str, required=True, help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument('--model', type=str, default='models/rf_genre.pkl', help="Path to trained model file")
    args = parser.parse_args()

    result = predict(args.file, args.model)
    print(result)
