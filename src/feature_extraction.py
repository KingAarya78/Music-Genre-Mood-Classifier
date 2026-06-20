import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path, sr=22050, n_mfcc=13):
    # Load audio
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)
    
    # Feature extraction
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Collect stats
    features = {
        'zcr_mean': np.mean(zcr), 'zcr_std': np.std(zcr),
        'rms_mean': np.mean(rms), 'rms_std': np.std(rms),
        'tempo': tempo
    }

    # MFCCs
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])

    # Chroma
    for i in range(chroma.shape[0]):
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i}_std'] = np.std(chroma[i])

    # Spectral Contrast
    for i in range(contrast.shape[0]):
        features[f'contrast_{i}_mean'] = np.mean(contrast[i])
        features[f'contrast_{i}_std'] = np.std(contrast[i])

    # Tonnetz
    for i in range(tonnetz.shape[0]):
        features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
        features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])

    return features


def create_feature_csv(dataset_dir, save_path='data/features.csv'):
    rows = []
    for genre in os.listdir(dataset_dir):
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fname in os.listdir(genre_dir):
            if not fname.lower().endswith(('.wav', '.mp3')):
                continue
            path = os.path.join(genre_dir, fname)
            print(f'Processing: {path}')
            try:
                feats = extract_features(path)
                feats['label'] = genre
                feats['file'] = fname
                rows.append(feats)
            except Exception as e:
                print(f'Error processing {fname}: {e}')
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f'Saved features to {save_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='data/features.csv')
    args = parser.parse_args()
    create_feature_csv(args.dataset_dir, args.out)
