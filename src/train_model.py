import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib, os


def clean_numeric_strings(df):
    """
    Converts stringified numeric lists like '[99.38]' or '[0.123, 0.456]' to float or array of floats.
    """
    def try_convert(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            # Remove brackets and split if multiple numbers exist
            nums = x.strip('[]').split(',')
            try:
                vals = [float(n.strip()) for n in nums]
                return np.mean(vals) if len(vals) > 1 else vals[0]
            except:
                return x
        return x

    return df.applymap(try_convert)


def train(features_csv, model_out='models/rf_genre.pkl'):
    start_time = time.time()

    print("🔍 Loading feature data from:", features_csv)
    df = pd.read_csv(features_csv)

    # 🧹 Clean up any stringified numeric values like "[99.38]"
    df = clean_numeric_strings(df)

    # Drop unused columns
    X = df.drop(['label', 'file'], axis=1)
    y = df['label']

    print(f"✅ Loaded {len(df)} samples with {X.shape[1]} features.\n")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("🚀 Training model... (this may take a while)")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))
    print(f"🎯 Overall Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"\n💾 Model saved successfully to: {model_out}")

    elapsed = time.time() - start_time
    print(f"⏱️ Training completed in {elapsed/60:.2f} minutes.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Music Genre-Mood Classification Model")
    parser.add_argument('--features', type=str, default='data/features.csv', help="Path to extracted features CSV")
    parser.add_argument('--out', type=str, default='models/rf_genre.pkl', help="Output model path")
    args = parser.parse_args()

    train(args.features, args.out)
