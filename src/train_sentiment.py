"""
Basic Sentiment Analysis Using Machine Learning

This script demonstrates a basic workflow for sentiment analysis:
- Text preprocessing
- Feature extraction
- Model training
- Evaluation

This code is intended for educational and research foundation purposes.
"""
import pandas as pd
import pickle
import re
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_dataset():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "dataset", "social_media_reviews.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        return pd.read_csv(csv_path)

    except Exception as e:
        print("❌ Dataset loading failed:", e)
        sys.exit(1)


def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text
    except Exception:
        return ""


def train_and_save_model(data):
    try:
        data["text"] = data["text"].apply(clean_text)

        X = data["text"]
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        predictions = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)

        print(f"✅ Model Accuracy: {accuracy:.2f}")

        # Save files in SAME folder as this script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_dir, "sentiment_model.pkl")
        vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        print("✅ sentiment_model.pkl saved")
        print("✅ vectorizer.pkl saved")

    except Exception as e:
        print("❌ Training failed:", e)
        sys.exit(1)


def main():
    print("🚀 Starting Social Media Sentiment Model Training")
    data = load_dataset()
    train_and_save_model(data)
    print("🎉 Training completed successfully")


if __name__ == "__main__":
    main()

