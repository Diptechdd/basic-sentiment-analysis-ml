"""
Basic Sentiment Analysis Using Machine Learning

This script demonstrates a basic end-to-end workflow for sentiment analysis:
1. Load dataset
2. Clean text
3. Extract features using TF-IDF
4. Train a classifier
5. Evaluate performance

This code is intended for educational and foundational research purposes.
"""

import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "social_media_reviews.csv")
    return pd.read_csv(data_path)


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# --------------------------------------------------
# Main Training Function
# --------------------------------------------------
def train_model():
    print("🚀 Loading dataset...")
    data = load_dataset()

    # Expected columns: text, sentiment
    data = data.dropna()

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("⚙️ Extracting features...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🧠 Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("📊 Evaluating model...")
    predictions = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    train_model()

