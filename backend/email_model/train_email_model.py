"""
train_email_model.py

Trains a text-classification model to distinguish phishing emails from
legitimate emails, using TF-IDF (Term Frequency-Inverse Document
Frequency) features + Logistic Regression -- a standard, well-proven
combination for spam/phishing text classification.

Run this file directly:
    python train_email_model.py

It will:
  1. Load data/email_dataset.csv (a real, published dataset -- see the
     README for its source/citation)
  2. Vectorize the email text with TF-IDF
  3. Train + evaluate a Logistic Regression classifier with 5-fold
     cross-validation
  4. Print a confusion matrix + specificity
  5. Save the fitted vectorizer + model to phishing_email_model.pkl
"""

import os
import csv
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODEL_DIR, "..", "data", "email_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_email_model.pkl")


def clean_text(text: str) -> str:
    """Light text normalization before vectorization."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)  # normalize links
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)  # normalize email addresses
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. See the README for how to obtain "
            "the email dataset."
        )

    texts, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def main():
    print("Loading dataset...")
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} emails "
          f"({sum(labels)} phishing / {len(labels) - sum(labels)} legitimate)")

    print("Cleaning text...")
    cleaned = [clean_text(t) for t in texts]
    y = np.array(labels)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")
    print(f"Fold scores: {np.round(cv_scores, 4)}")

    print("\nTraining final model on the full training split...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nHeld-out test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print("Confusion matrix:")
    print("                 predicted legitimate   predicted phishing")
    print(f"actual legitimate        {cm[0][0]:<20d} {cm[0][1]:<20d}")
    print(f"actual phishing          {cm[1][0]:<20d} {cm[1][1]:<20d}")
    print(f"\nSensitivity (recall for phishing): {sensitivity:.4f}")
    print(f"Specificity (recall for legitimate): {specificity:.4f}")

    # Most predictive words -- useful for a project report's discussion
    # of what the model learned.
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_phishing_idx = np.argsort(coefs)[-10:][::-1]
    top_legit_idx = np.argsort(coefs)[:10]

    print("\nTop 10 words/phrases pushing toward 'phishing':")
    for i in top_phishing_idx:
        print(f"  {feature_names[i]:<20s} weight={coefs[i]:.3f}")

    print("\nTop 10 words/phrases pushing toward 'legitimate':")
    for i in top_legit_idx:
        print(f"  {feature_names[i]:<20s} weight={coefs[i]:.3f}")

    joblib.dump({"vectorizer": vectorizer, "model": clf}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
