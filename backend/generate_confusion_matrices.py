"""
generate_confusion_matrices.py

Generates confusion matrix chart images (PNG) for both the URL model
and the email model, using the same train/test split each training
script uses (random_state=42), so the numbers exactly match what
train_model.py / train_email_model.py print to the console.

Useful for dropping into a project report (e.g. Chapter 4 -- Results
and Evaluation).

Run from the `backend/` folder, after both models have been trained:
    python generate_confusion_matrices.py

Outputs:
    url_confusion_matrix.png
    email_confusion_matrix.png
"""

import csv
import sys
import os

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "email_model"))

from feature_extractor import extract_features  # noqa: E402
from train_email_model import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_confusion_matrix(cm, title, cmap, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=cmap)

    labels_text = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_text, fontsize=12)
    ax.set_yticklabels(labels_text, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_url_confusion_matrix():
    data_path = os.path.join(BASE_DIR, "data", "dataset.csv")
    model_path = os.path.join(BASE_DIR, "model", "phishing_model.pkl")

    urls, labels = [], []
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))

    X = np.array([extract_features(u) for u in urls])
    y = np.array(labels)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bundle = joblib.load(model_path)
    preds = bundle["model"].predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plot_confusion_matrix(
        cm,
        "URL Phishing Detector\nConfusion Matrix (Random Forest)",
        "Blues",
        os.path.join(BASE_DIR, "url_confusion_matrix.png"),
    )


def generate_email_confusion_matrix():
    data_path = os.path.join(BASE_DIR, "data", "email_dataset.csv")
    model_path = os.path.join(BASE_DIR, "email_model", "phishing_email_model.pkl")

    texts, labels = [], []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))

    cleaned = [clean_text(t) for t in texts]
    y = np.array(labels)

    _, X_test_text, _, y_test = train_test_split(
        cleaned, y, test_size=0.2, random_state=42, stratify=y
    )

    bundle = joblib.load(model_path)
    X_test = bundle["vectorizer"].transform(X_test_text)
    preds = bundle["model"].predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plot_confusion_matrix(
        cm,
        "Email Phishing Detector\nConfusion Matrix (Logistic Regression + TF-IDF)",
        "Greens",
        os.path.join(BASE_DIR, "email_confusion_matrix.png"),
    )


if __name__ == "__main__":
    print("Generating URL model confusion matrix...")
    generate_url_confusion_matrix()

    print("Generating email model confusion matrix...")
    generate_email_confusion_matrix()

    print("\nDone. Both PNG files are saved in the backend/ folder.")
