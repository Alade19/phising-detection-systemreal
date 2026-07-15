"""
train_model.py

Trains a RandomForestClassifier to distinguish phishing URLs from
legitimate URLs, using the lexical features defined in
feature_extractor.py, and saves the trained model to
model/phishing_model.pkl.

Run this file directly:
    python train_model.py

It will:
  1. Load data/dataset.csv (a real, published, labeled dataset by
     default -- see the README for its source/citation)
  2. Extract features for every URL
  3. Train + evaluate a RandomForest model with 5-fold cross-validation
  4. Print a confusion matrix
  5. Save the trained model + feature names to phishing_model.pkl
"""

import os
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_extractor import extract_features, FEATURE_NAMES
import generate_dataset

MODEL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODEL_DIR, "..", "data", "dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")


def load_dataset():
    if not os.path.exists(DATA_PATH):
        print("No dataset found, generating a synthetic one...")
        rows = generate_dataset.generate(1500)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "label"])
            writer.writerows(rows)

    urls, labels = [], []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))
    return urls, labels


def main():
    print("Loading dataset...")
    urls, labels = load_dataset()
    print(f"Loaded {len(urls)} URLs "
          f"({sum(labels)} phishing / {len(labels) - sum(labels)} legitimate)")

    print("Extracting features...")
    X = np.array([extract_features(u) for u in urls])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )

    # 5-fold cross-validation on the training split -- gives a more
    # trustworthy estimate of real-world performance than a single
    # train/test split, and is standard practice to report in a
    # project write-up.
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
    print("Confusion matrix:")
    print("                 predicted legitimate   predicted phishing")
    print(f"actual legitimate        {cm[0][0]:<20d} {cm[0][1]:<20d}")
    print(f"actual phishing          {cm[1][0]:<20d} {cm[1][1]:<20d}")

    # Feature importance -- useful for a project report to discuss
    # *which* signals mattered most.
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    print("\nTop 8 most important features:")
    for name, score in importances[:8]:
        print(f"  {name:<25s} {score:.4f}")

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

