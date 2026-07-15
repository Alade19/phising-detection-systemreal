"""
app.py

Flask API for the AI Phishing Detection System.
Covers both URL-based and email-based phishing detection.

Endpoints:
    GET    /api/health          -> simple health check

    POST   /api/predict         -> { url } -> URL prediction result
    GET    /api/history         -> last 10 URL predictions
    DELETE /api/history         -> clears URL prediction history

    POST   /api/predict-email   -> { subject, body } -> email prediction result
    GET    /api/email-history   -> last 10 email predictions
    DELETE /api/email-history   -> clears email prediction history

Run locally:
    python app.py
Server starts on http://127.0.0.1:5000
"""

import os
import json
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from model.feature_extractor import extract_features, is_valid_url
from email_model.train_email_model import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

URL_MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_model.pkl")
URL_HISTORY_PATH = os.path.join(BASE_DIR, "data", "history.json")

EMAIL_MODEL_PATH = os.path.join(BASE_DIR, "email_model", "phishing_email_model.pkl")
EMAIL_HISTORY_PATH = os.path.join(BASE_DIR, "data", "email_history.json")

MAX_HISTORY = 10
MIN_EMAIL_LENGTH = 10
MAX_EMAIL_LENGTH = 20000

app = Flask(__name__)
CORS(app)  # allow the Next.js frontend (different origin) to call this API

_history_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model loading (lazy -- loaded once, on first request)
# ---------------------------------------------------------------------------
_url_model_bundle = None
_email_model_bundle = None


def get_url_model():
    global _url_model_bundle
    if _url_model_bundle is None:
        if not os.path.exists(URL_MODEL_PATH):
            raise FileNotFoundError(
                "URL model file not found. Run `python model/train_model.py` first."
            )
        _url_model_bundle = joblib.load(URL_MODEL_PATH)
    return _url_model_bundle


def get_email_model():
    global _email_model_bundle
    if _email_model_bundle is None:
        if not os.path.exists(EMAIL_MODEL_PATH):
            raise FileNotFoundError(
                "Email model file not found. Run "
                "`python email_model/train_email_model.py` first."
            )
        _email_model_bundle = joblib.load(EMAIL_MODEL_PATH)
    return _email_model_bundle


# ---------------------------------------------------------------------------
# Generic history helpers (each history is stored as its own JSON file)
# ---------------------------------------------------------------------------
def _read_history(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _write_history(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _add_to_history(path, entry):
    with _history_lock:
        history = _read_history(path)
        history.insert(0, entry)
        history = history[:MAX_HISTORY]
        _write_history(path, history)
        return history


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# URL phishing detection
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please provide a URL to check."}), 400

    valid, reason = is_valid_url(url)
    if not valid:
        return jsonify({"error": reason}), 400

    try:
        bundle = get_url_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    model = bundle["model"]
    features = np.array([extract_features(url)])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    result = "phishing" if prediction == 1 else "legitimate"

    entry = {
        "url": url,
        "result": result,
        "confidence": round(confidence * 100, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history = _add_to_history(URL_HISTORY_PATH, entry)

    return jsonify({"prediction": entry, "history": history})


@app.route("/api/history", methods=["GET"])
def get_url_history():
    return jsonify({"history": _read_history(URL_HISTORY_PATH)})


@app.route("/api/history", methods=["DELETE"])
def clear_url_history():
    with _history_lock:
        _write_history(URL_HISTORY_PATH, [])
    return jsonify({"message": "History cleared.", "history": []})


# ---------------------------------------------------------------------------
# Email phishing detection
# ---------------------------------------------------------------------------
@app.route("/api/predict-email", methods=["POST"])
def predict_email():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    body = (data.get("body") or "").strip()
    combined = f"{subject} {body}".strip()

    if not combined:
        return jsonify({"error": "Please paste an email subject and/or body to check."}), 400

    if len(combined) < MIN_EMAIL_LENGTH:
        return jsonify({
            "error": "That's too short to analyze -- please paste more of the email."
        }), 400

    if len(combined) > MAX_EMAIL_LENGTH:
        return jsonify({
            "error": f"That email is too long (max {MAX_EMAIL_LENGTH} characters)."
        }), 400

    try:
        bundle = get_email_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    cleaned = clean_text(combined)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    result = "phishing" if prediction == 1 else "legitimate"

    entry = {
        "subject": subject[:200],
        "preview": combined[:120] + ("..." if len(combined) > 120 else ""),
        "result": result,
        "confidence": round(confidence * 100, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history = _add_to_history(EMAIL_HISTORY_PATH, entry)

    return jsonify({"prediction": entry, "history": history})


@app.route("/api/email-history", methods=["GET"])
def get_email_history():
    return jsonify({"history": _read_history(EMAIL_HISTORY_PATH)})


@app.route("/api/email-history", methods=["DELETE"])
def clear_email_history():
    with _history_lock:
        _write_history(EMAIL_HISTORY_PATH, [])
    return jsonify({"message": "History cleared.", "history": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
