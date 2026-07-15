# 🔍 AI Phishing Detection System

An AI-powered web app that checks whether a **URL** or an **email**
looks like a phishing attempt, using two separate machine learning
models trained on real, published datasets.

It's built as two separate pieces that talk to each other:

| Part | Tech | What it does |
|---|---|---|
| **Backend** | Python + Flask + scikit-learn | Two ML pipelines — one for URLs, one for email text — each extracting features and running them through a trained model to classify phishing vs. legitimate. Stores your last 10 predictions for each. |
| **Frontend** | Next.js + React + Tailwind CSS | The clean web interface with two tabs: **Check URL** and **Check Email**. |

---

## 1. How it actually works (the "AI" part)

The model doesn't visit the website — it looks at the **text of the URL
itself** and pulls out ~20 numeric signals that tend to differ between real
and fake links, for example:

- Is there an IP address instead of a domain name? (`http://192.168.1.5/login`)
- Does it use a URL shortener? (`bit.ly/...`)
- Does it contain scam-y words like `verify`, `login`, `secure`, `account`?
- How long is the URL? How many hyphens/dots/digits does it have?
- Does the domain look "random" (high entropy), like `xj29fqz-login.tk`?
- Does it use a suspicious top-level domain like `.tk`, `.xyz`, `.zip`?

These numbers are fed into a **Random Forest classifier** (a standard,
well-understood ML algorithm) that outputs a `phishing` / `legitimate` label
plus a confidence score.

### The dataset (real, published, academically citable)

`backend/data/dataset.csv` contains **11,430 real URLs** (5,715 legitimate /
5,715 phishing — perfectly balanced), sourced from a published academic
study:

> Hannousse, A., & Yahiouche, S. (2021). *Towards benchmark datasets for
> machine learning based website phishing detection: An experimental study.*
> Engineering Applications of Artificial Intelligence.
> Dataset: https://data.mendeley.com/datasets/c2gw7fy2j4/3
> (mirrored via https://github.com/Trieuh2/ml-url-phishing-classifier)

Only the raw `url` and its `status` (phishing/legitimate) label were kept —
this project's own `feature_extractor.py` re-derives its ~20 lexical
features directly from each URL string at both training and prediction
time, so training and live predictions always use identical logic.

**Current results (5-fold cross-validation + held-out test set):**
- Cross-validation accuracy: **~86%** (std. dev ~0.8%)
- Held-out test accuracy: **~86%**
- Precision/recall for both classes: **~0.83–0.89**

This is a believable, defensible number for a lexical/URL-only classifier —
unlike the earlier 100% figure from synthetic data, this reflects genuine
generalization to real-world phishing patterns the model wasn't explicitly
told about. `train_model.py` also prints a full confusion matrix and the
top 8 most predictive features every time it runs, useful for your project
report/evaluation section.

**Want an actual confusion matrix chart image (not just text) for your
report?** Run this after training both models:
```bash
python generate_confusion_matrices.py
```
This saves `url_confusion_matrix.png` and `email_confusion_matrix.png`
in the `backend/` folder — ready to drop straight into a Chapter 4 /
Results section.

> Want to regenerate the old procedurally-generated demo dataset instead?
> Delete `data/dataset.csv` and rerun `train_model.py` — it will fall back
> to `generate_dataset.py` automatically.

### URL validation (before the model ever sees it)

Not every string typed into the box is a real URL. Before feature
extraction happens, `feature_extractor.py`'s `is_valid_url()` function
checks the input against these conditions, in order, and rejects it
with a specific reason if any fail:

| # | Condition | Example rejected input |
|---|---|---|
| 1 | Not empty | (blank input) |
| 2 | At least 4 characters | `"v"`, `"ab"` |
| 3 | No longer than 2048 characters | (extremely long strings) |
| 4 | Contains no spaces | `"hello world"` |
| 5 | Only `http://` / `https://` schemes allowed (or no scheme) | `"javascript:alert(1)"` |
| 6 | Has a domain portion at all | `"https://"` |
| 7 | Domain is either a valid IPv4 address, or contains at least one dot | `"xyz"` (no dot, not an IP) |
| 8 | No empty labels between dots | `"example..com"` |
| 9 | Domain only contains valid URL characters (letters, digits, dots, hyphens) | `"exa mple.com"` |
| 10 | Ends in a plausible TLD (2+ letters) | `"example.1"` |

If a URL passes all 10 checks, it's considered structurally valid and
is passed to the model for classification. This was added specifically
to stop the model from being asked to judge meaningless input (e.g. a
single letter) — see the "Known limitations" section below for why
that mattered.

---

## 2. The Email Phishing Detector

The second half of the system classifies **email text** (subject + body)
as phishing or legitimate, using a completely separate model and
pipeline from the URL detector — different data, different features,
different algorithm — because email phishing detection is a text
classification problem, not a lexical-pattern problem like URLs.

### How it works

1. **Text cleaning** (`email_model/train_email_model.py`'s `clean_text()`):
   lowercases the text, replaces links with a placeholder token, replaces
   email addresses with a placeholder token, and strips punctuation.
2. **TF-IDF vectorization**: converts the cleaned text into up to 5,000
   weighted word/phrase features (unigrams + bigrams), where words that
   appear disproportionately in one class (e.g. "verify", "urgent",
   "click") get more weight than common words.
3. **Logistic Regression classifier**: a standard, well-understood
   algorithm for text classification, trained on those TF-IDF features.

### The dataset (real, published, academically citable)

`backend/data/email_dataset.csv` contains **12,000 real emails** (6,000
phishing / 6,000 legitimate — balanced by undersampling), built from the
**CEAS 2008 Phishing Corpus**, part of a larger published, peer-reviewed
benchmark:

> Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A.,
> & Zaman, S. A. U. (2024). *Novel Interpretable and Robust Web-based
> AI Platform for Phishing Email Detection.*
> Source data mirror: https://github.com/rokibulroni/Phishing-Email-Dataset

Only the `subject`, `body`, and `label` columns were used; text was
capped at 800 characters per email to keep the dataset lightweight and
training fast.

**Current results (5-fold cross-validation + held-out test set):**
- Cross-validation accuracy: **~98.8%**
- Held-out test accuracy: **~98.8%**
- Sensitivity (phishing recall): **~98.8%**
- Specificity (legitimate recall): **~98.7%**

This high accuracy is expected and believable for this type of dataset —
TF-IDF-based spam/phishing text classifiers commonly achieve 97–99% on
similar benchmarks, because phishing email *language* (urgency, scams,
requests for credentials) is lexically quite distinct from the everyday
legitimate email text in the corpus. `train_email_model.py` also prints
the top 10 words pushing toward each class every time it runs — useful
for your report's discussion of what the model learned.

> ⚠️ **Known limitation, worth noting in your report:** the "legitimate"
> half of this dataset comes largely from a technical mailing-list corpus
> (Python core developers). That means words like "python", "wrote", or
> "opensuse" are strong legitimate signals *in this dataset* — but that's
> a property of this particular corpus, not a universal rule. A legitimate
> email from, say, a retail newsletter or a personal message might not
> look like this training data. This is a good, honest limitation to
> discuss in your evaluation chapter (see Chapter 4 guidance further down
> if you asked for it).

### Email input validation

Before an email reaches the model, `app.py`'s `/api/predict-email` route
checks:
- Subject and body aren't both empty
- Combined text is at least 10 characters (rejects near-empty input)
- Combined text is no more than 20,000 characters (rejects unreasonably
  huge pastes)

---

## 3. About the "Retrain Model" button

Your screenshot had a **Retrain Model** button — I **removed it** in this
version, for a practical reason:

You mentioned you'll deploy this on **Vercel**. Vercel's backend
functions are "serverless" — every request may run on a fresh, temporary
container with no persistent disk. A "Retrain" button needs to:
1. Save new training data somewhere permanent, and
2. Save the newly trained model file somewhere permanent, and
3. Take real time to run (training isn't instant).

None of that works reliably on a stateless serverless host, so a retrain
button would look like it works locally but silently fail (or reset) in
production — which would be a confusing, broken feature for your users.

**If you still want it:** it's easy to add back if you host the backend
yourself on a normal server/VM (Render, Railway, a VPS, etc.) with a
persistent disk — you'd add one more route in `backend/app.py` that calls
`model/train_model.py` and reloads the model. I kept the codebase modular
specifically so this is a small addition later if you need it.

---

## 4. Project structure

```
phishing-url-detector/
├── backend/                       # Python / Flask API
│   ├── app.py                     # Main Flask app (URL + email routes)
│   ├── requirements.txt
│   ├── model/                     # URL detector
│   │   ├── feature_extractor.py   # Turns a URL into numeric features + validation
│   │   ├── generate_dataset.py    # Fallback synthetic dataset generator
│   │   ├── train_model.py         # Trains & saves the URL ML model
│   │   └── phishing_model.pkl     # The trained URL model (generated by you)
│   ├── email_model/                # Email detector
│   │   ├── train_email_model.py   # Cleans text, trains & saves the email model
│   │   └── phishing_email_model.pkl  # The trained email model (generated by you)
│   └── data/
│       ├── dataset.csv            # Real URL training data (Hannousse & Yahiouche, 2021)
│       ├── email_dataset.csv      # Real email training data (CEAS 2008 corpus)
│       ├── history.json           # Stores your last 10 URL predictions
│       └── email_history.json     # Stores your last 10 email predictions
│
├── frontend/                      # Next.js + Tailwind CSS website
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # The main UI (URL tab + Email tab)
│       │   ├── layout.tsx
│       │   └── globals.css
│       └── lib/
│           └── api.ts             # Talks to the Flask backend (both models)
│
└── README.md                      # You are here
```

---

## 5. Running it on your own computer

You need **Python 3.10+ (3.12 recommended)** and **Node.js 18+** installed.

### Step 1 — Start the backend

```bash
cd backend
python -m venv venv                 # create a virtual environment
source venv/bin/activate            # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train both models (only needs to be done once)
python model/train_model.py
python email_model/train_email_model.py

# Start the API server
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```
Leave this terminal running.

### Step 2 — Start the frontend

Open a **new** terminal:

```bash
cd frontend
npm install
cp .env.local.example .env.local    # tells the frontend where the backend is
npm run dev
```

Open **http://localhost:3000** in your browser. You'll see two tabs —
**Check URL** and **Check Email** — each calling its own model on the
Flask backend.

---

## 6. What each button does

| Button / element | What it does |
|---|---|
| **Check URL** tab | Switches to the URL detector. |
| **Check Email** tab | Switches to the email detector. |
| **Check URL** button | Sends the URL to `/api/predict`, gets back `phishing`/`legitimate` + confidence, adds it to the URL history table. |
| **Check Email** button | Sends the subject + body to `/api/predict-email`, gets back `phishing`/`legitimate` + confidence, adds it to the email history table. |
| **Last 10 Predictions / Email Checks** table | Automatically keeps your most recent 10 checks (oldest ones drop off) — each tab has its own separate history. |
| **Clear History** | Deletes all saved predictions for the currently active tab. |

---

## 7. Deploying it for real

### Frontend → Vercel (as you planned)
1. Push this repo to GitHub.
2. In Vercel, "Import Project" and point it at the `frontend/` folder
   (set **Root Directory** to `frontend` in the Vercel project settings).
3. Add an environment variable in Vercel:
   - `NEXT_PUBLIC_API_URL` = the URL of your deployed backend (see below)
4. Deploy. Vercel auto-detects Next.js — no extra config needed.

### Backend → needs a Python-friendly host
Vercel is not a great fit for a stateful Flask app with a `.pkl` model
file and a JSON history file. Good free/cheap options instead:
- **[Render](https://render.com)** — easiest, has a free tier for small Flask apps
- **[Railway](https://railway.app)**
- **[PythonAnywhere](https://www.pythonanywhere.com)**
- Any VPS (DigitalOcean, Linode, etc.) running `gunicorn app:app`

Whichever you pick, make sure to run both training scripts once during
the build step (or commit the generated `.pkl` files to the repo — a
few MB each) so both model files exist when the app starts:
```bash
python model/train_model.py
python email_model/train_email_model.py
```

A `gunicorn` entry is already in `requirements.txt` for production:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Once your backend has a public URL (e.g. `https://your-app.onrender.com`),
set `NEXT_PUBLIC_API_URL` to that in Vercel's environment variables.

---

## 8. Improving the models further

Both models already train on real, published datasets (see above). To
push accuracy higher:

**URL model:**
1. **Combine datasets.** Merge in more real URLs from sources like
   [PhishTank](https://www.phishtank.com) (phishing) and the
   [Tranco list](https://tranco-list.eu) (legitimate, top-ranked real
   domains) — append them to `backend/data/dataset.csv` in the same
   `url,label` format and rerun `train_model.py`.
2. **Try other algorithms.** Edit `train_model.py` to try
   `GradientBoostingClassifier` or `LogisticRegression` instead of
   `RandomForestClassifier` and compare cross-validation scores.
3. **Tune hyperparameters.** Try different `n_estimators` / `max_depth`
   values, or use `GridSearchCV` for a more systematic search.
4. **Add more features.** The original source dataset actually has ~87
   columns, including some that require live web requests (domain age,
   WHOIS data, page rank, DNS records). Our `feature_extractor.py`
   deliberately sticks to features computable instantly from the URL text
   alone (no network calls) — a natural "future work" discussion point.

**Email model:**
1. **Add more sources.** The same combined corpus (Al-Subaiey et al.,
   2024) also includes Enron, Ling, Nazario, Nigerian Fraud, and
   SpamAssassin subsets — mix in more of them for greater diversity
   beyond the CEAS 2008 subset used here.
2. **Try other algorithms.** Naive Bayes, SVM, or XGBoost are common
   alternatives to Logistic Regression for TF-IDF text classification.
3. **Try transformer-based models.** For a more advanced approach (and
   to more fully match "transformer-based models" language in a project
   brief), fine-tuning a small model like DistilBERT on this same
   dataset would be the natural next step — a heavier lift than the
   classical ML approach used here, but worth mentioning as future work.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Frontend says "Failed to check URL" / "Failed to check email" | Make sure the Flask backend is running on port 5000, and that `NEXT_PUBLIC_API_URL` in `frontend/.env.local` matches its address. |
| `ModuleNotFoundError` in Python | Make sure you activated the virtual environment and ran `pip install -r requirements.txt`. |
| "Model file not found" error | Run `python model/train_model.py` and/or `python email_model/train_email_model.py` inside `backend/` before starting `app.py`. |
| CORS errors in browser console | The backend already has `flask-cors` enabled for all origins — double check the backend is actually running and reachable at the URL in `.env.local`. |
| "That doesn't look like a valid URL" for a URL you know is real | Check it matches the conditions in the URL validation table above — the checker is intentionally strict about structure. |

---

## 10. Disclaimer

This is an educational/demo project. Both models are trained on real
datasets, but on limited slices of the phishing landscape (see the
"known limitations" notes above for each model) — they are **not a
substitute for a real security product**. Never rely solely on this
tool to decide whether a link or email is safe.
