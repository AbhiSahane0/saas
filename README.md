---
title: TSaaS Twitter Emotion Analysis
emoji: 🐦
sdk: docker
app_port: 7860
pinned: false
---

# TSaaS – Twitter Emotion Analysis

Web version of the NetBeans `Twitter_GUI` project. Classifies tweets into 6 emotions
(joy, sadness, anger, fear, love, surprise) with Naive Bayes, SVM, ANN, HySimNet and RNN-LSTM
across 6 feature modes (TF-IDF, co-occurrence, bigram, trigram, lemma+TF-IDF, lemma+co-occurrence).

**Metrics are measured** on a stratified 20% held-out split. (The original Java `HySimNet` and
`RNN_LSTM_N` classes hard-coded their accuracy; this version does not.)

## Run locally
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000  (login: admin / admin)
```

## Deploy free on Hugging Face Spaces
1. Create an account at huggingface.co → New Space → SDK **Docker** → Blank → Public.
2. Push this folder to the Space's git repo (or drag-and-drop the files in the web UI).
3. Space Settings → *Variables and secrets*: add secrets `TSAAS_USER` and `TSAAS_PASSWORD`.
4. It builds automatically; your app is live at `https://<user>-<space>.hf.space`.

Free Spaces sleep after inactivity and take ~1 min to wake; models retrain on each start (~15 s).

## Layout
- `app/nlp.py`, `features.py` – preprocessing, lexicon and feature modes (ported from Java)
- `app/models.py`, `experiment.py` – classifiers and evaluation
- `app/main.py` – FastAPI: login, `/api/run`, `/api/status`, `/api/predict`, `/api/stream` (SSE)
- `app/static/` – web UI
- `tools/extract_java_data.py` – how the lexicon/lemma/stopword data was extracted from the Java source
