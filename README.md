# TSaaS – Twitter Emotion Analysis

Web version of the NetBeans `Twitter_GUI` project. Classifies tweets into 6 emotions
(joy, sadness, anger, fear, love, surprise) with Naive Bayes, SVM, ANN, HySimNet and RNN-LSTM (pure NumPy)
across 6 feature modes (TF-IDF, co-occurrence, bigram, trigram, lemma+TF-IDF, lemma+co-occurrence).

**Metrics are measured** on a stratified 20% held-out split. (The original Java `HySimNet` and
`RNN_LSTM_N` classes hard-coded their accuracy; this version does not.)

## Run locally
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000  (login: admin / admin)
```

## Deploy free on Render (no credit card)
Hugging Face now only offers Docker Spaces on a paid plan, so use Render's free web service instead.
1. Push this repo to GitHub.
2. render.com → New → Web Service → connect the repo. Runtime **Docker**, instance type **Free**.
3. Add environment variables `TSAAS_USER` and `TSAAS_PASSWORD`.
4. Deploy. Models are pre-trained during the Docker build, so the app starts in seconds.

Free instances sleep after ~15 min idle (first request then takes ~1 min). Uploading a new dataset retrains
on the free instance's small CPU and can take several minutes (capped at 3000 rows via `TSAAS_MAX_ROWS`).

## Layout
- `app/nlp.py`, `features.py` – preprocessing, lexicon and feature modes (ported from Java)
- `app/models.py`, `experiment.py` – classifiers and evaluation
- `app/main.py` – FastAPI: login, `/api/run`, `/api/status`, `/api/predict`, `/api/stream` (SSE)
- `app/static/` – web UI
- `tools/extract_java_data.py` – how the lexicon/lemma/stopword data was extracted from the Java source
