"""TSaaS web app: login, train/compare classifiers, classify text, live emotion stream."""
import asyncio, json, os, random, secrets, threading
from pathlib import Path
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .experiment import load_dataset, run_experiments
from .nlp import Doc, LABELS

ROOT = Path(__file__).parent
USER = os.getenv("TSAAS_USER", "admin")
PASSWORD = os.getenv("TSAAS_PASSWORD", "admin")  # demo default - override via env var when deploying
MAX_ROWS = 20000

app = FastAPI(title="TSaaS - Twitter Emotion Analysis")
sessions: set[str] = set()
state = {"state": "idle", "progress": 0.0, "message": "", "results": [], "summary": None, "error": None}
trained: dict = {}      # (classifier, mode) -> (FeatureBuilder, model)
pool: list = []         # (text, true label) used by the live stream
lock = threading.Lock()


def require_login(request: Request):
    if request.cookies.get("session") not in sessions:
        raise HTTPException(401, "Not logged in")


class Login(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: Login, response: Response):
    if not (secrets.compare_digest(body.username, USER) and secrets.compare_digest(body.password, PASSWORD)):
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_urlsafe(24)
    sessions.add(token)
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    sessions.discard(request.cookies.get("session"))
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/me")
def me(_=Depends(require_login)):
    return {"user": USER}


def _job(texts, y):
    global trained, pool
    def progress(f, msg):
        state.update(progress=round(f, 3), message=msg)
    try:
        results, models, summary = run_experiments(texts, y, progress)
        test_texts = summary.pop("test_texts")
        # rebuild the held-out labels in the same split order for the live stream
        from sklearn.model_selection import train_test_split
        _, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=42, stratify=y)
        with lock:
            trained = models
            pool = [(texts[i], int(y[i])) for i in te]
            state.update(state="done", results=results, summary=summary, error=None)
    except Exception as e:  # surface to the UI instead of dying silently
        state.update(state="error", error=str(e))


def start_job(texts, y):
    with lock:
        if state["state"] == "running":
            raise HTTPException(409, "A run is already in progress")
        state.update(state="running", progress=0.0, message="Starting...", error=None)
    threading.Thread(target=_job, args=(texts, y), daemon=True).start()


@app.on_event("startup")
def warm_start():
    texts, y, _ = load_dataset((ROOT / "data" / "dataset.txt").read_text(encoding="utf-8"))
    start_job(texts, y)


@app.post("/api/run")
async def run(file: UploadFile | None = File(None), _=Depends(require_login)):
    if file is None or not file.filename:
        texts, y, skipped = load_dataset((ROOT / "data" / "dataset.txt").read_text(encoding="utf-8"))
    else:
        raw = await file.read()
        if len(raw) > 5_000_000:
            raise HTTPException(413, "File too large (5 MB max)")
        texts, y, skipped = load_dataset(raw.decode("utf-8", errors="ignore"))
    if len(texts) < 60 or len(set(y.tolist())) < 2:
        raise HTTPException(400, f"Need at least 60 valid 'text;label' rows with 2+ classes (got {len(texts)}, skipped {skipped}). "
                                 f"Labels must be one of: {', '.join(LABELS)}")
    if len(texts) > MAX_ROWS:
        idx = random.Random(1).sample(range(len(texts)), MAX_ROWS)
        texts, y = [texts[i] for i in idx], y[idx]
    start_job(texts, y)
    return {"started": True, "rows": len(texts), "skipped": skipped}


@app.get("/api/status")
def status(_=Depends(require_login)):
    return state


def _best_per_classifier():
    best = {}
    for r in state["results"]:
        if r["classifier"] not in best or r["accuracy"] > best[r["classifier"]]["accuracy"]:
            best[r["classifier"]] = r
    return best


def _predict(text: str):
    doc = Doc(text)
    out = {}
    for name, r in _best_per_classifier().items():
        fb, model = trained[(name, r["mode"])]
        X = fb.matrix([doc], r["mode"])
        out[name] = {"emotion": LABELS[int(model.predict(X, [doc])[0])], "mode": r["mode"],
                     "accuracy": r["accuracy"]}
    return out


class Text(BaseModel):
    text: str


@app.post("/api/predict")
def predict(body: Text, _=Depends(require_login)):
    if state["state"] != "done":
        raise HTTPException(409, "Models are still training")
    if not body.text.strip() or len(body.text) > 1000:
        raise HTTPException(400, "Enter 1-1000 characters")
    preds = _predict(body.text)
    best = max(preds.items(), key=lambda kv: kv[1]["accuracy"])
    return {"predictions": preds, "consensus": best[1]["emotion"], "best_model": best[0]}


JETSTREAM = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"


def _is_english(text: str) -> bool:
    """The author's language tag is unreliable, so double-check the text itself."""
    import re
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    body = re.sub(r"https?://\S+|www\.\S+|\S+\.\S+/\S*|[@#]\S+", " ", text)
    if len(re.findall(r"[A-Za-z]{2,}", body)) < 3:
        return False
    try:
        top = detect_langs(body)[0]
    except Exception:
        return False
    return top.lang == "en" and top.prob >= 0.9


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _bluesky_posts(keyword: str):
    """Yield English Bluesky posts (tag + detected language) in real time from the public Jetstream firehose (no login needed).
    Throttled to ~1 post/sec; auto-reconnects."""
    import ssl, certifi, websockets
    ctx = ssl.create_default_context(cafile=certifi.where())
    kw, last = keyword.lower().strip(), 0.0
    while True:
        try:
            async with websockets.connect(JETSTREAM, ssl=ctx, open_timeout=15) as ws:
                async for raw in ws:
                    m = json.loads(raw)
                    c = m.get("commit") or {}
                    rec = c.get("record") or {}
                    text = (rec.get("text") or "").strip()
                    if c.get("operation") != "create" or "en" not in (rec.get("langs") or []):
                        continue
                    if rec.get("reply") or not (20 <= len(text) <= 500):
                        continue                      # skip replies and very short posts
                    if kw and kw not in text.lower():
                        continue
                    now = asyncio.get_event_loop().time()
                    if now - last < 1.0 or not _is_english(text):
                        continue
                    last = now
                    yield text, f"https://bsky.app/profile/{m['did']}/post/{c['rkey']}"
        except Exception as e:
            yield None, f"Connection to Bluesky lost ({type(e).__name__}); retrying..."
            await asyncio.sleep(3)


@app.get("/api/stream")
async def stream(request: Request, source: str = "dataset", q: str = "", _=Depends(require_login)):
    """Server-sent events. source=dataset replays held-out tweets; source=bluesky streams live posts."""
    if state["state"] != "done":
        raise HTTPException(409, "Models are still training")
    if source not in ("dataset", "bluesky"):
        raise HTTPException(400, "source must be 'dataset' or 'bluesky'")
    best = max(_best_per_classifier().items(), key=lambda kv: kv[1]["accuracy"])
    fb, model = trained[(best[0], best[1]["mode"])]

    def classify(text):
        doc = Doc(text)
        return LABELS[int(model.predict(fb.matrix([doc], best[1]["mode"]), [doc])[0])]

    async def gen():
        if source == "bluesky":
            async for text, info in _bluesky_posts(q[:50]):
                if await request.is_disconnected():
                    break
                if text is None:
                    yield _sse({"notice": info}); continue
                yield _sse({"text": text, "predicted": classify(text), "url": info, "model": best[0], "mode": best[1]["mode"], "accuracy": best[1]["accuracy"]})
        else:
            items = pool[:]
            random.shuffle(items)
            for text, true in items:
                if await request.is_disconnected():
                    break
                yield _sse({"text": text, "predicted": classify(text), "actual": LABELS[true], "model": best[0], "mode": best[1]["mode"], "accuracy": best[1]["accuracy"]})
                await asyncio.sleep(1.2)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
