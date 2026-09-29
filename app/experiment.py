"""Train + evaluate every (classifier x feature-mode) combination with real held-out metrics."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from .nlp import Doc, LABELS
from .features import FeatureBuilder, MODES
from .models import CLASSIFIERS

IDX = {l: i for i, l in enumerate(LABELS)}


def load_dataset(path_or_text):
    """Parse 'text;label' lines. Returns (texts, labels, skipped)."""
    text = path_or_text
    texts, labels, skipped = [], [], 0
    for line in text.splitlines():
        line = line.strip()
        sep = line.rfind(";")
        if sep < 1:
            skipped += 1; continue
        t, l = line[:sep].strip(), line[sep + 1:].strip().lower()
        if not t or l not in IDX:
            skipped += 1; continue
        texts.append(t); labels.append(IDX[l])
    return texts, np.array(labels), skipped


def run_experiments(texts, y, progress=None, test_size=0.2, seed=42, classifiers=None):
    """Returns (results, trained) where trained[(clf, mode)] = (features, model)."""
    docs = [Doc(t) for t in texts]
    tr, te = train_test_split(np.arange(len(docs)), test_size=test_size, random_state=seed, stratify=y)
    dtr, dte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]
    fb = FeatureBuilder().fit(dtr)
    results, trained = [], {}
    classifiers = classifiers or CLASSIFIERS
    total, done = len(MODES) * len(classifiers), 0
    for mode in MODES:
        Xtr, Xte = fb.matrix(dtr, mode), fb.matrix(dte, mode)
        for C in classifiers:
            m = C()
            if progress: progress(done / total, f"Training {m.name} on {mode}")
            t0 = time.time(); m.fit(Xtr, ytr, dtr); train_ms = (time.time() - t0) * 1000
            t0 = time.time(); pred = m.predict(Xte, dte); infer_ms = (time.time() - t0) * 1000
            p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=range(6), average="macro", zero_division=0)
            pc, rc, fc, sc = precision_recall_fscore_support(yte, pred, labels=range(6), zero_division=0)
            results.append({
                "classifier": m.name, "mode": mode, "accuracy": accuracy_score(yte, pred),
                "precision": p, "recall": r, "f1": f, "train_ms": round(train_ms), "infer_ms": round(infer_ms),
                "confusion": confusion_matrix(yte, pred, labels=range(6)).tolist(),
                "per_class": [{"label": LABELS[i], "precision": pc[i], "recall": rc[i], "f1": fc[i], "support": int(sc[i])}
                              for i in range(6)],
            })
            trained[(m.name, mode)] = (fb, m)
            done += 1
    summary = {"total": len(docs), "train": len(dtr), "test": len(dte),
               "distribution": {LABELS[i]: int((y == i).sum()) for i in range(6)},
               "test_idx": te.tolist()}
    if progress: progress(1.0, "Done")
    return results, trained, summary
