"""Preprocessing + emotion lexicon (port of Preprocessor.java / EmotionLexicon.java)."""
import json, re
from pathlib import Path

DATA = Path(__file__).parent / "data"
LABELS = ["joy", "sadness", "anger", "fear", "love", "surprise"]
LEXICON = {w: v for w, v in json.loads((DATA / "lexicon.json").read_text()).items()}
LEMMA = json.loads((DATA / "lemmas.json").read_text())
STOPWORDS = set(json.loads((DATA / "stopwords.json").read_text()))


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(w: str) -> str:
    if len(w) < 4:
        return w
    for suf, keep, rep in (("ness", 0, ""), ("ment", 0, ""), ("ful", 0, ""), ("ous", 0, ""),
                           ("ive", 0, ""), ("ize", 0, ""), ("ise", 0, ""), ("ation", 0, "ate")):
        if w.endswith(suf):
            return w[: len(w) - len(suf)] + rep
    if w.endswith("ing") and len(w) > 6: return w[:-3]
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("ed") and len(w) > 5: return w[:-2]
    if w.endswith("ly") and len(w) > 5: return w[:-2]
    if w.endswith("er") and len(w) > 5: return w[:-2]
    if w.endswith("es") and len(w) > 4: return w[:-2]
    if w.endswith("s") and len(w) > 4: return w[:-1]
    return w


class Doc:
    """A preprocessed tweet."""
    __slots__ = ("raw", "tokens", "lemmas")

    def __init__(self, raw: str):
        self.raw = raw
        self.tokens = clean(raw).split()
        self.lemmas = [LEMMA.get(w, _stem(w)) for w in self.tokens if w not in STOPWORDS and len(w) > 2]


def lexicon_score(lemmas):
    """L1-normalised per-emotion lexicon score."""
    s = [0.0] * 6
    for w in lemmas:
        e = LEXICON.get(w)
        if e:
            for i in range(6):
                s[i] += e[i]
    tot = sum(s)
    return [x / tot for x in s] if tot > 0 else s


def lexicon_counts(lemmas):
    c = [0.0] * 6
    for w in lemmas:
        e = LEXICON.get(w)
        if e:
            for i in range(6):
                if e[i] > 0.5:
                    c[i] += 1
    return c
