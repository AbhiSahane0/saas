"""Feature extraction for the six feature modes (port of FeatureExtractor.java).
Everything is fitted on the training split only."""
import math
from collections import Counter, defaultdict
import numpy as np
from .nlp import LEXICON, lexicon_score, lexicon_counts

MODES = ["tfidf", "coco", "bigram", "ngram", "lemma+tfidf", "lemma+coco"]
TFIDF_VOCAB, BIGRAM_VOCAB, NGRAM_VOCAB, COCO_VOCAB, COCO_WINDOW = 1500, 600, 300, 500, 2


def _top(counter, n):
    return [w for w, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


class FeatureBuilder:
    def fit(self, docs):
        n = len(docs)
        df, cf = Counter(), Counter()
        for d in docs:
            df.update(set(d.lemmas))
            cf.update(d.lemmas)
        self.idf = {w: math.log((n + 1.0) / (c + 1)) + 1 for w, c in df.items()}
        self.tfidf_vocab = _top(cf, TFIDF_VOCAB)
        self.coco_vocab = self.tfidf_vocab[:COCO_VOCAB]
        bi, tri = Counter(), Counter()
        self.co = defaultdict(Counter)
        for d in docs:
            l = d.lemmas
            bi.update(f"{a}_{b}" for a, b in zip(l, l[1:]))
            tri.update(f"{a}_{b}_{c}" for a, b, c in zip(l, l[1:], l[2:]))
            for i, w in enumerate(l):
                for j in range(max(0, i - COCO_WINDOW), min(len(l) - 1, i + COCO_WINDOW) + 1):
                    if i != j:
                        self.co[w][l[j]] += 1
        self.bigram_vocab = _top(bi, BIGRAM_VOCAB)
        self.ngram_vocab = _top(tri, NGRAM_VOCAB)
        return self

    # --- blocks --------------------------------------------------------
    def _tfidf(self, d):
        tf = Counter(d.lemmas)
        n = max(1, len(d.lemmas))
        return [tf.get(w, 0) / n * self.idf.get(w, 1.0) for w in self.tfidf_vocab]

    def _binary(self, d):
        s = set(d.lemmas)
        return [1.0 if w in s else 0.0 for w in self.tfidf_vocab]

    def _coco(self, d):
        v = np.array([sum(self.co.get(w, {}).get(t, 0) for t in d.lemmas) for w in self.coco_vocab], float)
        return (v / (np.sqrt((v ** 2).sum()) + 1e-9)).tolist()

    @staticmethod
    def _grams(d, vocab, k):
        l = d.lemmas
        c = Counter("_".join(l[i:i + k]) for i in range(len(l) - k + 1))
        return [float(c.get(g, 0)) for g in vocab]

    @staticmethod
    def _nlp(d):
        raw = d.raw.lower()
        tok = d.tokens
        # scaled to roughly [0, 1] so they don't swamp the text features after normalisation
        return [min(len(tok), 40) / 40, min(len(d.lemmas), 20) / 20,
                (float(np.mean([len(t) for t in tok])) if tok else 0.0) / 10,
                len(set(tok)) / len(tok) if tok else 0.0,
                min(raw.count("!"), 3) / 3, min(raw.count("?"), 3) / 3,
                1.0 if ("not" in raw or "never" in raw or "no" in raw) else 0.0,
                min(sum(1 for w in d.lemmas if w in LEXICON), 5) / 5]

    @staticmethod
    def _lex(d):
        return lexicon_score(d.lemmas) + lexicon_counts(d.lemmas)

    def vector(self, d, mode):
        if mode == "tfidf":        v = self._tfidf(d) + self._nlp(d)
        elif mode == "coco":       v = self._coco(d) + self._nlp(d) + self._lex(d)
        elif mode == "bigram":     v = self._grams(d, self.bigram_vocab, 2) + self._nlp(d) + self._lex(d)
        elif mode == "ngram":      v = self._grams(d, self.ngram_vocab, 3) + self._nlp(d) + self._lex(d)
        elif mode == "lemma+tfidf": v = self._tfidf(d) + self._binary(d) + self._lex(d) + self._nlp(d)
        elif mode == "lemma+coco": v = self._binary(d) + self._coco(d) + self._lex(d) + self._nlp(d)
        else: raise ValueError(mode)
        return v

    def matrix(self, docs, mode):
        return np.array([self.vector(d, mode) for d in docs], dtype=np.float32)
