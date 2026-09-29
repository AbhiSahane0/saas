"""Five classifiers behind one interface: fit(X, y, docs) / predict(X, docs) -> label indices."""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler, normalize
from .nlp import lexicon_score, lexicon_counts

N_CLASSES = 6


class NaiveBayes:
    name = "Naive Bayes"
    def fit(self, X, y, docs): self.m = MultinomialNB(alpha=1.0).fit(np.clip(X, 0, None), y); return self
    def predict(self, X, docs): return self.m.predict(np.clip(X, 0, None))


class SVM:
    name = "SVM (Linear)"
    def fit(self, X, y, docs): self.m = LinearSVC(C=1.0, max_iter=5000).fit(normalize(X), y); return self
    def predict(self, X, docs): return self.m.predict(normalize(X))


class ANN:
    name = "ANN (128-64-6)"
    def fit(self, X, y, docs):
        self.sc = MaxAbsScaler().fit(X)
        self.m = MLPClassifier((128, 64), alpha=1e-3, max_iter=200, early_stopping=True,
                               n_iter_no_change=10, random_state=7).fit(self.sc.transform(X), y)
        return self
    def predict(self, X, docs): return self.m.predict(self.sc.transform(X))


class HySimNet:
    """Weighted-cosine similarity to class prototypes, with learned per-class attention
    and a lexicon boost at prediction time (same algorithm as HySimNet.java)."""
    name = "HySimNet"
    def __init__(self, epochs=15, lr=0.05): self.epochs, self.lr0 = epochs, lr

    @staticmethod
    def _wcos(x, p, w):
        return (w * x * p).sum() / (np.sqrt((w * x * x).sum()) * np.sqrt((w * p * p).sum()) + 1e-9)

    def _raw(self, x):
        return int(np.argmax([self._wcos(x, self.proto[c], self.attn[c]) for c in range(N_CLASSES)]))

    def fit(self, X, y, docs):
        X = normalize(X).astype(np.float64)
        d = X.shape[1]
        self.proto = np.zeros((N_CLASSES, d)); self.attn = np.full((N_CLASSES, d), 1.0 / d)
        for c in range(N_CLASSES):
            if (y == c).any(): self.proto[c] = X[y == c].mean(axis=0)
        rng, lr = np.random.RandomState(13), self.lr0
        for _ in range(self.epochs):
            for i in rng.permutation(len(X)):
                x, a = X[i], y[i]
                p = self._raw(x)
                if p != a:
                    self.attn[a] += lr * x * self.proto[a]
                    self.attn[p] -= lr * x * self.proto[p] * 0.5
                    self.attn = np.maximum(self.attn, 1e-9)
                    self.attn /= self.attn.sum(axis=1, keepdims=True)
            lr *= 0.92
        return self

    def predict(self, X, docs):
        X = normalize(X).astype(np.float64)
        out = []
        for x, d in zip(X, docs):
            s = np.array([self._wcos(x, self.proto[c], self.attn[c]) for c in range(N_CLASSES)])
            s += np.array(lexicon_score(d.lemmas)) * 3.0 + np.array(lexicon_counts(d.lemmas)) * 0.4
            out.append(int(np.argmax(s)))
        return np.array(out)


def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTM:
    """Feature vector is split into 10 time-steps and fed to a 64-unit LSTM (as in RNN_LSTM_N.java).
    Pure NumPy (forward pass + backprop-through-time + Adam) so no deep-learning framework is needed,
    which keeps the app small enough for free hosting."""
    name = "RNN-LSTM"
    SEQ, HID = 10, 64

    def __init__(self, epochs=25, lr=3e-3, batch=32): self.epochs, self.lr, self.batch = epochs, lr, batch

    def _seq(self, X):
        X = self.sc.transform(X)
        X = np.pad(X, ((0, 0), (0, (-X.shape[1]) % self.SEQ)))
        return X.reshape(len(X), self.SEQ, -1).astype(np.float32)

    def _forward(self, xs):
        n, T, _ = xs.shape
        H = self.HID
        h = np.zeros((n, H), np.float32); c = np.zeros((n, H), np.float32)
        cache = []
        for t in range(T):
            z = xs[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
            i, f, o, g = _sig(z[:, :H]), _sig(z[:, H:2 * H]), _sig(z[:, 2 * H:3 * H]), np.tanh(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            cache.append((xs[:, t], h, c, i, f, o, g, tc))
            h, c = o * tc, c_new
        return h, cache

    def fit(self, X, y, docs):
        rng = np.random.RandomState(17)
        self.sc = MaxAbsScaler().fit(X)
        xs = self._seq(X)
        F, H = xs.shape[2], self.HID
        self.p = {"Wx": (rng.randn(F, 4 * H) * np.sqrt(1.0 / F)).astype(np.float32),
                  "Wh": (rng.randn(H, 4 * H) * np.sqrt(1.0 / H)).astype(np.float32),
                  "b": np.zeros(4 * H, np.float32),
                  "Wo": (rng.randn(H, N_CLASSES) * np.sqrt(1.0 / H)).astype(np.float32),
                  "bo": np.zeros(N_CLASSES, np.float32)}
        self.p["b"][H:2 * H] = 1.0                       # forget-gate bias
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(xs))
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                xb, yb = xs[idx], y[idx]
                h, cache = self._forward(xb)
                logits = h @ self.p["Wo"] + self.p["bo"]
                pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
                d = pr; d[np.arange(len(yb)), yb] -= 1; d /= len(yb)
                g = {k: np.zeros_like(v) for k, v in self.p.items()}
                g["Wo"], g["bo"] = h.T @ d, d.sum(0)
                dh, dc = d @ self.p["Wo"].T, np.zeros_like(h)
                for x_t, h_prev, c_prev, i, f, o, gg, tc in reversed(cache):
                    dc = dc + dh * o * (1 - tc ** 2)
                    dz = np.concatenate([dc * gg * i * (1 - i), dc * c_prev * f * (1 - f),
                                         dh * tc * o * (1 - o), dc * i * (1 - gg ** 2)], axis=1)
                    g["Wx"] += x_t.T @ dz; g["Wh"] += h_prev.T @ dz; g["b"] += dz.sum(0)
                    dh, dc = dz @ self.p["Wh"].T, dc * f
                step += 1
                for k in self.p:                          # Adam with gradient clipping
                    gk = np.clip(g[k], -5, 5)
                    m[k] = 0.9 * m[k] + 0.1 * gk; v2[k] = 0.999 * v2[k] + 0.001 * gk * gk
                    self.p[k] -= self.lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
        return self

    def predict(self, X, docs):
        h, _ = self._forward(self._seq(X))
        return (h @ self.p["Wo"] + self.p["bo"]).argmax(1)


CLASSIFIERS = [NaiveBayes, SVM, ANN, HySimNet, LSTM]
