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


class LSTM:
    """Feature vector is split into 10 time-steps and fed to a 64-unit LSTM (as in RNN_LSTM_N.java)."""
    name = "RNN-LSTM"
    SEQ, HID = 10, 64

    def __init__(self, epochs=25): self.epochs = epochs

    def _seq(self, X):
        import torch
        X = self.sc.transform(X)
        pad = (-X.shape[1]) % self.SEQ
        X = np.pad(X, ((0, 0), (0, pad)))
        return torch.tensor(X.reshape(len(X), self.SEQ, -1), dtype=torch.float32)

    def fit(self, X, y, docs):
        import torch, torch.nn as nn
        torch.manual_seed(17); torch.set_num_threads(2)
        self.sc = MaxAbsScaler().fit(X)
        Xt, yt = self._seq(X), torch.tensor(y, dtype=torch.long)
        self.lstm = nn.LSTM(Xt.shape[2], self.HID, batch_first=True)
        self.out = nn.Linear(self.HID, N_CLASSES)
        opt = torch.optim.Adam(list(self.lstm.parameters()) + list(self.out.parameters()), lr=3e-3)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(Xt)).split(32):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(self._fwd(Xt[idx]), yt[idx])
                loss.backward(); opt.step()
        return self

    def _fwd(self, xs):
        _, (h, _) = self.lstm(xs)
        return self.out(h[-1])

    def predict(self, X, docs):
        import torch
        with torch.no_grad():
            return self._fwd(self._seq(X)).argmax(1).numpy()


CLASSIFIERS = [NaiveBayes, SVM, ANN, HySimNet, LSTM]
