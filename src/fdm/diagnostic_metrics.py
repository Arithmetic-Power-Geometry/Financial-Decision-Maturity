from __future__ import annotations
import numpy as np


def roc_auc(y, score):
    y = np.asarray(y, dtype=int); s = np.asarray(score, dtype=float)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    total = 0.0
    for i in pos:
        total += np.sum(s[i] > s[neg]) + 0.5*np.sum(s[i] == s[neg])
    return float(total / (len(pos)*len(neg)))


def average_precision(y, score):
    y = np.asarray(y, dtype=int); s = np.asarray(score, dtype=float)
    if y.sum() == 0: return float('nan')
    order = np.argsort(-s, kind='mergesort'); ys = y[order]
    tp = np.cumsum(ys); precision = tp / np.arange(1, len(ys)+1)
    return float(np.sum(precision * ys) / np.sum(ys))


def brier(y, prob):
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    return float(np.mean((p-y)**2))


def ece(y, prob, bins=10):
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0,1,bins+1); out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (p < hi if i < bins-1 else p <= hi)
        if np.any(mask): out += np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(out)


def top_fraction_precision(y, score, fraction=0.10):
    y = np.asarray(y, dtype=int); s = np.asarray(score, dtype=float)
    k = max(1, int(round(len(y)*fraction))); idx = np.argsort(-s)[:k]
    return float(np.mean(y[idx]))


def empirical_calibrate(train_score, train_y, test_score, bins=20):
    train_score = np.asarray(train_score, dtype=float); train_y = np.asarray(train_y, dtype=float); test_score = np.asarray(test_score, dtype=float)
    edges = np.unique(np.quantile(train_score, np.linspace(0,1,bins+1)))
    if len(edges) < 2: return np.repeat(np.mean(train_y), len(test_score))
    global_rate = float(np.mean(train_y)); vals=[]; rates=[]
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (train_score >= lo) & (train_score < hi if i < len(edges)-2 else train_score <= hi)
        vals.append((lo,hi)); rates.append(float(np.mean(train_y[mask])) if np.any(mask) else global_rate)
    out = np.empty(len(test_score), dtype=float)
    for j,x in enumerate(test_score):
        placed=False
        for i,(lo,hi) in enumerate(vals):
            if x >= lo and (x < hi or i == len(vals)-1): out[j]=rates[i]; placed=True; break
        if not placed: out[j] = rates[0] if x < vals[0][0] else rates[-1]
    return out
