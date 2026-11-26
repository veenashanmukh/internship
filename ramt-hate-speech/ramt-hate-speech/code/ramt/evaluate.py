"""Prediction, threshold calibration, temperature scaling and metrics
(Sections 6A, 8, 9; amendments A13/A14 are applied in ramt/stats.py)."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from ramt.data.datasets import LABELS, HX_CLASSES, collate
from ramt.data.rationales import subword_to_word

AOPC_BINS = (0.01, 0.05, 0.10, 0.20, 0.50)   # ERASER reference default (verified in its metrics.py)
THRESH_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ----------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, items: List[dict], heads: List[str], device: str, pad_id: int, batch_size: int = 128,
            fp16: bool = True, want_attention: bool = False) -> Dict[str, dict]:
    """One encoder pass per batch; returns logits (and word-level attention) for each head."""
    model.eval()
    order = np.argsort([len(it["ids"]) for it in items], kind="stable")
    out = {t: {"logits": [None] * len(items), "attn_words": [None] * len(items)} for t in heads}
    for s in range(0, len(items), batch_size):
        idx = order[s:s + batch_size]
        b = collate([items[i] for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            res = model.forward_all(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["content_mask"].to(device))
        for t in heads:
            lg, alpha = res[t]
            lg = lg.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[t]["logits"][i] = lg[j]
                if want_attention and alpha is not None:
                    n = len(items[i]["ids"])
                    out[t]["attn_words"][i] = subword_to_word(alpha[j, :n].cpu().numpy(), items[i]["wid"],
                                                              len(items[i]["words"]))
    for t in heads:
        out[t]["logits"] = np.stack(out[t]["logits"])
    return out


def to_probs(task: str, logits: np.ndarray, temperature: Optional[np.ndarray] = None) -> np.ndarray:
    z = logits if temperature is None else logits / temperature
    if task == "hx":
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-z))


def toxicity_score(task: str, probs: np.ndarray) -> np.ndarray:
    """Binary 'harmful' score per head for cross-dataset scoring (Section 6A)."""
    if task == "explicit":
        return probs[:, 0]                      # p(toxic)
    if task == "implicit":
        return probs[:, 0]                      # p(implicit)
    return 1.0 - probs[:, 2]                    # 1 - p(normal)


# ----------------------------------------------------------------------------- thresholds / temperature
def f1_curve(y: np.ndarray, p: np.ndarray, grid=THRESH_GRID) -> np.ndarray:
    """F1 at every threshold, vectorised."""
    y = y.astype(bool)
    P = p[None, :] >= grid[:, None]
    tp = (P & y[None, :]).sum(1)
    fp = (P & ~y[None, :]).sum(1)
    fn = (~P & y[None, :]).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def bootstrap_threshold(y: np.ndarray, p: np.ndarray, n_boot: int = 200, seed: int = 0) -> float:
    """Section 8 step 2: argmax of the bootstrap-averaged F1 curve."""
    if y.sum() == 0:
        return 0.5
    rng = np.random.RandomState(seed)
    acc = np.zeros(len(THRESH_GRID))
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        acc += f1_curve(y[i], p[i])
    return float(THRESH_GRID[int(np.argmax(acc / n_boot))])


def fit_temperature(logits: np.ndarray, y: np.ndarray, multiclass: bool) -> float:
    """Single temperature minimising validation NLL (grid + refinement); calibration only."""
    def nll(T):
        if multiclass:
            z = logits / T
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(y)), y.astype(int)].mean()
        p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    best = grid[int(np.argmin([nll(T) for T in grid]))]
    fine = np.linspace(best * 0.9, best * 1.1, 41)
    return float(fine[int(np.argmin([nll(T) for T in fine]))])


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


# ----------------------------------------------------------------------------- metrics
def _safe(fn, y, p):
    try:
        return float(fn(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        return float("nan")


def binary_report(y, p, t):
    pred = (p >= t).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "acc": float(accuracy_score(y, pred)),
            "threshold": float(t)}


def explicit_metrics(Y, P, thr: Dict[str, float]):
    Y = Y.astype(int)
    out = {"per_label": {}}
    for k, l in enumerate(LABELS):
        out["per_label"][l] = {"at_0.5": binary_report(Y[:, k], P[:, k], 0.5),
                               "at_tuned": binary_report(Y[:, k], P[:, k], thr[l]),
                               "auprc": _safe(average_precision_score, Y[:, k], P[:, k]),
                               "roc_auc": _safe(roc_auc_score, Y[:, k], P[:, k]),
                               "ece": ece(Y[:, k], P[:, k]), "support": int(Y[:, k].sum())}
    for tag, T in (("0.5", np.full(6, 0.5)), ("tuned", np.array([thr[l] for l in LABELS]))):
        pred = (P >= T).astype(int)
        out[f"macro_f1@{tag}"] = float(f1_score(Y, pred, average="macro", zero_division=0))
        out[f"micro_f1@{tag}"] = float(f1_score(Y, pred, average="micro", zero_division=0))
    out["macro_auprc"] = float(np.nanmean([out["per_label"][l]["auprc"] for l in LABELS]))
    out["macro_roc_auc"] = float(np.nanmean([out["per_label"][l]["roc_auc"] for l in LABELS]))
    return out


def implicit_metrics(y, p, t):
    y = y.astype(int).ravel(); p = p.ravel()
    return {"at_0.5": binary_report(y, p, 0.5), "at_tuned": binary_report(y, p, t),
            "auprc": _safe(average_precision_score, y, p), "roc_auc": _safe(roc_auc_score, y, p), "ece": ece(y, p)}


def hx_metrics(y, P, t_bin):
    pred = P.argmax(1)
    yb, pb = (y != 2).astype(int), 1 - P[:, 2]
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "acc": float(accuracy_score(y, pred)),
            "per_class_f1": dict(zip(HX_CLASSES, map(float, f1_score(y, pred, average=None, labels=[0, 1, 2])))),
            "binary_toxic": {"at_tuned": binary_report(yb, pb, t_bin), "roc_auc": _safe(roc_auc_score, yb, pb),
                             "auprc": _safe(average_precision_score, yb, pb)}}


def val_score(task: str, Y: np.ndarray, P: np.ndarray) -> float:
    """Threshold-free per-task validation score used for checkpoint selection (A2, 8.3)."""
    if task == "explicit":
        return float(np.nanmean([_safe(average_precision_score, Y[:, k].astype(int), P[:, k]) for k in range(6)]))
    if task == "implicit":
        return _safe(average_precision_score, Y[:, 0].astype(int), P[:, 0])
    return float(f1_score(Y.astype(int), P.argmax(1), average="macro"))


# ----------------------------------------------------------------------------- explanations
def plausibility(word_scores: List[np.ndarray], gold: List[np.ndarray]) -> dict:
    """Token AUPRC (primary) and token-F1@k, k = number of gold words (Section 9)."""
    aps, f1s = [], []
    for s, g in zip(word_scores, gold):
        if s is None:
            continue
        g = np.asarray(g[:len(s)], dtype=int); s = np.asarray(s, dtype=float)[:len(g)]
        if g.sum() == 0 or len(g) == 0:
            continue
        k = int(g.sum())
        top = np.zeros_like(g); top[np.argsort(-s, kind="stable")[:k]] = 1
        tp = int((top & g).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (top.sum() + g.sum()))
        if g.sum() < len(g):
            aps.append(average_precision_score(g, s))
    return {"token_auprc": float(np.mean(aps)) if aps else float("nan"),
            "token_f1_at_k": float(np.mean(f1s)) if f1s else float("nan"), "n_posts": len(f1s)}


def _k(frac, n):
    return max(1, int(math.ceil(frac * n)))   # IMPL: at least one word per bin


def aopc_faithfulness(score_fn: Callable[[List[List[str]]], np.ndarray], words: List[List[str]],
                      word_scores: List[np.ndarray], bins=AOPC_BINS, random_draws: int = 5, seed: int = 0) -> dict:
    """Comprehensiveness / sufficiency averaged over bins (AOPC) w.r.t. the explained
    score s(x) returned by score_fn, plus the random-explanation baseline (Section 9).
    score_fn maps a list of word lists to the explained quantity (e.g. 1 - p(normal))."""
    rng = np.random.RandomState(seed)
    base = score_fn(words)
    comp, suff, rcomp, rsuff = [], [], [], []
    for frac in bins:
        rm, keep = [], []
        for w, s in zip(words, word_scores):
            top = set(np.argsort(-np.asarray(s), kind="stable")[:_k(frac, len(w))].tolist())
            rm.append([x for i, x in enumerate(w) if i not in top])
            keep.append([x for i, x in enumerate(w) if i in top])
        comp.append(base - score_fn(rm)); suff.append(base - score_fn(keep))
        rc, rs = [], []
        for _ in range(random_draws):
            rm, keep = [], []
            for w in words:
                top = set(rng.permutation(len(w))[:_k(frac, len(w))].tolist())
                rm.append([x for i, x in enumerate(w) if i not in top])
                keep.append([x for i, x in enumerate(w) if i in top])
            rc.append(base - score_fn(rm)); rs.append(base - score_fn(keep))
        rcomp.append(np.mean(rc, 0)); rsuff.append(np.mean(rs, 0))
    comp, suff, rcomp, rsuff = map(lambda a: np.mean(np.stack(a), 0), (comp, suff, rcomp, rsuff))
    return {"aopc_comprehensiveness": float(comp.mean()), "aopc_sufficiency": float(suff.mean()),
            "random_aopc_comprehensiveness": float(rcomp.mean()), "random_aopc_sufficiency": float(rsuff.mean()),
            "comp_minus_random": float((comp - rcomp).mean()), "suff_minus_random": float((suff - rsuff).mean()),
            "per_post": {"comp": comp, "suff": suff, "rand_comp": rcomp, "rand_suff": rsuff}, "bins": list(bins)}


# ----------------------------------------------------------------------------- fixed-set faithfulness (analysis)
def _perturb(words, scores, frac, keep_top):
    out = []
    for w, s in zip(words, scores):
        top = set(np.argsort(-np.asarray(s), kind="stable")[:_k(frac, len(w))].tolist())
        out.append([x for i, x in enumerate(w) if (i in top) == keep_top])
    return out


def aopc_method(score_fn, words, word_scores, base=None, bins=AOPC_BINS):
    """Per-post AOPC comprehensiveness and sufficiency of one explanation method
    (same definitions as aopc_faithfulness, without the random baseline)."""
    base = score_fn(words) if base is None else base
    comp = np.mean([base - score_fn(_perturb(words, word_scores, f, False)) for f in bins], 0)
    suff = np.mean([base - score_fn(_perturb(words, word_scores, f, True)) for f in bins], 0)
    return comp, suff, base


def aopc_random(score_fn, words, base=None, bins=AOPC_BINS, random_draws=5, seed=0):
    """Per-post random-explanation AOPC (shared by every method on the same model and posts)."""
    rng = np.random.RandomState(seed)
    base = score_fn(words) if base is None else base
    rc, rs = [], []
    for f in bins:
        c, s = [], []
        for _ in range(random_draws):
            sc = [rng.rand(len(w)) for w in words]
            c.append(base - score_fn(_perturb(words, sc, f, False)))
            s.append(base - score_fn(_perturb(words, sc, f, True)))
        rc.append(np.mean(c, 0)); rs.append(np.mean(s, 0))
    return np.mean(rc, 0), np.mean(rs, 0)


def per_post_ap(word_scores, gold):
    """Per-post token AUPRC as in plausibility(); NaN where undefined."""
    out = np.full(len(word_scores), np.nan)
    for i, (s, g) in enumerate(zip(word_scores, gold)):
        if s is None:
            continue
        g = np.asarray(g[:len(s)], dtype=int); s = np.asarray(s, dtype=float)[:len(g)]
        if 0 < g.sum() < len(g):
            out[i] = average_precision_score(g, s)
    return out
