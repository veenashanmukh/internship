"""Statistics for the confirmatory and exploratory analyses (methodology R4, 19.3, 19.4).

Paired bootstrap: every metric is computed as a *weighted* statistic, where a bootstrap
resample of test instances is a vector of multinomial counts. The same count vectors are
used for every seed and both systems of a comparison (R4).

Combined test for systems A, B over shared seeds k = 1..K:
    D_k = metric_A,k - metric_B,k  (full test set)          D = mean_k D_k
    V_inst = Var_b( mean_k D_k^(b) )                         (bootstrap, same resamples)
    V_seed = max(s_D^2, 2 sigma_pool^2) / K                  (19.3 pooled floor)
    df_seed = K - 1 if s_D^2 is used, else the pooled df
    nu = (V_seed + V_inst)^2 / (V_seed^2 / df_seed)          (Welch-Satterthwaite, V_inst exact)
    T = D / sqrt(V_seed + V_inst), two-sided p from t_nu
Configurations whose seed SD exceeds 3 x sigma_pool are flagged and use s_D^2 alone (df = K - 1).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats as st

B_DEFAULT = 2000


# ----------------------------------------------------------------------------- resampling
def boot_counts(n: int, B: int = B_DEFAULT, seed: int = 0, chunk: int = 250):
    """Yield (b0, counts[chunk, n]) multinomial resample counts; deterministic per (n, seed)."""
    rng = np.random.RandomState(seed)
    for b0 in range(0, B, chunk):
        m = min(chunk, B - b0)
        idx = rng.randint(0, n, size=(m, n))
        c = np.zeros((m, n), dtype=np.float32)
        for r in range(m):
            c[r] = np.bincount(idx[r], minlength=n)
        yield b0, c


# ----------------------------------------------------------------------------- weighted metrics
def w_average_precision(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted average precision, one value per row of W (rows = weight vectors).
    Equals sklearn.metrics.average_precision_score for unit weights, ties included."""
    y = np.asarray(y).astype(np.float32); s = np.asarray(s, dtype=np.float64)
    W = np.atleast_2d(W).astype(np.float64)
    o = np.argsort(-s, kind="mergesort")
    ys, ss, Ws = y[o], s[o], W[:, o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]          # end of each tie group
    tp = np.cumsum(Ws * ys, 1)[:, last]
    al = np.cumsum(Ws, 1)[:, last]
    dtp = np.diff(np.concatenate([np.zeros((W.shape[0], 1)), tp], 1), axis=1)
    prec = np.divide(tp, al, out=np.zeros_like(tp), where=al > 0)
    tot = tp[:, -1]
    return np.divide((dtp * prec).sum(1), tot, out=np.full(W.shape[0], np.nan), where=tot > 0)


def w_roc_auc(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC (ties count 1/2), one value per row of W."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    W = np.atleast_2d(W).astype(np.float64)
    o = np.argsort(s, kind="mergesort")
    ys, ss, Ws = y[o], s[o], W[:, o]
    grp = np.r_[0, np.cumsum(np.diff(ss) != 0)]                      # tie-group id, ascending
    G = grp[-1] + 1
    out = np.empty(W.shape[0])
    for r in range(W.shape[0]):
        wn = np.bincount(grp, weights=Ws[r] * ~ys, minlength=G)
        wp = np.bincount(grp, weights=Ws[r] * ys, minlength=G)
        below = np.cumsum(wn) - wn
        num = (wp * (below + 0.5 * wn)).sum()
        den = wp.sum() * wn.sum()
        out[r] = num / den if den > 0 else np.nan
    return out


def w_macro_f1(y: np.ndarray, pred: np.ndarray, W: np.ndarray, n_cls: int) -> np.ndarray:
    W = np.atleast_2d(W).astype(np.float64)
    f1 = []
    for c in range(n_cls):
        tp = W @ ((pred == c) & (y == c)).astype(float)
        fp = W @ ((pred == c) & (y != c)).astype(float)
        fn = W @ ((pred != c) & (y == c)).astype(float)
        d = 2 * tp + fp + fn
        f1.append(np.divide(2 * tp, d, out=np.zeros_like(d), where=d > 0))
    return np.mean(f1, 0)


def w_mean(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted mean of a per-instance quantity, NaN entries ignored."""
    x = np.asarray(x, dtype=np.float64); ok = ~np.isnan(x)
    W = np.atleast_2d(W)[:, ok].astype(np.float64)
    return (W @ x[ok]) / W.sum(1)


def generalised_mean(x, p: float = -5.0):
    x = np.asarray(x, dtype=np.float64)
    return np.power(np.nanmean(np.power(x, p), axis=-1), 1.0 / p)


# ----------------------------------------------------------------------------- combined test
def pooled_seed_variance(per_config: Dict[str, List[float]]) -> tuple:
    """sigma_pool^2 and its df from per-configuration seed values of one metric (19.3)."""
    num, df = 0.0, 0
    for vals in per_config.values():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if len(v) >= 2:
            num += (len(v) - 1) * v.var(ddof=1); df += len(v) - 1
    return (num / df if df else np.nan), df


def combined_test(full_A: np.ndarray, full_B: np.ndarray, boot_A: np.ndarray, boot_B: np.ndarray,
                  sigma2_pool: float, df_pool: int, flagged: bool = False) -> dict:
    """full_*: metric on the full test set per shared seed [K]; boot_*: per resample [K, B].
    Returns the R4 report for A - B."""
    full_A, full_B = np.asarray(full_A, float), np.asarray(full_B, float)
    K = len(full_A)
    Dk = full_A - full_B
    D = float(Dk.mean())
    Db = np.nanmean(np.asarray(boot_A) - np.asarray(boot_B), 0)
    V_inst = float(np.nanvar(Db, ddof=1))
    sD2 = float(Dk.var(ddof=1)) if K > 1 else np.nan
    floor = 2 * sigma2_pool if np.isfinite(sigma2_pool) else 0.0
    if flagged or not np.isfinite(floor) or (np.isfinite(sD2) and sD2 >= floor):
        V_seed, df_seed, used = sD2 / K, K - 1, "s_D^2"
    else:
        V_seed, df_seed, used = floor / K, df_pool, "pooled floor"
    V = V_seed + V_inst
    if not (np.isfinite(D) and np.isfinite(V)):                  # not computable: never "significant"
        return {"diff": D, "seed_sd": np.nan, "K": K, "per_seed_diff": Dk.tolist(), "sign_consistent": False,
                "n_seeds_A_better": int((Dk > 0).sum()), "V_seed": V_seed, "V_inst": V_inst, "seed_term": used,
                "df": np.nan, "se": np.nan, "p": np.nan, "ci_combined": [np.nan, np.nan],
                "ci_instance_only": [np.nan, np.nan], "corr_A_B_over_seeds": np.nan, "not_computable": True}
    nu = (V ** 2) / (V_seed ** 2 / df_seed) if V_seed > 0 else np.inf
    se = np.sqrt(V)
    t = D / se if se > 0 else np.inf
    p = float(2 * st.t.sf(abs(t), nu)) if np.isfinite(nu) else float(2 * st.norm.sf(abs(t)))
    q = st.t.ppf(0.975, nu) if np.isfinite(nu) else st.norm.ppf(0.975)
    lo_i, hi_i = np.nanpercentile(Db, [2.5, 97.5])
    return {"diff": D, "seed_sd": float(np.sqrt(sD2)) if np.isfinite(sD2) else np.nan, "K": K,
            "per_seed_diff": Dk.tolist(), "sign_consistent": bool(np.all(Dk > 0) or np.all(Dk < 0)),
            "n_seeds_A_better": int((Dk > 0).sum()), "V_seed": V_seed, "V_inst": V_inst, "seed_term": used,
            "df": float(nu), "se": float(se), "p": p,
            "ci_combined": [D - q * se, D + q * se], "ci_instance_only": [float(lo_i), float(hi_i)],
            "corr_A_B_over_seeds": float(np.corrcoef(full_A, full_B)[0, 1]) if K > 2 else np.nan}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjustment. A test that could not be computed (NaN) counts as p = 1:
    it can never be significant, and it still counts towards the family size."""
    pvals = {k: (1.0 if v is None or not np.isfinite(v) else float(v)) for k, v in pvals.items()}
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def lower_bound_95(full: np.ndarray, boot: np.ndarray, sigma2_pool: float, df_pool: int) -> float:
    """LB of the combined 95% CI of a single system's effect (e.g. IG minus random) over seeds,
    used to estimate the active control's advantage m on validation (19.4)."""
    r = combined_test(full, np.zeros_like(full), boot, np.zeros_like(boot), sigma2_pool, df_pool)
    return r["ci_combined"][0]
