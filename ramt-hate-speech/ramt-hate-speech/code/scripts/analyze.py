"""Phase 5 analysis: confirmatory family (R8 8.8), H3 decision rule (R5), hypothesis E (R7 +
19.4 margins), fairness, exploratory comparisons and descriptive tables. CPU only.

    python scripts/analyze.py --runs <trained runs> --splits <splits> --explain <explain out>
                              [--explain_cpu <cpu timing out>] --out /kaggle/working/analysis [--B 2000]

Every test statistic is computed from per-instance test outputs with the paired bootstrap and
the combined seed + instance variance with the pooled floor (ramt/stats.py). Validation data
are used only to derive non-inferiority margins (19.4); the test set never sets its own margin.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt import stats as S  # noqa: E402
from ramt.data.prep import LABELS  # noqa: E402
from ramt.data.rationales import aggregate  # noqa: E402
from ramt.evaluate import per_post_ap  # noqa: E402

MAIN_CONFIGS = ["M1_RAMT", "M2_MTL_noRat", "M3_random_matched", "M4_shared_pool", "M5_STL_explicit",
                "M5_STL_implicit", "M5_STL_hx", "M6_STL_hx_rationale", "M7_pair_explicit+implicit",
                "M7_pair_explicit+hx", "M7_pair_implicit+hx", "M8_MTL_noRat_hatebert", "M8_MTL_noRat_deberta"]
FRACS = (0.25, 0.5, 0.75)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Data:
    """Per-instance test outputs and labels, aligned by file order (asserted on post ids)."""

    def __init__(self, runs, splits):
        self.splits = splits
        self.dirs = {}
        for m in glob.glob(os.path.join(runs, "**", "metrics.json"), recursive=True):
            d = os.path.dirname(m); p = d.split(os.sep)
            if p[-3].startswith("pilot"):
                continue
            self.dirs[(p[-2], p[-1])] = d                       # (config, seed) -> dir
        self.metrics = {k: json.load(open(os.path.join(d, "metrics.json"))) for k, d in self.dirs.items()}
        self.cache = {}
        hx = pd.read_json(os.path.join(splits, "hx_test.jsonl"), lines=True)
        self.hx = hx
        self.hx_y = hx["label"].values.astype(int)
        ex = pd.read_csv(os.path.join(splits, "explicit_test.csv"))
        self.ex_Y = ex[LABELS].values.astype(int)
        im = pd.read_csv(os.path.join(splits, "implicit_test.csv"))
        self.im_y = im["label"].values.astype(int)
        hc = pd.read_csv(os.path.join(splits, "hatecheck.csv"))
        self.hc_y = hc["label"].values.astype(int) if "label" in hc else None
        self.W = int(json.load(open(os.path.join(splits, "data_stats.json")))["max_words_W"])
        self.gold = [aggregate(r, len(w), "majority")[:self.W] for w, r in zip(hx["words"], hx["rationales"])]

    def seeds(self, cfg):
        return sorted([s for c, s in self.dirs if c == cfg], key=lambda s: int(s[4:]))

    def preds(self, cfg, seed, testset, head):
        k = (cfg, seed, testset, head)
        if k not in self.cache:
            df = pd.read_csv(os.path.join(self.dirs[(cfg, seed)], f"test_{testset}__{head}.csv"))
            lg = df[[c for c in df.columns if c.startswith("logit_")]].values.astype(np.float64)
            pr = softmax(lg) if head == "hx" else sigmoid(lg)
            self.cache[k] = (df["post_id"].astype(str).values, pr)
        return self.cache[k]

    def attention(self, cfg, seed, head):
        k = (cfg, seed, "attn", head)
        if k not in self.cache:
            z = np.load(os.path.join(self.dirs[(cfg, seed)], "attention_hx_test.npz"), allow_pickle=True)
            self.cache[k] = per_post_ap(list(z[head]), self.gold)
        return self.cache[k]


# ----------------------------------------------------------------------------- metric definitions
def m_explicit_auprc(D, cfg, seed, W):
    _, P = D.preds(cfg, seed, "explicit", "explicit")
    assert len(P) == len(D.ex_Y), "explicit predictions and labels differ in length"
    return np.nanmean([S.w_average_precision(D.ex_Y[:, j], P[:, j], W) for j in range(len(LABELS))], 0)


def m_implicit_auprc(D, cfg, seed, W):
    _, P = D.preds(cfg, seed, "implicit", "implicit")
    return S.w_average_precision(D.im_y, P[:, 0], W)


def m_hx_f1(D, cfg, seed, W):
    _, P = D.preds(cfg, seed, "hx", "hx")
    return S.w_macro_f1(D.hx_y, P.argmax(1), W, 3)


def m_plaus(head):
    def f(D, cfg, seed, W):
        return S.w_mean(D.attention(cfg, seed, head), W)
    return f


def communities(D, top=10):
    c = Counter()
    per_post = []
    for tl in D.hx["targets"]:
        cnt = Counter(g for ann in (tl or []) for g in set(ann or []))
        s = {g for g, n in cnt.items() if n > 1 and g not in ("None", "Other")}
        per_post.append(s); c.update(s)
    names = [g for g, _ in c.most_common(top)]
    return names, [np.array([g in s for s in per_post]) for g in names]


def m_bias(kind):
    def f(D, cfg, seed, W):
        _, P = D.preds(cfg, seed, "hx", "hx")
        sc, tox = 1 - P[:, 2], D.hx_y != 2
        vals = []
        for sub in D.subgroups:
            if kind == "subgroup":
                mask = sub
            elif kind == "bpsn":
                mask = (sub & ~tox) | (~sub & tox)
            else:
                mask = (sub & tox) | (~sub & ~tox)
            vals.append(S.w_roc_auc(tox, sc, np.atleast_2d(W) * mask))
        V = np.stack(vals, -1)                                  # [B, communities]
        V = np.where(np.isfinite(V), V, np.nan)
        return S.generalised_mean(V, -5.0)
    return f


def m_hatecheck_fpr(head):
    def f(D, cfg, seed, W):
        if head == "hx":
            _, P = D.preds(cfg, seed, "hatecheck", "hx")
            thr = json.load(open(os.path.join(D.dirs[(cfg, seed)], "thresholds.json")))["hx_binary"]
            flag = (1 - P[:, 2]) >= thr
        neg = D.hc_y == 0
        W = np.atleast_2d(W)
        return (W @ (flag & neg).astype(float)) / (W @ neg.astype(float))
    return f


TEST_N = {}
BOOT_SEED = {"explicit": 11, "implicit": 12, "hx": 13, "hatecheck": 14}   # one resample set per test set


def boot_metric(D, fn, cfg, seeds, testset, B):
    """full [K] and boot [K, B] for one configuration on one test set."""
    n = TEST_N[testset]
    full = np.array([float(fn(D, cfg, s, np.ones(n))[0]) for s in seeds])
    boot = np.zeros((len(seeds), B))
    for b0, C in S.boot_counts(n, B, seed=BOOT_SEED[testset]):
        for i, s in enumerate(seeds):
            boot[i, b0:b0 + C.shape[0]] = fn(D, cfg, s, C)
    return full, boot


# ----------------------------------------------------------------------------- faithfulness from explain outputs
def faith_arrays(explain, cfg, seed, task, split):
    for g in ("main", "main_extra"):
        p = os.path.join(explain, "faith", g, cfg, seed, f"{task}_{split}.npz")
        if os.path.exists(p):
            return dict(np.load(p, allow_pickle=True))
    return None


def faith_boot(explain, cfg, seeds, task, split, quantity, B, seed_rs=21):
    """quantity(arrays) -> per-post vector; mean over the fixed set, bootstrapped over its posts
    (same resamples for every configuration and method)."""
    arrs = [faith_arrays(explain, cfg, s, task, split) for s in seeds]
    if any(a is None for a in arrs):
        return None, None
    ids = [tuple(a["post_id"]) for a in arrs]
    assert all(i == ids[0] for i in ids), "fixed post sets differ between seeds"
    X = np.stack([quantity(a) for a in arrs])                   # [K, n]
    n = X.shape[1]
    full = np.nanmean(X, 1)
    boot = np.zeros((len(seeds), B))
    for b0, C in S.boot_counts(n, B, seed=seed_rs + {"hx": 0, "explicit": 1, "implicit": 2}[task]):
        for i in range(len(seeds)):
            boot[i, b0:b0 + C.shape[0]] = S.w_mean(X[i], C)
    return full, boot


Q = {"comp_attn": lambda a: a["comp_attn"] - a["comp_rand"], "suff_attn": lambda a: a["suff_attn"] - a["suff_rand"],
     "comp_ig": lambda a: a["comp_ig"] - a["comp_rand"], "suff_ig": lambda a: a["suff_ig"] - a["suff_rand"],
     "C_ig_minus_rand": lambda a: a["comp_ig"] - a["comp_rand"], "S_rand_minus_ig": lambda a: a["suff_rand"] - a["suff_ig"],
     "C_attn_minus_rand": lambda a: a["comp_attn"] - a["comp_rand"], "S_rand_minus_attn": lambda a: a["suff_rand"] - a["suff_attn"]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--explain", default=None)
    ap.add_argument("--explain_cpu", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=S.B_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = a.B
    D = Data(a.runs, a.splits)
    D.community_names, D.subgroups = communities(D)
    TEST_N.update({"explicit": len(D.ex_Y), "implicit": len(D.im_y), "hx": len(D.hx_y),
                   "hatecheck": len(D.hc_y) if D.hc_y is not None else 0})
    print("configs:", {c: len(D.seeds(c)) for c in sorted({c for c, _ in D.dirs})})
    print("communities:", D.community_names)

    METRICS = {  # name -> (fn, test set, configs that have it)
        "jigsaw_macro_auprc": (m_explicit_auprc, "explicit", lambda c: "explicit" in tasks_of(D, c)),
        "ihc_auprc": (m_implicit_auprc, "implicit", lambda c: "implicit" in tasks_of(D, c)),
        "hx_macro_f1": (m_hx_f1, "hx", lambda c: "hx" in tasks_of(D, c)),
        "plaus_hx_head": (m_plaus("hx"), "hx", lambda c: "hx" in tasks_of(D, c) and pooled_attn(D, c)),
        "plaus_explicit_head": (m_plaus("explicit"), "hx", lambda c: "explicit" in tasks_of(D, c) and pooled_attn(D, c)),
        "plaus_implicit_head": (m_plaus("implicit"), "hx", lambda c: "implicit" in tasks_of(D, c) and pooled_attn(D, c)),
        "gmb_bpsn_auc": (m_bias("bpsn"), "hx", lambda c: "hx" in tasks_of(D, c)),
        "gmb_bnsp_auc": (m_bias("bnsp"), "hx", lambda c: "hx" in tasks_of(D, c)),
        "gmb_subgroup_auc": (m_bias("subgroup"), "hx", lambda c: "hx" in tasks_of(D, c)),
        "hatecheck_nonhateful_fpr_hx_head": (m_hatecheck_fpr("hx"), "hatecheck", lambda c: "hx" in tasks_of(D, c)),
    }
    # ---- full-test metrics for every run (descriptive + pooled variance)
    configs = sorted({c for c, _ in D.dirs})
    full = defaultdict(dict)
    for name, (fn, ts, has) in METRICS.items():
        if ts == "hatecheck" and D.hc_y is None:
            continue
        for c in configs:
            if has(c):
                for s in D.seeds(c):
                    full[name][(c, s)] = float(fn(D, c, s, np.ones(TEST_N[ts]))[0])
    pooled = {}
    for name in full:
        per = defaultdict(list)
        for (c, s), v in full[name].items():
            if c in MAIN_CONFIGS:
                per[c].append(v)
        pooled[name] = S.pooled_seed_variance(per) + ({c: float(np.std(v, ddof=1)) for c, v in per.items() if len(v) > 1},)
    sanity = {"M1_seed42_jigsaw_macro_auprc_recomputed": full["jigsaw_macro_auprc"].get(("M1_RAMT", "seed42")),
              "M1_seed42_jigsaw_macro_auprc_in_run": D.metrics.get(("M1_RAMT", "seed42"), {}).get("explicit", {}).get("macro_auprc")}
    print("sanity:", sanity)

    boot_cache = {}

    def get(name, cfg, seeds):
        fn, ts, _ = METRICS[name]
        k = (name, cfg, tuple(seeds))
        if k not in boot_cache:
            boot_cache[k] = boot_metric(D, fn, cfg, seeds, ts, B)
        return boot_cache[k]

    def flagged(name, *cfgs):
        s2, _, sds = pooled[name]
        return any(sds.get(c, 0) > 3 * np.sqrt(s2) for c in cfgs) if np.isfinite(s2) else False

    def compare(name, A, Bc):
        seeds = [s for s in D.seeds(A) if s in D.seeds(Bc)]
        fa, ba = get(name, A, seeds); fb, bb = get(name, Bc, seeds)
        s2, df, _ = pooled[name]
        r = S.combined_test(fa, fb, ba, bb, s2, df, flagged(name, A, Bc))
        r.update({"metric": name, "A": A, "B": Bc, "seeds": seeds, "mean_A": float(fa.mean()), "mean_B": float(fb.mean())})
        return r

    # ---- faithfulness pooled variances (attention comp/suff minus random, fixed sets, M1-M3)
    fpool = {}
    for task in ("hx", "explicit", "implicit"):
        for qn in ("comp_attn", "suff_attn"):
            per = {}
            for c in ("M1_RAMT", "M2_MTL_noRat", "M3_random_matched"):
                f, _ = faith_boot(a.explain, c, D.seeds(c), task, "test", Q[qn], 2) if a.explain else (None, None)
                if f is not None:
                    per[c] = list(f)
            fpool[(task, qn)] = S.pooled_seed_variance(per) if per else (np.nan, 0)

    def fcompare(task, qa, A, qb, Bc, split="test"):
        seeds = [s for s in D.seeds(A) if s in D.seeds(Bc)]
        fa, ba = faith_boot(a.explain, A, seeds, task, split, Q[qa], B)
        fb, bb = faith_boot(a.explain, Bc, seeds, task, split, Q[qb], B)
        if fa is None or fb is None:
            return None
        s2, df = fpool[(task, "comp_attn" if "comp" in qa else "suff_attn")]
        r = S.combined_test(fa, fb, ba, bb, s2, df)
        r.update({"metric": f"faith_{qa}_vs_{qb}_{task}", "A": A, "B": Bc, "seeds": seeds,
                  "mean_A": float(fa.mean()), "mean_B": float(fb.mean())})
        return r

    def margin(task, cfg, qn):
        """m = LB95 of the active control's advantage over random on VALIDATION (19.4)."""
        seeds = D.seeds(cfg)
        f, b = faith_boot(a.explain, cfg, seeds, task, "val", Q[qn], B)
        if f is None:
            return np.nan
        s2, df = fpool[(task, "comp_attn" if qn.startswith("C") else "suff_attn")]
        return S.lower_bound_95(f, b, s2, df)

    # ---- plausibility normalisation constants (R5): random floor and leave-one-annotator-out ceiling
    rng = np.random.RandomState(0)
    rand_ap, ceil_ap = [], []
    for w, rats, g in zip(D.hx["words"], D.hx["rationales"], D.gold):
        n = len(g)
        if n == 0 or not (0 < g.sum() < n):
            continue
        rand_ap.append(np.nanmean([per_post_ap([rng.rand(n)], [g])[0] for _ in range(20)]))
        vs = [np.asarray(v[:n], float) for v in rats if len(v) >= n]
        if len(vs) >= 3:
            loo = []
            for i in range(len(vs)):
                others = [v for j, v in enumerate(vs) if j != i]
                go = (np.mean(others, 0) > 0.5).astype(int)
                if 0 < go.sum() < n:
                    loo.append(per_post_ap([vs[i]], [go])[0])
            if loo:
                ceil_ap.append(np.mean(loo))
    plaus_floor, plaus_ceiling = float(np.mean(rand_ap)), float(np.mean(ceil_ap)) if ceil_ap else np.nan
    norm = lambda x: (x - plaus_floor) / (plaus_ceiling - plaus_floor)

    # ================================================================= confirmatory family (12)
    C = {}
    C["H1_explicit"] = compare("jigsaw_macro_auprc", "M2_MTL_noRat", "M5_STL_explicit")
    C["H1_implicit"] = compare("ihc_auprc", "M2_MTL_noRat", "M5_STL_implicit")
    C["H1_hx"] = compare("hx_macro_f1", "M2_MTL_noRat", "M5_STL_hx")
    for other, tag in (("M2_MTL_noRat", "vsM2"), ("M3_random_matched", "vsM3")):
        C[f"H2_hx_macro_f1_{tag}"] = compare("hx_macro_f1", "M1_RAMT", other)
        C[f"H2_plausibility_{tag}"] = compare("plaus_hx_head", "M1_RAMT", other)
        r = fcompare("hx", "comp_attn", "M1_RAMT", "comp_attn", other) if a.explain else None
        C[f"H2_faithfulness_{tag}"] = r
    h3 = {}
    for head in ("explicit", "implicit"):
        r12 = compare(f"plaus_{head}_head", "M1_RAMT", "M2_MTL_noRat")
        r13 = compare(f"plaus_{head}_head", "M1_RAMT", "M3_random_matched")
        p_iut = max(r12["p"], r13["p"])                        # intersection-union test
        C[f"H3_{head}_head"] = {"metric": f"plaus_{head}_head (normalised)", "A": "M1_RAMT", "B": "M2 and M3",
                                "mean_A": r12["mean_A"], "mean_B": max(r12["mean_B"], r13["mean_B"]),
                                "p": p_iut, "diff": min(r12["diff"], r13["diff"]),
                                "sign_consistent": r12["sign_consistent"] and r13["sign_consistent"],
                                "both_positive": r12["diff"] > 0 and r13["diff"] > 0,
                                "vs_M2": r12, "vs_M3": r13,
                                "normalised": {"M1": norm(r12["mean_A"]), "M2": norm(r12["mean_B"]), "M3": norm(r13["mean_B"])}}
        h3[head] = C[f"H3_{head}_head"]
    C["F_gmb_bpsn"] = compare("gmb_bpsn_auc", "M1_RAMT", "M2_MTL_noRat")
    pv = {k: v["p"] for k, v in C.items() if v is not None}
    adj = S.holm(pv)
    for k, v in C.items():
        if v is None:
            continue
        v["p_holm"] = adj[k]
        direction_ok = v.get("both_positive", True)
        v["claim"] = bool(v["p_holm"] < 0.05 and v["sign_consistent"] and direction_ok)

    # ================================================================= H3 decision rule
    deltas = {}
    for task in ("hx", "explicit", "implicit"):
        for ctrl, qC, qS in (("M2_MTL_noRat", "C_attn_minus_rand", "S_rand_minus_attn"),
                             ("M1_RAMT", "C_ig_minus_rand", "S_rand_minus_ig")):
            mC, mS = margin(task, ctrl, qC), margin(task, ctrl, qS)
            deltas[(task, ctrl)] = {"m_C": mC, "m_S": mS,
                                    **{f"delta_C@{f}": (f * mC if mC > 0 else np.nan) for f in FRACS},
                                    **{f"delta_S@{f}": (f * mS if mS > 0 else np.nan) for f in FRACS}}
    noninf = {}
    for task in ("hx", "explicit", "implicit"):                  # E5 / H3 faithfulness leg: M1 attn vs M2 attn
        rc = fcompare(task, "comp_attn", "M1_RAMT", "comp_attn", "M2_MTL_noRat") if a.explain else None
        rs = fcompare(task, "suff_attn", "M1_RAMT", "suff_attn", "M2_MTL_noRat") if a.explain else None
        dl = deltas[(task, "M2_MTL_noRat")]
        noninf[task] = {"comp": rc, "suff": rs, **{
            f"noninferior@{f}": (None if rc is None or not np.isfinite(dl[f"delta_C@{f}"]) else
                                 bool(rc["ci_combined"][0] > -dl[f"delta_C@{f}"] and
                                      (rs is None or not np.isfinite(dl[f"delta_S@{f}"]) or rs["ci_combined"][1] < dl[f"delta_S@{f}"])))
            for f in FRACS}}
    h3_verdict = {}
    for head in ("explicit", "implicit"):
        plaus_claim = C[f"H3_{head}_head"]["claim"]
        only_m2 = (not plaus_claim) and h3[head]["vs_M2"]["p"] < 0.05 and h3[head]["vs_M2"]["sign_consistent"] and h3[head]["vs_M2"]["diff"] > 0
        ni = noninf[head]["noninferior@0.5"]
        h3_verdict[head] = ("supported" if plaus_claim and ni else
                            "partially supported" if (plaus_claim and ni is False) or only_m2 else "not supported")
    H3 = ("supported" if "supported" in h3_verdict.values() else
          "partially supported" if "partially supported" in h3_verdict.values() else "not supported")

    # ================================================================= hypothesis E
    E = {"criteria": {}, "timing": {}}
    for tag, root in (("gpu", a.explain), ("cpu", a.explain_cpu)):
        if not root:
            continue
        files = sorted(glob.glob(os.path.join(root, "timing", f"{tag}_seed*.json")))
        if not files:
            continue
        raw = defaultdict(list); per_seed = {}
        for f in files:
            j = json.load(open(f))
            per_seed[os.path.basename(f)[:-5]] = {k: j[k] for k in ("pred", "pred_attn", "ig", "lime") if k in j}
            for k, v in j["raw_seconds"].items():
                raw[k] += list(v[20:])
        med = {k: 1000 * float(np.median(v)) for k, v in raw.items() if v and k != "ig_gap"}
        p95 = {k: 1000 * float(np.percentile(v, 95)) for k, v in raw.items() if v and k != "ig_gap"}
        E["timing"][tag] = {"median_ms": med, "p95_ms": p95, "per_seed": per_seed}
        E["criteria"][f"E1_{tag}"] = {"ratio": med["pred_attn"] / med["pred"], "pass": med["pred_attn"] <= 1.05 * med["pred"]}
        e2 = {"ratio_ig": med["pred_attn"] / med["ig"], "pass_ig": med["pred_attn"] <= med["ig"] / 10}
        if "lime" in med:
            e2.update({"ratio_lime": med["pred_attn"] / med["lime"], "pass_lime": med["pred_attn"] <= med["lime"] / 100})
        e2["pass"] = e2["pass_ig"] and e2.get("pass_lime", False)
        E["criteria"][f"E2_{tag}"] = e2
        E["criteria"][f"E3_{tag}"] = {"p95_ms": p95["pred_attn"], "target_ms": 100 if tag == "gpu" else 1000,
                                      "met": p95["pred_attn"] <= (100 if tag == "gpu" else 1000)}
    E4 = {}
    for task in ("hx", "explicit", "implicit"):
        rc = fcompare(task, "comp_attn", "M1_RAMT", "comp_ig", "M1_RAMT") if a.explain else None
        rs = fcompare(task, "suff_attn", "M1_RAMT", "suff_ig", "M1_RAMT") if a.explain else None
        dl = deltas[(task, "M1_RAMT")]
        E4[task] = {"comp_attn_minus_ig": rc, "suff_attn_minus_ig": rs, "margins": dl, **{
            f"pass@{f}": (None if rc is None or not np.isfinite(dl[f"delta_C@{f}"]) or not np.isfinite(dl[f"delta_S@{f}"]) else
                          bool(rc["ci_combined"][0] >= -dl[f"delta_C@{f}"] and rs["ci_combined"][1] <= dl[f"delta_S@{f}"]))
            for f in FRACS}}
    E["criteria"]["E4"] = E4
    E["criteria"]["E5"] = noninf
    e4_ok = [v["pass@0.5"] for v in E4.values() if v["pass@0.5"] is not None]
    need = {k: E["criteria"].get(k, {}).get("pass") for k in ("E1_gpu", "E2_gpu", "E1_cpu", "E2_cpu")}
    if any(v is None for v in need.values()) or not e4_ok:
        E["accepted"] = "undecided: " + ", ".join([k for k, v in need.items() if v is None] + ([] if e4_ok else ["E4"])) + " not measured"
    else:
        E["accepted"] = bool(all(need.values()) and all(e4_ok))
    E["note"] = ("E accepted only if E1, E2 (GPU and CPU where measured) and E4 (every assessable head, 50% margin) hold;"
                 " E3 and E5 reported either way (R7).")

    # ================================================================= exploratory
    X = []
    pairs = [("jigsaw_macro_auprc", "M1_RAMT", "M5_STL_explicit"), ("ihc_auprc", "M1_RAMT", "M5_STL_implicit"),
             ("hx_macro_f1", "M1_RAMT", "M5_STL_hx"), ("jigsaw_macro_auprc", "M1_RAMT", "M2_MTL_noRat"),
             ("ihc_auprc", "M1_RAMT", "M2_MTL_noRat"), ("plaus_hx_head", "M1_RAMT", "M4_shared_pool"),
             ("hx_macro_f1", "M1_RAMT", "M4_shared_pool"), ("plaus_hx_head", "M6_STL_hx_rationale", "M5_STL_hx"),
             ("hx_macro_f1", "M6_STL_hx_rationale", "M5_STL_hx"), ("gmb_bpsn_auc", "M1_RAMT", "M3_random_matched"),
             ("gmb_bnsp_auc", "M1_RAMT", "M2_MTL_noRat"), ("gmb_subgroup_auc", "M1_RAMT", "M2_MTL_noRat"),
             ("hatecheck_nonhateful_fpr_hx_head", "M1_RAMT", "M2_MTL_noRat"),
             ("hatecheck_nonhateful_fpr_hx_head", "M1_RAMT", "M3_random_matched")]
    for t, m in (("explicit", "jigsaw_macro_auprc"), ("implicit", "ihc_auprc"), ("hx", "hx_macro_f1")):
        for c in configs:
            if c.startswith(("M7_", "M8_", "CLS_STL_" + t)) and METRICS[m][2](c):
                pairs.append((m, c, f"M5_STL_{t}"))
    for m, A, Bc in pairs:
        if m in full and A in configs and Bc in configs and D.seeds(A) and D.seeds(Bc) and \
                set(D.seeds(A)) & set(D.seeds(Bc)) and METRICS[m][2](A) and METRICS[m][2](Bc):
            X.append(compare(m, A, Bc))
    if a.explain:
        for task in ("hx", "explicit", "implicit"):
            for qn in ("comp_attn", "suff_attn"):
                for other in ("M2_MTL_noRat", "M3_random_matched"):
                    if task == "hx" and qn == "comp_attn":
                        continue                                   # confirmatory already
                    r = fcompare(task, qn, "M1_RAMT", qn, other)
                    if r:
                        X.append(r)

    # ================================================================= write
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    def row(k, v):
        return {"test": k, "metric": v["metric"], "A": v["A"], "B": v["B"], "mean_A": v.get("mean_A"), "mean_B": v.get("mean_B"),
                "diff_A_minus_B": v["diff"], "seed_sd": v.get("seed_sd"), "ci_combined_lo": v.get("ci_combined", [None, None])[0],
                "ci_combined_hi": v.get("ci_combined", [None, None])[1], "ci_instance_lo": v.get("ci_instance_only", [None, None])[0],
                "ci_instance_hi": v.get("ci_instance_only", [None, None])[1], "df": v.get("df"), "seed_term": v.get("seed_term"),
                "p": v["p"], "p_holm": v.get("p_holm"), "seeds_A_better": v.get("n_seeds_A_better"), "K": v.get("K"),
                "sign_consistent": v["sign_consistent"], "claim": v.get("claim")}
    conf = pd.DataFrame([row(k, v) for k, v in C.items() if v is not None])
    conf.to_csv(os.path.join(a.out, "confirmatory_tests.csv"), index=False)
    pd.DataFrame([row(f"X{i+1}", v) for i, v in enumerate(X)]).to_csv(os.path.join(a.out, "exploratory_tests.csv"), index=False)
    desc = []
    for name, vals in full.items():
        per = defaultdict(list)
        for (c, s), v in vals.items():
            per[c].append(v)
        for c, v in per.items():
            desc.append({"metric": name, "config": c, "n_seeds": len(v), "mean": float(np.mean(v)),
                         "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                         "normalised_mean": float(norm(np.mean(v))) if name.startswith("plaus") else None})
    pd.DataFrame(desc).sort_values(["metric", "config"]).to_csv(os.path.join(a.out, "descriptive.csv"), index=False)
    out = {"confirmatory": C, "H3": {"per_head": h3_verdict, "verdict": H3, "faithfulness_noninferiority": noninf},
           "E": E, "margins": {f"{t}|{c}": v for (t, c), v in deltas.items()},
           "plausibility_normalisation": {"random_floor": plaus_floor, "human_ceiling_loo": plaus_ceiling,
                                          "n_posts_ceiling": len(ceil_ap)},
           "pooled_seed_variance": {k: {"sigma2": v[0], "df": v[1], "config_sd": v[2]} for k, v in pooled.items()},
           "faith_pooled_variance": {f"{t}|{q}": {"sigma2": v[0], "df": v[1]} for (t, q), v in fpool.items()},
           "communities": D.community_names, "sanity": sanity, "B": B}
    json.dump(clean(out), open(os.path.join(a.out, "analysis.json"), "w"), indent=1)
    print("\nCONFIRMATORY (Holm over", len(pv), "tests)")
    print(conf[["test", "mean_A", "mean_B", "diff_A_minus_B", "ci_combined_lo", "ci_combined_hi", "p_holm",
                "sign_consistent", "claim"]].to_string(index=False))
    print("\nH3:", h3_verdict, "->", H3)
    print("E accepted:", E["accepted"], json.dumps(clean({k: (v if k != "E4" and k != "E5" else
          {t: {kk: vv for kk, vv in d.items() if kk.startswith(("pass", "noninferior"))} for t, d in v.items()})
          for k, v in E["criteria"].items()}), indent=0)[:2500])
    print("\nwrote", a.out)


def tasks_of(D, cfg):
    s = D.seeds(cfg)[0]
    return json.load(open(os.path.join(D.dirs[(cfg, s)], "config.json")))["tasks"]


def pooled_attn(D, cfg):
    s = D.seeds(cfg)[0]
    return json.load(open(os.path.join(D.dirs[(cfg, s)], "config.json"))).get("pooling", "task") != "cls"


if __name__ == "__main__":
    main()
