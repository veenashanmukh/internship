"""HateXplain-protocol evaluation (methodology 8.6), corrected fairness test F, and IHC shortcut
check (6.5). CPU only.

    python scripts/hatexplain_eval.py --runs <trained runs> --splits <splits> --out /kaggle/working/hx_protocol
           [--with_models]   # adds ERASER comprehensiveness / sufficiency for runs with best.pt (slow on CPU)

Definitions follow HateXplain's released code (github.com/hate-alert/HateXplain, MIT), read 25 Nov 2026:
  classification  accuracy, macro-F1, AUROC = roc_auc_score(y, probs, multi_class="ovo", average="macro"),
                  all test posts, 3 classes.
  plausibility    non-normal test posts; gold = UNION of annotators' rationales; hard rationale = top-5
                  words by attention (Python sorted() on attention, last 5 indices); IOU-F1 = ERASER
                  partial_match_score, threshold 0.5, macro; token-F1 = ERASER instance_macro F1;
                  AUPRC = ERASER _auprc (auc of precision_recall_curve per post, averaged).
  faithfulness    ERASER one-shot: p_full[c] - p_without_top5[c] (comprehensiveness) and
                  p_full[c] - p_only_top5[c] (sufficiency), c = predicted class, 3-class softmax.
  bias            community = named by more than one annotator; top-10 communities over the whole
                  dataset (list below, reproduced from HateXplain's code on the raw data); posts with
                  none of the 10 are EXCLUDED from every set; toxic = hatespeech or offensive;
                  score = 1 - p(normal); subgroup / BPSN / BNSP AUC; generalised mean, p = -5.
Unit differences: our tokens are words (HateXplain: BERT word pieces); our models are RoBERTa.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt import stats as S  # noqa: E402

HX_TOP10 = ["African", "Islam", "Jewish", "Homosexual", "Women", "Refugee", "Arab", "Caucasian", "Asian", "Hispanic"]
MAIN = {"M1_RAMT", "M2_MTL_noRat", "M3_random_matched", "M4_shared_pool", "M5_STL_hx", "M6_STL_hx_rationale",
        "M7_pair_explicit+hx", "M7_pair_implicit+hx", "M8_MTL_noRat_hatebert", "M8_MTL_noRat_deberta"}


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def top5(scores):
    return sorted(range(len(scores)), key=lambda i: scores[i])[-5:]


def spans(idx):
    idx = sorted(idx); out = []
    for i in idx:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(s) for s in out]


def eraser_plausibility(attn_list, gold_list):
    """IOU-F1 (macro, 0.5), token-F1 (instance macro) and AUPRC as in ERASER metrics.py."""
    n_truth, n_pred, tps = {}, {}, {}
    tok_f1, aucs = [], []
    for k, (a, g) in enumerate(zip(attn_list, gold_list)):
        a = np.asarray(a, float); g = np.asarray(g[:len(a)], int)
        pred_idx = top5(list(a))
        P, T = spans(pred_idx), spans(np.nonzero(g)[0].tolist())
        if T:
            n_truth[k] = len(T)
        n_pred[k] = len(P)
        tps[k] = sum(int(max([len(set(range(*p)) & set(range(*t))) / len(set(range(*p)) | set(range(*t))) for t in T]
                                 or [0.0]) >= 0.5) for p in P)
        ts, ps = set(np.nonzero(g)[0].tolist()), set(pred_idx)
        pr = len(ts & ps) / len(ps) if ps else 0.0
        rc = len(ts & ps) / len(ts) if ts else 0.0
        tok_f1.append(0.0 if pr == 0 or rc == 0 else 2 * pr * rc / (pr + rc))
        if g.sum() > 0:
            p_, r_, _ = precision_recall_curve(g, a)
            aucs.append(auc(r_, p_))
    mr = np.mean([tps[k] / n for k, n in n_truth.items()]) if n_truth else 0.0
    mp = np.mean([tps[k] / n for k, n in n_pred.items()]) if n_pred else 0.0
    iou_f1 = 0.0 if mr == 0 or mp == 0 else 2 * mr * mp / (mr + mp)
    return {"iou_f1": float(iou_f1), "token_f1": float(np.mean(tok_f1)), "auprc": float(np.mean(aucs)), "n_posts": len(gold_list)}


def bias_sets(hx):
    comm = []
    for tl in hx["targets"]:
        cnt = Counter(g for ann in (tl or [])[:3] for g in (ann or []))
        s = {g for g, n in cnt.items() if n > 1} & set(HX_TOP10)
        comm.append(s)
    targeted = np.array([len(s) > 0 for s in comm])
    tox = hx["label"].values != 2
    subs = {c: np.array([c in s for s in comm]) for c in HX_TOP10}
    sets = {}
    for c, sub in subs.items():
        sets[("subgroup", c)] = targeted & sub
        sets[("bpsn", c)] = targeted & ((sub & ~tox) | (~sub & tox))
        sets[("bnsp", c)] = targeted & ((sub & tox) | (~sub & ~tox))
    return sets, tox


def gmb(scores, tox, sets, kind, W=None):
    W = np.ones((1, len(tox))) if W is None else np.atleast_2d(W)
    vals = np.stack([S.w_roc_auc(tox, scores, W * m) for (k, c), m in sets.items() if k == kind], -1)
    return S.generalised_mean(vals, -5.0), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_models", action="store_true")
    ap.add_argument("--B", type=int, default=S.B_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hx = pd.read_json(os.path.join(a.splits, "hx_test.jsonl"), lines=True)
    y = hx["label"].values.astype(int)
    W = int(json.load(open(os.path.join(a.splits, "data_stats.json")))["max_words_W"])
    union = [(np.max(np.asarray(r, int), 0) if len(r) else np.zeros(len(w), int))[:W] for w, r in zip(hx["words"], hx["rationales"])]
    nonnormal = np.nonzero(y != 2)[0]
    sets, tox = bias_sets(hx)
    print("test posts:", len(y), "| non-normal:", len(nonnormal), "| posts in a top-10 community:",
          int(sets[("subgroup", HX_TOP10[0])].sum()), "(", HX_TOP10[0], ")")

    runs = {}
    for m in glob.glob(os.path.join(a.runs, "**", "test_hx__hx.csv"), recursive=True):
        d = os.path.dirname(m); p = d.split(os.sep)
        if not p[-3].startswith("pilot"):
            runs[(p[-2], p[-1])] = d
    rows, bias_full = [], {}
    for (cfg, seed), d in sorted(runs.items()):
        df = pd.read_csv(os.path.join(d, "test_hx__hx.csv"))
        P = softmax(df[[c for c in df.columns if c.startswith("logit_")]].values.astype(float))
        assert len(P) == len(y)
        r = {"config": cfg, "seed": seed, "accuracy": accuracy_score(y, P.argmax(1)),
             "macro_f1": f1_score(y, P.argmax(1), average="macro"),
             "auroc": roc_auc_score(y, P, multi_class="ovo", average="macro")}
        sc = 1 - P[:, 2]
        for kind in ("subgroup", "bpsn", "bnsp"):
            g, _ = gmb(sc, tox, sets, kind)
            r[f"gmb_{kind}"] = float(g[0])
        bias_full[(cfg, seed)] = r["gmb_bpsn"]
        ap_ = os.path.join(d, "attention_hx_test.npz")
        cfgj = json.load(open(os.path.join(d, "config.json")))
        if os.path.exists(ap_) and cfgj.get("pooling", "task") != "cls":
            z = np.load(ap_, allow_pickle=True)
            r.update({f"plaus_{k}": v for k, v in eraser_plausibility([z["hx"][i] for i in nonnormal],
                                                                         [union[i] for i in nonnormal]).items()})
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "hatexplain_protocol_per_run.csv"), index=False)
    agg = T.drop(columns="seed").groupby("config").agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.round(4).to_csv(os.path.join(a.out, "hatexplain_protocol_summary.csv"))
    print(T.drop(columns="seed").groupby("config").mean(numeric_only=True).round(4).to_string())

    # ---------------- corrected confirmatory F: M1 vs M2 GMB-BPSN under HateXplain's exact sets
    per = {}
    for (c, s), v in bias_full.items():
        if c in MAIN:
            per.setdefault(c, []).append(v)
    s2, dfp = S.pooled_seed_variance(per)
    seeds = sorted(set(s for c, s in runs if c == "M1_RAMT") & set(s for c, s in runs if c == "M2_MTL_noRat"),
                   key=lambda s: int(s[4:]))
    def load_sc(c, s):
        df = pd.read_csv(os.path.join(runs[(c, s)], "test_hx__hx.csv"))
        return 1 - softmax(df[[k for k in df.columns if k.startswith("logit_")]].values.astype(float))[:, 2]
    A = {s: load_sc("M1_RAMT", s) for s in seeds}; Bm = {s: load_sc("M2_MTL_noRat", s) for s in seeds}
    fa = np.array([bias_full[("M1_RAMT", s)] for s in seeds]); fb = np.array([bias_full[("M2_MTL_noRat", s)] for s in seeds])
    ba = np.zeros((len(seeds), a.B)); bb = np.zeros((len(seeds), a.B)); zero_auc = 0
    for b0, C in S.boot_counts(len(y), a.B, seed=13):
        for i, s in enumerate(seeds):
            ga, va = gmb(A[s], tox, sets, "bpsn", C); gb, vb = gmb(Bm[s], tox, sets, "bpsn", C)
            ba[i, b0:b0 + len(C)] = ga; bb[i, b0:b0 + len(C)] = gb
            zero_auc += int((va < 0.05).sum() + (vb < 0.05).sum())
    sds = {c: np.std(v, ddof=1) for c, v in per.items() if len(v) > 1}
    flag = any(sds.get(c, 0) > 3 * np.sqrt(s2) for c in ("M1_RAMT", "M2_MTL_noRat"))
    F = S.combined_test(fa, fb, ba, bb, s2, dfp, flag)
    F.update({"definition": "HateXplain exact (untargeted posts excluded; top-10 over whole dataset)",
              "mean_A": float(fa.mean()), "mean_B": float(fb.mean()), "seeds": seeds,
              "bootstrap_community_aucs_below_0.05": zero_auc})
    # per-community BPSN, mean over seeds (descriptive)
    pc = {}
    for c in ("M1_RAMT", "M2_MTL_noRat", "M3_random_matched"):
        vs = []
        for s in sorted(s for cc, s in runs if cc == c):
            _, v = gmb(load_sc(c, s), tox, sets, "bpsn"); vs.append(v[0])
        pc[c] = dict(zip(HX_TOP10, np.mean(vs, 0).round(4).tolist()))
    F["per_community_bpsn_mean"] = pc
    json.dump(F, open(os.path.join(a.out, "F_corrected.json"), "w"), indent=1, default=float)
    print("\nCORRECTED F (M1 - M2, GMB-BPSN):", {k: F[k] for k in ("diff", "ci_combined", "p", "sign_consistent", "n_seeds_A_better")})
    print("per-community BPSN:", json.dumps(pc))

    # ---------------- ERASER comprehensiveness / sufficiency (runs with checkpoints)
    if a.with_models:
        import torch
        from ramt.data.datasets import _encode, load_tokenizer
        from ramt.evaluate import predict
        from ramt.model import RAMT
        torch.set_num_threads(os.cpu_count() or 1)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        L = json.load(open(os.path.join(a.splits, "subword_limit.json")))["max_subwords_L"] if os.path.exists(os.path.join(a.splits, "subword_limit.json")) else 256
        fr = []
        for (cfg, seed), d in sorted(runs.items()):
            if not os.path.exists(os.path.join(d, "best.pt")):
                continue
            cj = json.load(open(os.path.join(d, "config.json")))
            m = RAMT(cj["encoder"], cj["tasks"], cj.get("pooling", "task"))
            m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in torch.load(os.path.join(d, "best.pt"), map_location="cpu").items()})
            m.to(dev).eval(); tok = load_tokenizer(cj["encoder"])
            z = np.load(os.path.join(d, "attention_hx_test.npz"), allow_pickle=True)
            def probs(wl):
                enc = _encode(tok, [w[:W] for w in wl], L)
                it = [{"ids": i, "wid": w_, "y": np.zeros(1, np.float32)} for i, w_ in enc]
                return softmax(predict(m, it, ["hx"], dev, tok.pad_token_id, 128, dev == "cuda")["hx"]["logits"].astype(float))
            words = [list(hx["words"].iloc[i])[:W] for i in nonnormal]
            keep = [set(top5(list(z["hx"][i]))) for i in nonnormal]
            full = probs(words); c = full.argmax(1)
            wo = probs([[w for j, w in enumerate(ws) if j not in k] for ws, k in zip(words, keep)])
            on = probs([[w for j, w in enumerate(ws) if j in k] for ws, k in zip(words, keep)])
            rr = np.arange(len(c))
            fr.append({"config": cfg, "seed": seed, "comprehensiveness": float(np.mean(full[rr, c] - wo[rr, c])),
                       "sufficiency": float(np.mean(full[rr, c] - on[rr, c]))})
            print("eraser faithfulness", cfg, seed, fr[-1])
        pd.DataFrame(fr).to_csv(os.path.join(a.out, "hatexplain_protocol_faithfulness.csv"), index=False)

    # ---------------- IHC shortcut check (6.5)
    from sklearn.linear_model import LogisticRegression
    tr = pd.read_csv(os.path.join(a.splits, "implicit_train.csv")); te = pd.read_csv(os.path.join(a.splits, "implicit_test.csv"))
    def feats(t):
        t = t.astype(str)
        return np.c_[t.str.match(r"^\s*(rt|RT)\b").astype(int), t.str.count('"'), t.str.count("'"),
                     t.str.count(r"@USER|<user>|@user"), t.str.count(r"URL|<url>|http"), t.str.count("#"),
                     t.str.split().str.len()]
    names = ["starts_with_RT", "double_quotes", "single_quotes", "user_mentions", "urls", "hashtags", "n_words"]
    Xtr, Xte = feats(tr["text"]), feats(te["text"])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, tr["label"])
    auc_sc = roc_auc_score(te["label"], clf.predict_proba(Xte)[:, 1])
    rates = {n: {"implicit": float(Xte[te["label"] == 1, i].mean()), "not_hate": float(Xte[te["label"] == 0, i].mean())}
             for i, n in enumerate(names)}
    SC = {"test_auroc_surface_features_only": float(auc_sc), "feature_means_test": rates,
          "coefficients": dict(zip(names, clf.coef_[0].round(4).tolist()))}
    json.dump(SC, open(os.path.join(a.out, "ihc_shortcut_check.json"), "w"), indent=1)
    print("\nIHC shortcut check: AUROC from surface features alone =", round(auc_sc, 4))
    print(json.dumps(rates, indent=0)[:1500])


if __name__ == "__main__":
    main()
