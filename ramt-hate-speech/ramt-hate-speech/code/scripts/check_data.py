"""Gate G1 (Phase 2): integrity checks on a prepared splits directory.

    python scripts/check_data.py --splits /kaggle/working/splits \
        [--tokenizers GroNLP/hateBERT roberta-base microsoft/deberta-v3-base]

Checks (fail = non-zero exit):
  1. split files present and identical to split_hashes.json
  2. zero normalised-text overlap train x {val, test, OLID, HateCheck}; val x test
  3. HateXplain: rationale vectors match post length; normal posts carry none; eligible
     share per split; P(32-post batch has no eligible post); degenerate random-control posts
  4. word limit W: share of training posts kept whole per task (target >= 95% overall)
  5. per tokenizer: sub-word alignment keeps rationale mass (eligible posts that become
     ineligible after truncation at W / L), special tokens get zero mass
Writes <splits>/check_data.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt.data.datasets import available, load_items, load_tokenizer  # noqa: E402
from ramt.data.prep import norm_key  # noqa: E402
from ramt.data.rationales import aggregate  # noqa: E402
from ramt.utils import md5_file, write_json  # noqa: E402


def _keys(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)["words"].map(" ".join).map(norm_key)
    return pd.read_csv(path)["text"].astype(str).map(norm_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tokenizers", nargs="*", default=[])
    ap.add_argument("--max_subwords", type=int, default=0, help="0 = read subword_limit.json if present")
    a = ap.parse_args()
    S = a.splits
    R, fails = {}, []

    def check(name, ok, detail):
        R[name] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(detail, default=str)[:300])
        if not ok:
            fails.append(name)

    st = json.load(open(os.path.join(S, "data_stats.json")))
    # 1 hashes
    hashes = json.load(open(os.path.join(S, "split_hashes.json")))
    bad = [f for f, h in hashes.items() if not os.path.exists(os.path.join(S, f)) or md5_file(os.path.join(S, f)) != h]
    check("split_hashes", not bad, {"files": len(hashes), "mismatched_or_missing": bad})
    check("ihc_status_recorded", "ihc_status" in st, st.get("ihc_status"))

    # 2 overlap
    files = sorted(os.listdir(S))
    evals = [f for f in files if f.endswith(("_test.csv", "_test.jsonl")) or f in ("olid_test.csv", "hatecheck.csv")]
    trains = [f for f in files if f.endswith(("_train.csv", "_train.jsonl"))]
    vals = [f for f in files if f.endswith(("_val.csv", "_val.jsonl"))]
    ek = set().union(*[set(_keys(os.path.join(S, f))) for f in evals]) if evals else set()
    tk = set().union(*[set(_keys(os.path.join(S, f))) for f in evals if "_test" in f]) if evals else set()
    ov = {f: len(set(_keys(os.path.join(S, f))) & ek) for f in trains}
    check("train_eval_overlap_zero", all(v == 0 for v in ov.values()), ov)
    ovv = {f: len(set(_keys(os.path.join(S, f))) & tk) for f in vals}
    check("val_test_overlap_zero", all(v == 0 for v in ovv.values()), ovv)

    # 3 HateXplain rationale structure
    hx = {}
    for sp in ("train", "val", "test"):
        df = pd.read_json(os.path.join(S, f"hx_{sp}.jsonl"), lines=True)
        len_ok = all(len(v) == len(w) for w, rs in zip(df["words"], df["rationales"]) for v in rs)
        normal_clean = bool((df.loc[df["label"] == 2, "rationales"].map(len) == 0).all())
        elig = np.array([lab != 2 and len(rs) > 0 and aggregate(rs, len(w), "soft").sum() > 0
                         for w, lab, rs in zip(df["words"], df["label"], df["rationales"])])
        p = float(elig.mean())
        all_marked = int(sum(lab != 2 and len(rs) > 0 and (aggregate(rs, len(w), "soft") > 0).all()
                             for w, lab, rs in zip(df["words"], df["label"], df["rationales"])))
        hx[sp] = {"n": len(df), "vector_lengths_match": len_ok, "normal_posts_without_rationales": normal_clean,
                  "eligible_share": p, "p_empty_batch_32": (1 - p) ** 32,
                  "degenerate_random_control_posts_all_words_marked": all_marked}
    check("hx_rationale_structure", all(v["vector_lengths_match"] and v["normal_posts_without_rationales"]
                                        for v in hx.values()), hx)
    check("hx_empty_batch_negligible", hx["train"]["p_empty_batch_32"] < 1e-6,
          {"p_empty_batch_32_train": hx["train"]["p_empty_batch_32"]})

    # 4 word limit W
    W = int(st["max_words_W"])
    cov = {}
    for t in ("explicit", "implicit", "hx"):
        if not available(S, t, "train"):
            continue
        p = [f for f in trains if f.startswith(t + "_")][0]
        df = pd.read_json(os.path.join(S, p), lines=True) if p.endswith("jsonl") else pd.read_csv(os.path.join(S, p))
        n = df["words"].map(len) if "words" in df else df["text"].astype(str).str.split().str.len()
        cov[t] = {"n": len(n), "share_whole_at_W": float((n <= W).mean())}
    allw = sum(v["n"] * v["share_whole_at_W"] for v in cov.values()) / max(sum(v["n"] for v in cov.values()), 1)
    check("word_limit_W_coverage", allw >= 0.95 - 1e-9, {"W": W, "pooled_share_whole": allw, **cov})

    # 5 tokenizer alignment
    L = a.max_subwords
    lim = os.path.join(S, "subword_limit.json")
    if not L and os.path.exists(lim):
        L = json.load(open(lim))["max_subwords_L"]
    L = L or 256
    tokres = {}
    for name in a.tokenizers:
        tok = load_tokenizer(name)
        items = load_items(S, "hx", "train", tok, W, L, "soft", "human")
        raw = pd.read_json(os.path.join(S, "hx_train.jsonl"), lines=True)
        pre = sum(lab != 2 and len(rs) > 0 for lab, rs in zip(raw["label"], raw["rationales"]))
        post = sum(bool(it["elig"]) for it in items)
        special_mass = max(float(np.asarray(it["q"])[[i for i, w in enumerate(it["wid"]) if w is None]].sum())
                           if it["elig"] else 0.0 for it in items)
        mass_ok = all(float(np.sum(it["q"])) > 0 for it in items if it["elig"])   # loss normalises q
        trunc = float(np.mean([len(it["ids"]) >= L for it in items]))
        tokres[name] = {"eligible_before": int(pre), "eligible_after_truncation": int(post),
                        "lost_to_truncation": int(pre - post), "max_mass_on_special_tokens": special_mass,
                        "eligible_have_positive_mass": mass_ok, "share_posts_hitting_L": trunc, "L": L}
    if tokres:
        check("subword_alignment", all(v["max_mass_on_special_tokens"] == 0 and v["eligible_have_positive_mass"]
                                       for v in tokres.values()), tokres)

    R["_summary"] = {"failed": fails, "passed": not fails}
    write_json(R, os.path.join(S, "check_data.json"))
    print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
