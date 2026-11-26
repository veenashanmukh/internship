"""Raw datasets -> frozen, de-duplicated splits (Section 1, Section 14 step S0).

Run once in a Kaggle CPU notebook:  python scripts/prepare_data.py --help

Outputs in --out:
  explicit_{train,val,valsub,test}.csv   text + 6 Jigsaw labels (test = official Kaggle test)
  implicit_{train,val,test}.csv          ONLY if IHC permission is recorded (D1)
  hx_{train,val,test}.jsonl              words, label, per-annotator rationale vectors, targets
  olid_test.csv, hatecheck.csv           zero-shot only, never trained on
  data_stats.json, dedup_report.json, split_hashes.json, label_semantics.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ramt.utils import md5_file, write_json

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HX_CLASSES = ["hatespeech", "offensive", "normal"]          # index = label id

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
USER_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


# ----------------------------------------------------------------------------- text
def clean_text(t) -> str:
    """Identical light cleaning for every dataset (Section 1.8). No lower-casing (7.4)."""
    t = str(t)
    t = t.replace("<user>", " @user ").replace("<url>", " http ")
    t = URL_RE.sub(" http ", t)
    t = USER_RE.sub(" @user ", t)
    t = t.replace("@USER", "@user").replace(" URL ", " http ")
    return WS_RE.sub(" ", t).strip()


def clean_word(w: str) -> str:
    c = clean_text(w)
    return c if c else w


def norm_key(t) -> str:
    """Aggressive normalisation used only to detect duplicates (Section 1.6)."""
    t = clean_text(t).lower()
    t = re.sub(r"@user|\bhttp\b|\brt\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return WS_RE.sub(" ", t).strip()


# ----------------------------------------------------------------------------- io
def _find(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} under {d}")
    return hits[0]


def _read(path, **kw):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith((".csv", ".tsv"))][0]
            with z.open(name) as f:
                return pd.read_csv(f, sep="\t" if name.endswith(".tsv") else ",", **kw)
    return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", **kw)


def _strat_key(df, cols, min_count=2):
    key = df[cols].astype(int).astype(str).agg("".join, axis=1)
    cnt = key.value_counts()
    key = key.where(key.map(cnt) >= min_count, "rare")
    if (key == "rare").sum() < min_count:                  # a lone rare row joins the largest stratum
        key = key.where(key != "rare", key.value_counts().idxmax())
    return key


# ----------------------------------------------------------------------------- loaders
def load_jigsaw(jdir, n_clean_train, val_frac, valsub_n, seed):
    tr = _read(_find(jdir, "train.csv*"))
    te = _read(_find(jdir, "test.csv*")).merge(_read(_find(jdir, "test_labels.csv*")), on="id")
    n_test_raw = len(te)
    te = te[(te[LABELS] >= 0).all(axis=1)].copy()                     # 1.4: drop -1 rows
    for d in (tr, te):
        d["text"] = d["comment_text"].map(clean_text)
    tr, te = tr[["id", "text"] + LABELS], te[["id", "text"] + LABELS]
    tr_part, va = train_test_split(tr, test_size=val_frac, random_state=seed, stratify=_strat_key(tr, LABELS))
    tox = tr_part[tr_part[LABELS].sum(axis=1) > 0]
    clean = tr_part[tr_part[LABELS].sum(axis=1) == 0]
    n_clean_avail = len(clean)
    if n_clean_train and n_clean_train < len(clean):
        clean = clean.sample(n=n_clean_train, random_state=seed)
    tr_sub = pd.concat([tox, clean]).sample(frac=1, random_state=seed).reset_index(drop=True)
    info = {"train_csv_rows": len(tr), "train_toxic_rows_after_val_split": len(tox),
            "train_clean_available": n_clean_avail, "train_clean_used": len(clean),
            "test_rows_raw": n_test_raw, "test_rows_labelled": len(te)}
    return tr_sub, va.reset_index(drop=True), te.reset_index(drop=True), info


def load_ihc(path, seed):
    """IHC stage-1: implicit_hate vs not_hate; explicit_hate kept aside as a probe (1.9)."""
    df = _read(path)
    tcol = next(c for c in ["post", "text", "tweet"] if c in df.columns)
    lcol = next(c for c in ["class", "label"] if c in df.columns)
    idcol = next((c for c in ["ID", "id", "post_id"] if c in df.columns), None)
    df = df.rename(columns={tcol: "text", lcol: "cls"})
    df["row_index"] = np.arange(len(df))                                   # release key if no ID column
    df["src_id"] = df[idcol].astype(str) if idcol else df["row_index"].astype(str)
    cls = df["cls"].astype(str).str.strip().str.lower()
    if "not_hate" not in set(cls):
        raise ValueError("No 'not_hate' rows: this is not the IHC stage-1 file.")
    counts = cls.value_counts().to_dict()
    df["text"] = df["text"].map(clean_text)
    probe = df[cls == "explicit_hate"][["src_id", "text"]].reset_index(drop=True)
    df = df[cls.isin(["implicit_hate", "not_hate"])].copy()
    df["label"] = (df["cls"].astype(str).str.strip().str.lower() == "implicit_hate").astype(int)
    df = df[["src_id", "text", "label"]]
    tr, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    tr, va = train_test_split(tr, test_size=0.125, random_state=seed, stratify=tr["label"])  # 70/10/20
    return (tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True),
            probe, {"class_counts_in_file": counts})


def load_hatexplain(hdir):
    data = json.load(open(_find(hdir, "dataset.json")))
    div = json.load(open(_find(hdir, "post_id_divisions.json")))
    split_of = {pid: s for s, ids in div.items() for pid in ids}
    rows = {"train": [], "val": [], "test": []}
    info = Counter()
    for pid, p in data.items():
        s = split_of.get(pid)
        if s is None:
            info["not_in_divisions"] += 1
            continue
        labs = [a["label"] for a in p["annotators"]]
        lab, cnt = Counter(labs).most_common(1)[0]
        if cnt < 2:
            info["no_majority_in_divisions"] += 1
            continue
        n = len(p["post_tokens"])
        vecs = [list(map(int, r)) for r in p.get("rationales", []) if len(r) == n]
        info["rationale_vectors_dropped_length_mismatch"] += len(p.get("rationales", [])) - len(vecs)
        if lab == "normal":
            vecs = []                                   # 3.4/3.5: rationales only for toxic majority
        rows[s].append({"id": pid, "words": [clean_word(w) for w in p["post_tokens"]],
                        "label": HX_CLASSES.index(lab), "rationales": vecs,
                        "targets": [a.get("target", []) for a in p["annotators"]],
                        "text": " ".join(p["post_tokens"])})
    return {k: pd.DataFrame(v) for k, v in rows.items()}, dict(info)


def load_olid(odir):
    try:
        t = pd.read_csv(_find(odir, "testset-levela.tsv"), sep="\t")
        lab = pd.read_csv(_find(odir, "labels-levela.csv"), header=None, names=["id", "lab"])
        df = t.merge(lab, on="id")
        df["label"] = (df["lab"] == "OFF").astype(int)
        df["text"] = df["tweet"].map(clean_text)
    except FileNotFoundError:
        df = _read(_find(odir, "*test*.csv"))
        df["text"] = df["text"].map(clean_text)
        df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def load_hatecheck(path):
    df = pd.read_csv(path)
    out = pd.DataFrame({"text": df["test_case"].map(clean_text),
                        "label": (df["label_gold"] == "hateful").astype(int),
                        "functionality": df["functionality"],
                        "target_ident": df.get("target_ident", pd.Series([None] * len(df)))})
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- helpers
def choose_W(word_counts, cover=0.95):
    """Section 7.1: smallest W keeping >= 95% of training posts whole."""
    wc = np.sort(np.asarray(word_counts))
    return int(wc[int(np.ceil(cover * len(wc))) - 1])


def degenerate_random_control(rationale_vectors_list):
    """Posts where a within-post permutation of the soft target cannot change it (8.5)."""
    n = 0
    for vecs in rationale_vectors_list:
        if not vecs:
            continue
        rho = np.mean(np.asarray(vecs, float), axis=0)
        if rho.max() > 0 and np.allclose(rho, rho[0]):
            n += 1
    return n


# ----------------------------------------------------------------------------- main
def prepare(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    report, stats = {}, {}

    ex_tr, ex_va, ex_te, jinfo = load_jigsaw(a.jigsaw_dir, a.n_clean_train, a.val_frac, a.valsub_n, a.seed)
    hx, hxinfo = load_hatexplain(a.hx_dir)
    olid = load_olid(a.olid_dir) if a.olid_dir else None
    hc = load_hatecheck(a.hatecheck_file) if a.hatecheck_file else None

    ihc_status = "not provided"
    im = None
    if a.ihc_file:
        if a.ihc_permission_record and os.path.exists(a.ihc_permission_record):
            im = load_ihc(a.ihc_file, a.seed)
            ihc_status = f"used; permission record: {os.path.basename(a.ihc_permission_record)}"
        else:
            ihc_status = "BLOCKED: no written permission record supplied (D1); IHC not processed"
    report["ihc_status"] = ihc_status
    print("IHC:", ihc_status)

    def dedup_within(df, name):
        k = df["text"].map(norm_key)
        keep = ~k.duplicated() & (k.str.len() > 0)
        if name.endswith("_test"):
            # official test sets stay intact for comparability with published results;
            # duplicates are only counted
            report[f"{name}_within_duplicates_counted_not_removed"] = int((~keep).sum())
            return df
        report[f"{name}_within_duplicates_removed"] = int((~keep).sum())
        return df[keep].reset_index(drop=True)

    ex_tr, ex_va, ex_te = (dedup_within(ex_tr, "explicit_train"), dedup_within(ex_va, "explicit_val"),
                           dedup_within(ex_te, "explicit_test"))
    hx = {s: dedup_within(d, f"hx_{s}") for s, d in hx.items()}
    if im:
        im_tr, im_va, im_te, im_probe, iminfo = im
        im_tr, im_va, im_te = (dedup_within(im_tr, "implicit_train"), dedup_within(im_va, "implicit_val"),
                               dedup_within(im_te, "implicit_test"))

    test_frames = [ex_te, hx["test"]] + ([im_te] if im else []) + [f for f in (olid, hc) if f is not None]
    val_frames = [ex_va, hx["val"]] + ([im_va] if im else [])
    keys = lambda frames: set().union(*[set(f["text"].map(norm_key)) for f in frames])
    test_keys, eval_keys = keys(test_frames), keys(test_frames + val_frames)

    def drop(df, name, ks):
        bad = df["text"].map(norm_key).isin(ks)
        report[f"{name}_overlap_removed"] = int(bad.sum())
        return df[~bad].reset_index(drop=True)

    ex_tr = drop(ex_tr, "explicit_train", eval_keys)
    hx["train"] = drop(hx["train"], "hx_train", eval_keys)
    ex_va = drop(ex_va, "explicit_val", test_keys)
    hx["val"] = drop(hx["val"], "hx_val", test_keys)
    if im:
        im_tr = drop(im_tr, "implicit_train", eval_keys)
        im_va = drop(im_va, "implicit_val", test_keys)

    train_keys = set(ex_tr["text"].map(norm_key)) | set(hx["train"]["text"].map(norm_key))
    if im:
        train_keys |= set(im_tr["text"].map(norm_key))
    assert not (train_keys & test_keys), "train/test overlap remains"
    report["final_train_test_overlap"] = 0

    # fixed 5,000-post stratified Jigsaw validation subset for per-round validation (8.2)
    n_sub = min(a.valsub_n, len(ex_va))
    ex_vs = ex_va if n_sub == len(ex_va) else train_test_split(
        ex_va, train_size=n_sub, random_state=a.seed, stratify=_strat_key(ex_va, LABELS))[0]

    # ---- write
    w = lambda df, n: df.to_csv(os.path.join(a.out, n), index=False)
    w(ex_tr, "explicit_train.csv"); w(ex_va, "explicit_val.csv"); w(ex_vs, "explicit_valsub.csv"); w(ex_te, "explicit_test.csv")
    for s, d in hx.items():
        with open(os.path.join(a.out, f"hx_{s}.jsonl"), "w") as f:
            for r in d.to_dict("records"):
                f.write(json.dumps({k: r[k] for k in ("id", "words", "label", "rationales", "targets")}) + "\n")
    if im:
        w(im_tr, "implicit_train.csv"); w(im_va, "implicit_val.csv"); w(im_te, "implicit_test.csv")
        w(im_probe, "implicit_explicit_probe.csv")
    if olid is not None:
        w(olid, "olid_test.csv")
    if hc is not None:
        w(hc, "hatecheck.csv")

    # ---- stats
    def lens(texts):
        l = texts.astype(str).str.split().str.len()
        return {"mean_words": float(l.mean()), "p95_words": float(l.quantile(.95)), "max_words": int(l.max())}
    stats["jigsaw"] = jinfo
    stats["explicit"] = {s: {"n": len(d), **{l: int(d[l].sum()) for l in LABELS},
                             "clean": int((d[LABELS].sum(axis=1) == 0).sum()), **lens(d["text"])}
                         for s, d in [("train", ex_tr), ("val", ex_va), ("valsub", ex_vs), ("test", ex_te)]}
    stats["hx"] = {}
    for s, d in hx.items():
        elig = d["rationales"].map(lambda v: len(v) > 0 and np.mean(np.asarray(v, float), 0).max() > 0)
        stats["hx"][s] = {"n": len(d), **{HX_CLASSES[k]: int(v) for k, v in d["label"].value_counts().items()},
                          "eligible_for_rationale_loss": int(elig.sum()), "eligible_share": float(elig.mean()),
                          "degenerate_random_control_posts": degenerate_random_control(d["rationales"]),
                          **lens(d["words"].map(" ".join))}
    stats["hx_load_info"] = hxinfo
    if im:
        stats["implicit"] = {s: {"n": len(d), "implicit_hate": int(d["label"].sum()), **lens(d["text"])}
                             for s, d in [("train", im_tr), ("val", im_va), ("test", im_te)]}
        stats["implicit"]["explicit_hate_probe"] = len(im_probe)
        stats["implicit_file_info"] = iminfo
    if olid is not None:
        stats["olid_test"] = {"n": len(olid), "offensive": int(olid["label"].sum())}
    if hc is not None:
        stats["hatecheck"] = {"n": len(hc), "hateful": int(hc["label"].sum()),
                              "functionalities": int(hc["functionality"].nunique())}
    all_train_words = list(ex_tr["text"].astype(str).str.split().str.len()) + list(hx["train"]["words"].map(len))
    if im:
        all_train_words += list(im_tr["text"].astype(str).str.split().str.len())
    stats["max_words_W"] = choose_W(all_train_words)
    stats["training_sizes"] = {"explicit": len(ex_tr), "hx": len(hx["train"]), "implicit": len(im_tr) if im else None}
    stats["ihc_status"] = ihc_status

    write_json(stats, os.path.join(a.out, "data_stats.json"))
    write_json(report, os.path.join(a.out, "dedup_report.json"))
    write_json({"explicit": "Jigsaw: 6 binary toxicity labels (Wikipedia talk pages)",
                "implicit": "IHC stage-1: implicit hate (1) vs not hate (0), Twitter",
                "hx": "HateXplain: hatespeech / offensive / normal (Twitter + Gab); toxic = hate or offensive",
                "olid": "OLID level A: offensive (1) vs not (0), Twitter",
                "hatecheck": "HateCheck: hateful (1) vs non-hateful (0), synthetic templates"},
               os.path.join(a.out, "label_semantics.json"))
    hashes = {f: md5_file(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))
              if f.endswith((".csv", ".jsonl"))}
    write_json(hashes, os.path.join(a.out, "split_hashes.json"))
    return {"stats": stats, "report": report}
