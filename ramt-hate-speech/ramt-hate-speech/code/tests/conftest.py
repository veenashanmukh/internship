"""Fixtures: a tiny local BERT encoder (no downloads) and synthetic raw datasets that
mirror the real file formats (Jigsaw CSVs, IHC stage-1 TSV, HateXplain JSON, OLID,
HateCheck), processed through the real data-preparation code."""
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOOD = "the a cat dog is nice great movie today weather lovely friend coffee book read park walk happy".split()
BAD = "idiot stupid kill hate trash moron die ugly scum dumb".split()
GRP = "immigrants they them those people group".split()
# multi-piece words exercise sub-word alignment
EXTRA = ["##ly", "##ness", "un", "##happy"]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "@user", "http"] + GOOD + BAD + GRP + EXTRA
    (d / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(d / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(str(d))
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=128)).save_pretrained(str(d))
    return str(d)


def _sent(tox, rnd):
    w = rnd.choices(GOOD, k=rnd.randint(4, 12))
    if tox:
        for _ in range(rnd.randint(1, 2)):
            w.insert(rnd.randrange(len(w) + 1), rnd.choice(BAD))
    if rnd.random() < 0.3:
        w.append("unhappy")                                  # two sub-word pieces
    return w


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    rnd = random.Random(0)
    d = tmp_path_factory.mktemp("raw")
    # Jigsaw
    (d / "jigsaw").mkdir()

    def jig(n, start):
        rows = []
        for i in range(n):
            t = rnd.random() < 0.3
            lab = [int(t)] + [int(t and rnd.random() < p) for p in [.1, .5, .1, .5, .1]]
            rows.append([f"id{start + i}", " ".join(_sent(t, rnd)) + f" {i % 7}"] + lab)
        return pd.DataFrame(rows, columns=["id", "comment_text"] + LABELS)
    jig(900, 0).to_csv(d / "jigsaw" / "train.csv", index=False)
    te = jig(400, 10000)
    te[["id", "comment_text"]].to_csv(d / "jigsaw" / "test.csv", index=False)
    tl = te[["id"] + LABELS].copy(); tl.loc[:40, LABELS] = -1
    tl.to_csv(d / "jigsaw" / "test_labels.csv", index=False)
    # IHC stage-1
    rows = []
    for i in range(700):
        c = rnd.choice(["not_hate", "not_hate", "implicit_hate", "explicit_hate"])
        w = rnd.choices(GOOD, k=6) + (rnd.choices(GRP, k=2) if c == "implicit_hate" else []) + \
            (["kill"] if c == "explicit_hate" else [])
        rnd.shuffle(w)
        rows.append([" ".join(w) + f" {i % 11}", c])
    pd.DataFrame(rows, columns=["post", "class"]).to_csv(d / "ihc.tsv", sep="\t", index=False)
    # HateXplain (2-3 annotators, rationales only from toxic labels, one length-mismatched vector)
    data, div = {}, {"train": [], "val": [], "test": []}
    for i in range(600):
        lab = rnd.choice(["hatespeech", "offensive", "normal"])
        w = _sent(lab != "normal", rnd) + [str(i % 5)]
        others = [rnd.choice(["hatespeech", "offensive", "normal"])]
        anns = [{"label": lab, "annotator_id": k, "target": ["African" if lab != "normal" else "None"]} for k in range(2)] + \
               [{"label": others[0], "annotator_id": 3, "target": ["Islam"]}]
        rats = []
        if lab != "normal":
            rats = [[int(x in BAD) for x in w] for _ in range(rnd.choice([2, 3]))]
            if i == 5:
                rats.append([1] * (len(w) + 2))               # length mismatch -> dropped
        pid = f"p{i}"
        data[pid] = {"post_id": pid, "annotators": anns, "rationales": rats, "post_tokens": w}
        div["train" if i < 420 else "val" if i < 510 else "test"].append(pid)
    data["undecided"] = {"post_id": "undecided", "rationales": [], "post_tokens": ["a"],
                         "annotators": [{"label": l, "annotator_id": k, "target": []}
                                        for k, l in enumerate(["normal", "offensive", "hatespeech"])]}
    (d / "hx").mkdir()
    json.dump(data, open(d / "hx" / "dataset.json", "w"))
    json.dump(div, open(d / "hx" / "post_id_divisions.json", "w"))
    # OLID official test format
    (d / "olid").mkdir()
    o = []
    for i in range(120):
        t = rnd.random() < .3
        o.append((50000 + i, " ".join(_sent(t, rnd)) + " @USER", "OFF" if t else "NOT"))
    pd.DataFrame([(a, b) for a, b, c in o], columns=["id", "tweet"]).to_csv(d / "olid" / "testset-levela.tsv", sep="\t", index=False)
    pd.DataFrame([(a, c) for a, b, c in o]).to_csv(d / "olid" / "labels-levela.csv", index=False, header=False)
    # HateCheck
    h = []
    for i in range(100):
        t = rnd.random() < .5
        h.append((f"f{i % 5}", i, " ".join(_sent(t, rnd)), "hateful" if t else "non-hateful", "women"))
    pd.DataFrame(h, columns=["functionality", "case_id", "test_case", "label_gold", "target_ident"]).to_csv(
        d / "hatecheck.csv", index=False)
    (d / "permission.txt").write_text("synthetic permission record for tests")
    return d


def prep_args(raw, out, permission=True):
    return SimpleNamespace(jigsaw_dir=str(raw / "jigsaw"), hx_dir=str(raw / "hx"), olid_dir=str(raw / "olid"),
                           hatecheck_file=str(raw / "hatecheck.csv"), ihc_file=str(raw / "ihc.tsv"),
                           ihc_permission_record=str(raw / "permission.txt") if permission else None,
                           out=str(out), n_clean_train=300, val_frac=0.10, valsub_n=50, seed=42)


@pytest.fixture(scope="session")
def splits(raw_dir, tmp_path_factory):
    from ramt.data.prep import prepare
    out = tmp_path_factory.mktemp("splits")
    prepare(prep_args(raw_dir, out, permission=True))
    return str(out)
