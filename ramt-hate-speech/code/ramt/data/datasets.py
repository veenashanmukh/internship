"""Tokenisation, word-level truncation (7.1), rationale targets and batching."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ramt.data.rationales import aggregate, align_to_subwords, random_matched

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HX_CLASSES = ["hatespeech", "offensive", "normal"]
EVAL_ONLY = ("olid", "hatecheck")


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    kw = {"add_prefix_space": True} if "roberta" in name.lower() else {}
    return AutoTokenizer.from_pretrained(name, use_fast=True, **kw)


def split_path(splits_dir: str, task: str, split: str) -> str:
    if task == "hx":
        return os.path.join(splits_dir, f"hx_{split}.jsonl")
    if task == "olid":
        return os.path.join(splits_dir, "olid_test.csv")
    if task == "hatecheck":
        return os.path.join(splits_dir, "hatecheck.csv")
    return os.path.join(splits_dir, f"{task}_{split}.csv")


def available(splits_dir: str, task: str, split: str) -> bool:
    return os.path.exists(split_path(splits_dir, task, split))


def _encode(tok, words_list: List[List[str]], max_subwords: int):
    safe = [w if len(w) else ["."] for w in words_list]
    enc = tok(safe, is_split_into_words=True, truncation=True,
              max_length=max_subwords if max_subwords > 0 else 512)
    out = []
    for i in range(len(safe)):
        out.append((enc["input_ids"][i], enc.word_ids(i)))
    return out


def load_items(splits_dir: str, task: str, split: str, tok, max_words: int = 0, max_subwords: int = 0,
               aggregation: str = "soft", rationale_target: str = "human",
               data_frac: float = 1.0) -> List[dict]:
    """Return one dict per example. For HateXplain each item also carries
    q (sub-word rationale target), elig (rationale-loss eligibility) and the
    word-level gold rationales used for evaluation (majority and union)."""
    path = split_path(splits_dir, task, split)
    if task == "hx":
        rows = [json.loads(l) for l in open(path)]
        words = [r["words"] for r in rows]
        ys = [np.int64(r["label"]) for r in rows]
        ids = [r["id"] for r in rows]
    else:
        df = pd.read_csv(path)
        words = [str(t).split() for t in df["text"]]
        if task == "explicit":
            ys = list(df[LABELS].values.astype(np.float32))
        elif task == "implicit":
            ys = list(df[["label"]].values.astype(np.float32))
        else:
            ys = list(df["label"].values.astype(np.float32))
        ids = [str(i) for i in (df["id"] if "id" in df.columns else df.get("src_id", pd.Series(range(len(df)))))]
        rows = None
    if split == "train" and data_frac < 1.0:
        keep = np.random.RandomState(0).permutation(len(words))[:max(8, int(len(words) * data_frac))]
        keep.sort()
        words = [words[i] for i in keep]; ys = [ys[i] for i in keep]; ids = [ids[i] for i in keep]
        rows = [rows[i] for i in keep] if rows is not None else None

    full_words = words
    if max_words > 0:
        words = [w[:max_words] for w in words]
    enc = _encode(tok, words, max_subwords)

    items = []
    for i, (input_ids, wid) in enumerate(enc):
        it = {"ids": input_ids, "wid": wid, "y": ys[i], "words": words[i], "post_id": ids[i],
              "n_words_full": len(full_words[i])}
        if task == "hx":
            r = rows[i]
            n = len(full_words[i])
            vecs = r["rationales"]
            rho = aggregate(vecs, n, aggregation)
            if rationale_target == "random_matched":
                rho = random_matched(rho, r["id"])
            q = align_to_subwords(rho, wid)
            it["q"] = q
            it["elig"] = bool(int(r["label"]) in (0, 1) and len(vecs) > 0 and q.sum() > 0)
            it["gold_majority"] = aggregate(vecs, n, "majority")[:len(words[i])]
            it["gold_union"] = aggregate(vecs, n, "union")[:len(words[i])]
            it["targets"] = r.get("targets", [])
        items.append(it)
    return items


def collate(items: List[dict], pad_id: int) -> Dict[str, torch.Tensor]:
    L = max(len(it["ids"]) for it in items)
    B = len(items)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    content = torch.zeros((B, L), dtype=torch.bool)
    q = torch.zeros((B, L), dtype=torch.float32)
    elig = torch.zeros(B, dtype=torch.bool)
    for i, it in enumerate(items):
        n = len(it["ids"])
        ids[i, :n] = torch.tensor(it["ids"])
        mask[i, :n] = 1
        content[i, :n] = torch.tensor([w is not None for w in it["wid"]])
        if "q" in it:
            q[i, :n] = torch.from_numpy(np.asarray(it["q"], dtype=np.float32))
            elig[i] = bool(it["elig"])
    y = torch.from_numpy(np.stack([np.asarray(it["y"]) for it in items]))
    return {"input_ids": ids, "attention_mask": mask, "content_mask": content, "q": q, "elig": elig, "y": y}


class TaskStream:
    """Batches drawn without replacement, reshuffled after each full pass (Section 2).
    With length_bucketing, each pass is cut into chunks of 50 batches sorted by length
    (padding efficiency only; every example still appears exactly once per pass)."""

    def __init__(self, items: List[dict], batch_size: int, rng: np.random.RandomState,
                 length_bucketing: bool = True):
        self.items, self.bs, self.rng, self.bucket = items, batch_size, rng, length_bucketing
        self.queue: List[List[int]] = []
        self.passes = 0
        self.examples_seen = 0

    def _refill(self):
        idx = self.rng.permutation(len(self.items))
        batches = []
        if self.bucket:
            chunk = self.bs * 50
            for s in range(0, len(idx), chunk):
                c = sorted(idx[s:s + chunk], key=lambda i: len(self.items[i]["ids"]))
                batches += [list(c[j:j + self.bs]) for j in range(0, len(c), self.bs)]
            order = self.rng.permutation(len(batches))
            batches = [batches[k] for k in order]
        else:
            batches = [list(idx[j:j + self.bs]) for j in range(0, len(idx), self.bs)]
        self.queue = batches
        self.passes += 1

    def next(self) -> List[int]:
        if not self.queue:
            self._refill()
        b = self.queue.pop(0)
        self.examples_seen += len(b)
        return b
