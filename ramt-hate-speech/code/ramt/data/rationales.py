"""Rationale aggregation, sub-word alignment and the random-matched control (Sections 3, 5).

Notation follows Section 3: rho = word-level target, q = sub-word target before
normalisation. Normalisation to q~ happens inside the loss.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ramt.utils import stable_hash

RANDOM_CONTROL_SEED = 0          # Section 5: permutation fixed per post with seed 0


def aggregate(vectors: Sequence[Sequence[int]], n_words: int, method: str = "soft") -> np.ndarray:
    """Step 1. vectors = rationale vectors of the annotators who supplied one."""
    if not vectors:
        return np.zeros(n_words, dtype=np.float32)
    m = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    if method == "soft":
        return m
    if method == "majority":
        return (m >= 0.5).astype(np.float32)
    if method == "union":
        return (m > 0).astype(np.float32)
    if method == "intersection":
        return (m >= 1.0 - 1e-6).astype(np.float32)
    raise ValueError(method)


def random_matched(rho: np.ndarray, post_id: str) -> np.ndarray:
    """Section 5: the post's own rho values permuted to random word positions,
    fixed per post (seed 0 combined with a stable hash of the post id)."""
    rng = np.random.RandomState((RANDOM_CONTROL_SEED + stable_hash(post_id)) % (2 ** 32))
    return rho[rng.permutation(len(rho))]


def align_to_subwords(rho: np.ndarray, word_ids: List[Optional[int]]) -> np.ndarray:
    """Step 2: each word's mass split evenly over its sub-word pieces; special and
    padding positions (word id None) get 0. Words cut by truncation simply vanish."""
    pieces = {}
    for w in word_ids:
        if w is not None:
            pieces[w] = pieces.get(w, 0) + 1
    q = np.zeros(len(word_ids), dtype=np.float32)
    for j, w in enumerate(word_ids):
        if w is not None and w < len(rho):
            q[j] = rho[w] / pieces[w]
    return q


def subword_to_word(scores: np.ndarray, word_ids: List[Optional[int]], n_words: int) -> np.ndarray:
    """Step 5: e_i = sum of sub-word scores of word i (words lost to truncation get 0)."""
    e = np.zeros(n_words, dtype=np.float32)
    for s, w in zip(scores, word_ids):
        if w is not None and w < n_words:
            e[w] += s
    return e
