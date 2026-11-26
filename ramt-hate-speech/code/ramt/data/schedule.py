"""Round-based task schedule with exposure matching (D8, amendments A1, A2, A6).

Quotas are always derived from the FULL three-task training sizes, so a run that
contains only some tasks keeps exactly the same per-task quota per round.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

ALL_TASKS = ("explicit", "implicit", "hx")


def sampling_probs(full_sizes: Dict[str, int], temperature: float = 2.0) -> Dict[str, float]:
    w = {t: float(full_sizes[t]) ** (1.0 / temperature) for t in ALL_TASKS if full_sizes.get(t)}
    z = sum(w.values())
    return {t: v / z for t, v in w.items()}


def round_quotas(full_sizes: Dict[str, int], round_len_full: int = 1000, temperature: float = 2.0) -> Dict[str, int]:
    """q_t = round(R * p_t), adjusted by largest remainder so the quotas sum to R."""
    p = sampling_probs(full_sizes, temperature)
    raw = {t: round_len_full * v for t, v in p.items()}
    q = {t: int(np.floor(v)) for t, v in raw.items()}
    short = round_len_full - sum(q.values())
    for t in sorted(raw, key=lambda t: raw[t] - q[t], reverse=True)[:short]:
        q[t] += 1
    return q


def round_sequence(tasks: List[str], quotas: Dict[str, int], rng: np.random.RandomState) -> List[str]:
    """One round: exactly quotas[t] batches of every included task, in shuffled order."""
    seq = [t for t in tasks for _ in range(quotas[t])]
    rng.shuffle(seq)
    return seq


def run_length(tasks: List[str], quotas: Dict[str, int], rounds: int = 6) -> int:
    return rounds * sum(quotas[t] for t in tasks)
