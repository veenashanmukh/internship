"""Task losses and the rationale-alignment loss (Sections 2, 3; D9, amendment A3)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def pos_weight_from_labels(Y: np.ndarray, clip: float = 10.0) -> torch.Tensor:
    """w_k = clip(sqrt(neg_k / pos_k), 1, clip) from training counts (Section 2)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    pos = Y.sum(0)
    neg = len(Y) - pos
    return torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1.0)), 1.0, clip), dtype=torch.float32)


def task_loss(task: str, logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None):
    if task == "hx":
        return F.cross_entropy(logits.float(), y.long())
    return F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pos_weight)


def rationale_loss(alpha: torch.Tensor | None, q: torch.Tensor, elig: torch.Tensor) -> torch.Tensor:
    """L_rat = (1/B) * sum over eligible posts of  -sum_j q~_j log(alpha_j + eps).

    Divided by the batch size B, not by the eligible count (D9), so an empty eligible
    set gives exactly 0 with zero gradient, and every eligible post weighs 1/B.
    Computed in fp32."""
    B = q.shape[0]
    if alpha is None:
        return torch.zeros((), device=q.device)
    mass = q.sum(-1)
    use = elig & (mass > 0)
    if not bool(use.any()):
        return alpha.sum() * 0.0                       # exact zero, keeps the graph valid
    qn = q[use] / mass[use].unsqueeze(-1)
    ce = -(qn * torch.log(alpha[use].float() + EPS)).sum(-1)
    return ce.sum() / B
