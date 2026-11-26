"""Shared encoder + attention pooling over content tokens + one linear head per task
(Section 3 Step 3, Section 14 "Model")."""
from __future__ import annotations

import torch
import torch.nn as nn

N_OUT = {"explicit": 6, "implicit": 1, "hx": 3}


class ContentAttentionPool(nn.Module):
    """s_j = v^T tanh(W h_j + b); softmax over content tokens only (no padding, no
    special tokens). Scores and softmax are computed in fp32 (Section 3.6)."""

    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.W = nn.Linear(hidden, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor, content_mask: torch.Tensor):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self._forward(h.float(), content_mask, h.dtype)

    def _forward(self, h: torch.Tensor, content_mask: torch.Tensor, out_dtype):
        s = self.v(torch.tanh(self.W(h))).squeeze(-1)                          # fp32
        s = s.masked_fill(~content_mask, float("-inf"))
        empty = ~content_mask.any(dim=1)                                        # guard: no content tokens
        if empty.any():
            s = torch.where(empty.unsqueeze(1), torch.zeros_like(s), s)
        alpha = torch.softmax(s, dim=-1)
        alpha = alpha * content_mask.float()                                    # exact zeros off-content
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp(min=1e-12)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return z.to(out_dtype), alpha


class RAMT(nn.Module):
    def __init__(self, encoder_name: str, tasks, pooling: str = "task", dropout: float = 0.1,
                 encoder_module: nn.Module | None = None):
        super().__init__()
        from transformers import AutoModel
        self.enc = encoder_module if encoder_module is not None else AutoModel.from_pretrained(encoder_name)
        # Some checkpoints (e.g. microsoft/deberta-v3-base) are stored in fp16 and recent transformers
        # keeps the stored dtype. AMP needs fp32 master weights; fp16 is applied only by autocast.
        self.enc.float()
        H = self.enc.config.hidden_size
        self.tasks, self.pooling = list(tasks), pooling
        if pooling == "task":
            self.pools = nn.ModuleDict({t: ContentAttentionPool(H) for t in self.tasks})
        elif pooling == "shared":
            self.shared_pool = ContentAttentionPool(H)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(H, N_OUT[t]) for t in self.tasks})

    def encode(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        return self.enc(input_ids=None if inputs_embeds is not None else input_ids,
                        inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

    def pool(self, task, h, content_mask):
        if self.pooling == "task":
            return self.pools[task](h, content_mask)
        if self.pooling == "shared":
            return self.shared_pool(h, content_mask)
        return h[:, 0], None                                                   # [CLS] baseline (D13)

    def forward(self, input_ids, attention_mask, content_mask, task, inputs_embeds=None):
        h = self.encode(input_ids, attention_mask, inputs_embeds)
        z, alpha = self.pool(task, h, content_mask)
        return self.heads[task](self.drop(z)), alpha

    def forward_all(self, input_ids, attention_mask, content_mask):
        """One encoder pass, every head (used for cross-dataset scoring)."""
        h = self.encode(input_ids, attention_mask)
        out = {}
        for t in self.tasks:
            z, alpha = self.pool(t, h, content_mask)
            out[t] = (self.heads[t](self.drop(z)), alpha)
        return out
