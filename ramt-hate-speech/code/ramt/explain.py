"""Post-hoc explanations: Integrated Gradients and LIME, both reduced to one score per
word (Section 9). The explained quantity is the head's toxicity score:
explicit p(toxic), implicit p(implicit), HateXplain 1 - p(normal)."""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch

from ramt.data.datasets import _encode, collate
from ramt.data.rationales import subword_to_word
from ramt.evaluate import to_probs, toxicity_score


def _score_from_logits(head: str, logits: torch.Tensor) -> torch.Tensor:
    if head == "hx":
        return 1.0 - torch.softmax(logits.float(), -1)[:, 2]
    return torch.sigmoid(logits.float())[:, 0]


def integrated_gradients(model, tok, words: List[str], head: str, device: str, W: int, L: int,
                         steps: int = 50, chunk: int = 25, amp: bool = False):
    """Returns (word_scores, completeness_gap, seconds). Baseline: pad-token embedding at
    content positions, special tokens kept. Riemann midpoint rule; fp32 unless amp (fp16
    autocast on CUDA, used for the R7 timing protocol only)."""
    t0 = time.time()
    (ids, wid), = _encode(tok, [words[:W] if W else words], L)
    b = collate([{"ids": ids, "wid": wid, "y": np.zeros(1, np.float32)}], tok.pad_token_id)
    input_ids, am, cm = (b[k].to(device) for k in ("input_ids", "attention_mask", "content_mask"))
    emb = model.enc.get_input_embeddings()
    model.eval()
    with torch.no_grad():
        x = emb(input_ids).float()
        x0 = x.clone()
        pad_vec = emb(torch.tensor([[tok.pad_token_id]], device=device)).float()[0, 0]
        x0[0, cm[0]] = pad_vec
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    grads = torch.zeros_like(x)
    for s in range(0, steps, chunk):
        a = alphas[s:s + chunk].view(-1, 1, 1)
        xi = (x0 + a * (x - x0)).requires_grad_(True)
        n = xi.shape[0]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
            logits, _ = model(input_ids.expand(n, -1), am.expand(n, -1), cm.expand(n, -1), head, inputs_embeds=xi)
        out = _score_from_logits(head, logits).sum()
        g, = torch.autograd.grad(out, xi)
        grads += g.sum(0, keepdim=True)
    attr = ((x - x0) * grads / steps).sum(-1)[0]
    with torch.no_grad():
        fx = _score_from_logits(head, model(input_ids, am, cm, head, inputs_embeds=x)[0])[0]
        f0 = _score_from_logits(head, model(input_ids, am, cm, head, inputs_embeds=x0)[0])[0]
    diff = float(fx - f0)
    gap = abs(float(attr.sum()) - diff) / max(abs(diff), 1e-8)
    ws = subword_to_word(attr.detach().cpu().numpy(), wid, len(words[:W] if W else words))
    return ws, gap, time.time() - t0


def lime_scores(model, tok, words: List[str], head: str, device: str, W: int, L: int,
                num_samples: int = 1000, seed: int = 0, batch_size: int = 128, fp16: bool = True):
    """Word-level LIME (bow=False so features are word positions). Returns (scores, seconds)."""
    from lime.lime_text import LimeTextExplainer
    from ramt.evaluate import predict
    t0 = time.time()
    words = words[:W] if W else words

    def clf(texts):
        wl = [t.split(" ") if t else [] for t in texts]
        wl = [[w for w in x if w != "" and w != MASK] for x in wl]   # removal = deletion (as in AOPC)
        enc = _encode(tok, wl, L)
        items = [{"ids": i, "wid": w, "y": np.zeros(1, np.float32)} for i, w in enc]
        lg = predict(model, items, [head], device, tok.pad_token_id, batch_size, fp16)[head]["logits"]
        s = toxicity_score(head, to_probs(head, lg))
        return np.stack([1 - s, s], 1)

    MASK = "\u2063RAMTMASK\u2063"                      # never a real word; dropped before encoding
    ex = LimeTextExplainer(class_names=["non-toxic", "toxic"], bow=False, split_expression=r" ",
                           mask_string=MASK, random_state=seed)
    exp = ex.explain_instance(" ".join(words), clf, num_features=len(words), num_samples=num_samples, labels=(1,))
    scores = np.zeros(len(words), dtype=np.float32)
    # bow=False: feature i is word position i (verified against lime.IndexedString, see log)
    assert exp.domain_mapper.indexed_string.num_words() == len(words)
    for feat, w in exp.as_map()[1]:
        scores[int(feat)] = w
    return scores, time.time() - t0
