"""One training run: 6 rounds of fixed per-task quotas, best-of-6 checkpoint, then
validation calibration and (main runs only) test evaluation.

Writes to <out_root>/<run_name>/ :
  config.json env.json train_log.csv exposure.json thresholds.json temperatures.json
  val_*.csv test_<set>__<head>.csv attention_hx_test.npz faithfulness_*.npz
  timing.json metrics.json (written last = run complete) [best.pt if save_ckpt]
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from ramt.config import Config
from ramt.data.datasets import LABELS, HX_CLASSES, available, collate, load_items, load_tokenizer, TaskStream
from ramt.data.schedule import round_quotas, round_sequence, run_length
from ramt.evaluate import (aopc_faithfulness, bootstrap_threshold, explicit_metrics, fit_temperature, hx_metrics,
                           implicit_metrics, plausibility, predict, to_probs, toxicity_score, val_score, _safe)
from ramt.losses import pos_weight_from_labels, rationale_loss, task_loss
from ramt.model import RAMT
from ramt.utils import capture_env, git_commit, seed_everything, write_json

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SETS = ("explicit", "implicit", "hx", "olid", "hatecheck")


class IHCBlocked(RuntimeError):
    pass


def _binary_labels(set_name: str, items: List[dict]) -> np.ndarray:
    Y = np.stack([np.asarray(it["y"]) for it in items])
    if set_name == "explicit":
        return (Y.reshape(len(Y), -1).max(1) > 0).astype(int)
    if set_name == "hx":
        return (Y != 2).astype(int)
    return Y.reshape(len(Y), -1)[:, 0].astype(int)


def _resolve_lengths(cfg: Config):
    stats_path = os.path.join(cfg.splits_dir, "data_stats.json")
    W = cfg.max_words or (json.load(open(stats_path)).get("max_words_W", 0) if os.path.exists(stats_path) else 0)
    L = cfg.max_subwords
    if not L:
        b0 = os.path.join(cfg.splits_dir, "subword_limit.json")
        L = json.load(open(b0))["max_subwords_L"] if os.path.exists(b0) else 256
    return int(W), int(L)


def _full_sizes(cfg: Config) -> Dict[str, int]:
    if cfg.full_sizes:
        return dict(cfg.full_sizes)
    st = json.load(open(os.path.join(cfg.splits_dir, "data_stats.json")))["training_sizes"]
    if not st.get("implicit"):
        if cfg.group in ("smoke", "bench") and "implicit" not in cfg.tasks:
            return {t: v for t, v in st.items() if v}       # smoke/bench only; never main or pilot
        raise IHCBlocked("Quotas need the IHC training size, but IHC is not prepared (D1: permission "
                         "pending). Pass full_sizes explicitly only for smoke tests / benchmarks.")
    return st


def _make_score_fn(model, tok, head: str, W: int, L: int, device: str, cfg: Config):
    from ramt.data.datasets import _encode

    @torch.no_grad()
    def fn(word_lists):
        enc = _encode(tok, [w[:W] if W else w for w in word_lists], L)
        items = [{"ids": ids, "wid": wid, "y": np.zeros(1, np.float32)} for ids, wid in enc]
        logits = predict(model, items, [head], device, tok.pad_token_id, cfg.eval_batch_size, cfg.fp16)[head]["logits"]
        return toxicity_score(head, to_probs(head, logits))
    return fn


def run(cfg: Config):
    cfg.validate()
    t_run0 = time.time()
    out = os.path.join(cfg.out_root, cfg.run_name)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print("already complete:", out)
        return
    os.makedirs(out, exist_ok=True)
    cfg.dump(os.path.join(out, "config.json"))
    env = capture_env()
    env["git_commit"] = git_commit(REPO_DIR)
    write_json(env, os.path.join(out, "env.json"))
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.fp16 and device == "cuda"

    hp = os.path.join(cfg.splits_dir, "split_hashes.json")
    if os.path.exists(hp):                                   # same bytes as the checked splits (G1)
        from ramt.utils import md5_file
        bad = [f for f, h in json.load(open(hp)).items()
               if os.path.exists(os.path.join(cfg.splits_dir, f)) and md5_file(os.path.join(cfg.splits_dir, f)) != h]
        if bad:
            raise RuntimeError(f"split files changed since preparation: {bad}")
        env["split_hashes_verified"] = True
        write_json(env, os.path.join(out, "env.json"))

    if "implicit" in cfg.tasks and not available(cfg.splits_dir, "implicit", "train"):
        raise IHCBlocked("implicit task requested but IHC splits are absent (D1: written permission pending)")

    tok = load_tokenizer(cfg.encoder)
    pad = tok.pad_token_id
    W, L = _resolve_lengths(cfg)
    load = lambda task, split, frac=1.0: load_items(cfg.splits_dir, task, split, tok, W, L, cfg.aggregation,
                                                    cfg.rationale_target, frac)
    train = {t: load(t, "train", cfg.data_frac) for t in cfg.tasks}
    val_periodic = {t: load(t, "valsub" if t == "explicit" else "val") for t in cfg.tasks}
    val_full = {t: (load(t, "val") if t == "explicit" else val_periodic[t]) for t in cfg.tasks}

    quotas = round_quotas(_full_sizes(cfg), cfg.round_len_full, cfg.sampling_temperature)
    total_steps = run_length(cfg.tasks, quotas, cfg.rounds)
    model = RAMT(cfg.encoder, cfg.tasks, cfg.pooling, cfg.dropout).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(round(cfg.warmup_frac * total_steps)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pw = {t: pos_weight_from_labels(np.stack([it["y"] for it in train[t]])).to(device)
          for t in cfg.tasks if t != "hx"}

    rng = np.random.RandomState(cfg.seed)
    streams = {t: TaskStream(train[t], cfg.batch_size, rng, cfg.length_bucketing) for t in cfg.tasks}
    batches_done = {t: 0 for t in cfg.tasks}
    log, best, best_round = [], -math.inf, 0
    tmp_ckpt = os.path.join(out, "_best_tmp.pt")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    round_secs, val_secs = [], []
    for r in range(1, cfg.rounds + 1):
        model.train()
        t0 = time.time()
        seq = round_sequence(cfg.tasks, quotas, rng)
        acc = {t: [0.0, 0] for t in cfg.tasks}
        rat_sum, rat_n, elig_counts = 0.0, 0, []
        for t in seq:
            idx = streams[t].next()
            b = collate([train[t][i] for i in idx], pad)
            ids, am, cm, y = (b[k].to(device) for k in ("input_ids", "attention_mask", "content_mask", "y"))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, alpha = model(ids, am, cm, task=t)
            loss = task_loss(t, logits, y, pw.get(t))
            acc[t][0] += float(loss.detach()); acc[t][1] += 1
            if t == "hx" and cfg.mu > 0:
                rl = rationale_loss(alpha, b["q"].to(device), b["elig"].to(device))
                elig_counts.append(int(b["elig"].sum()))
                loss = loss + cfg.mu * rl
                rat_sum += float(rl.detach()); rat_n += 1
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at round {r}, step {step}, task {t}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            batches_done[t] += 1; step += 1
        round_secs.append(time.time() - t0)

        # per-round validation (A2): threshold-free task scores, composite = mean
        t_v = time.time()
        parts = {}
        for t in cfg.tasks:
            pr = predict(model, val_periodic[t], [t], device, pad, cfg.eval_batch_size, cfg.fp16)[t]["logits"]
            Y = np.stack([np.asarray(it["y"]) for it in val_periodic[t]])
            parts[t] = val_score(t, Y, to_probs(t, pr))
        comp = float(np.mean(list(parts.values())))
        val_secs.append(time.time() - t_v)
        row = {"round": r, "step": step, "val_composite": comp, **{f"val_{k}": v for k, v in parts.items()},
               **{f"train_loss_{t}": acc[t][0] / max(acc[t][1], 1) for t in cfg.tasks},
               "train_loss_rationale": rat_sum / max(rat_n, 1) if rat_n else None,
               "eligible_per_hx_batch_mean": float(np.mean(elig_counts)) if elig_counts else None,
               "hx_batches_with_zero_eligible": int(sum(c == 0 for c in elig_counts)) if elig_counts else None,
               "round_sec": round_secs[-1], "lr": sched.get_last_lr()[0]}
        log.append(row)
        pd.DataFrame(log).to_csv(os.path.join(out, "train_log.csv"), index=False)
        print(json.dumps(row), flush=True)
        if comp > best:
            best, best_round = comp, r
            torch.save({k: (v.detach().half() if v.is_floating_point() else v.detach())
                        for k, v in model.state_dict().items()}, tmp_ckpt)

    exposure = {t: {"batches": batches_done[t], "examples": streams[t].examples_seen,
                    "passes": streams[t].examples_seen / max(len(train[t]), 1), "quota_per_round": quotas[t]}
                for t in cfg.tasks}
    write_json({"quotas_full_model": quotas, "total_steps": total_steps, "steps_run": step,
                "best_round": best_round, "per_task": exposure}, os.path.join(out, "exposure.json"))

    sd = torch.load(tmp_ckpt, map_location=device)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, strict=True)

    # ---- calibration on full validation (Section 8)
    t_cal0 = time.time()
    thr, temps, M = {}, {}, {"run": cfg.run_name, "best_round": best_round, "best_val_composite": best}
    for t in cfg.tasks:
        want_a = (t == "hx" and cfg.pooling != "cls")
        pv = predict(model, val_full[t], [t], device, pad, cfg.eval_batch_size, cfg.fp16, want_attention=want_a)[t]
        pr = pv["logits"]
        if want_a:   # validation plausibility: tie-break for mu (8.3); hx head, majority gold, as on test
            M["plausibility_hx_val"] = plausibility(pv["attn_words"], [it["gold_majority"] for it in val_full[t]])
        Y = np.stack([np.asarray(it["y"]) for it in val_full[t]])
        P = to_probs(t, pr)
        if t == "explicit":
            thr["explicit"] = {l: bootstrap_threshold(Y[:, k], P[:, k], cfg.threshold_bootstrap) for k, l in enumerate(LABELS)}
            temps["explicit"] = {l: fit_temperature(pr[:, k], Y[:, k], False) for k, l in enumerate(LABELS)}
        elif t == "implicit":
            thr["implicit"] = bootstrap_threshold(Y[:, 0], P[:, 0], cfg.threshold_bootstrap)
            temps["implicit"] = fit_temperature(pr[:, 0], Y[:, 0], False)
        else:
            thr["hx_binary"] = bootstrap_threshold((Y != 2).astype(int), 1 - P[:, 2], cfg.threshold_bootstrap)
            temps["hx"] = fit_temperature(pr, Y, True)
        cols = {"post_id": [it["post_id"] for it in val_full[t]]}
        for j in range(pr.shape[1]):
            cols[f"logit_{j}"] = pr[:, j]
        pd.DataFrame(cols).to_csv(os.path.join(out, f"val_{t}.csv"), index=False)
    write_json(thr, os.path.join(out, "thresholds.json"))
    write_json(temps, os.path.join(out, "temperatures.json"))
    head_thr = {"explicit": thr.get("explicit", {}).get("toxic"), "implicit": thr.get("implicit"),
                "hx": thr.get("hx_binary")}

    calib_sec = time.time() - t_cal0
    test_sec = faith_sec = 0.0
    # ---- test evaluation (main runs only; A7)
    if cfg.evaluate_test:
        t_te0 = time.time()
        cross = {}
        for s in TEST_SETS:
            split = "test"
            if not available(cfg.splits_dir, s, split):
                continue
            items = load(s, split)
            pred = predict(model, items, cfg.tasks, device, pad, cfg.eval_batch_size, cfg.fp16,
                           want_attention=(s == "hx" and cfg.pooling != "cls"))
            yb = _binary_labels(s, items)
            cross[s] = {}
            for h in cfg.tasks:
                lg = pred[h]["logits"]
                P = to_probs(h, lg)
                cols = {"post_id": [it["post_id"] for it in items], "y_binary": yb}
                for j in range(lg.shape[1]):
                    cols[f"logit_{j}"] = lg[:, j]
                pd.DataFrame(cols).to_csv(os.path.join(out, f"test_{s}__{h}.csv"), index=False)
                sc = toxicity_score(h, P)
                cross[s][h] = {"roc_auc": _safe(roc_auc_score, yb, sc),
                               "f1_at_source_threshold": float(f1_score(yb, (sc >= head_thr[h]).astype(int),
                                                                        zero_division=0))}
                if s == h:                                    # in-domain metrics
                    Y = np.stack([np.asarray(it["y"]) for it in items])
                    M[h] = (explicit_metrics(Y, P, thr["explicit"]) if h == "explicit" else
                            implicit_metrics(Y, P, thr["implicit"]) if h == "implicit" else
                            hx_metrics(Y.astype(int), P, thr["hx_binary"]))
            # combined pre-registered rule (Section 6A)
            scores = np.max(np.stack([toxicity_score(h, to_probs(h, pred[h]["logits"])) for h in cfg.tasks]), 0)
            flags = np.any(np.stack([toxicity_score(h, to_probs(h, pred[h]["logits"])) >= head_thr[h]
                                     for h in cfg.tasks]), 0).astype(int)
            cross[s]["combined_rule"] = {"roc_auc": _safe(roc_auc_score, yb, scores),
                                         "f1": float(f1_score(yb, flags, zero_division=0))}

            if s == "hx" and cfg.pooling != "cls":
                gold_m = [it["gold_majority"] for it in items]
                gold_u = [it["gold_union"] for it in items]
                attn = {h: pred[h]["attn_words"] for h in cfg.tasks}
                np.savez_compressed(os.path.join(out, "attention_hx_test.npz"),
                                    post_id=np.array([it["post_id"] for it in items]),
                                    **{h: np.array(attn[h], dtype=object) for h in cfg.tasks})
                M["plausibility_hx_test"] = {h: {"gold_majority": plausibility(attn[h], gold_m),
                                                 "gold_union_hatexplain_comparable": plausibility(attn[h], gold_u)}
                                             for h in cfg.tasks}
        M["cross_dataset"] = cross
        test_sec = time.time() - t_te0

        if cfg.faithfulness_in_run and cfg.pooling != "cls":
            t_f0 = time.time()
            M["faithfulness_attention"] = _faithfulness(model, tok, cfg, W, L, device, out, load, head_thr)
            faith_sec = time.time() - t_f0

    n_par = sum(p.numel() for p in model.parameters())
    timing = {"params_total": n_par, "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "round_sec": round_secs, "train_steps": step,
              "train_steps_per_sec": step / max(sum(round_secs), 1e-9), "device": device,
              "val_sec_per_round": val_secs, "calibration_sec": calib_sec, "test_eval_sec": test_sec,
              "faithfulness_sec": faith_sec, "wall_sec_total": time.time() - t_run0,
              "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None, "fp16": use_amp,
              "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None}
    write_json(timing, os.path.join(out, "timing.json"))
    if cfg.save_ckpt:
        os.replace(tmp_ckpt, os.path.join(out, "best.pt"))
    else:
        os.remove(tmp_ckpt)
    write_json(M, os.path.join(out, "metrics.json"))
    print("DONE", out, flush=True)
    return M


def _faithfulness(model, tok, cfg, W, L, device, out, load, head_thr):
    """Attention AOPC on posts predicted positive by each head on its own test set
    (and on HateXplain for the HateXplain head); fixed sample capped per set (IMPL)."""
    res = {}
    for h in cfg.tasks:
        s = h
        if not available(cfg.splits_dir, s, "test"):
            continue
        items = load(s, "test")
        pred = predict(model, items, [h], device, tok.pad_token_id, cfg.eval_batch_size, cfg.fp16, want_attention=True)[h]
        sc = toxicity_score(h, to_probs(h, pred["logits"]))
        pos = np.where(sc >= head_thr[h])[0]
        rng = np.random.RandomState(0)
        if len(pos) > cfg.faithfulness_max_posts:
            pos = np.sort(rng.choice(pos, cfg.faithfulness_max_posts, replace=False))
        if len(pos) == 0:
            continue
        words = [items[i]["words"] for i in pos]
        ws = [pred["attn_words"][i] for i in pos]
        f = aopc_faithfulness(_make_score_fn(model, tok, h, W, L, device, cfg), words, ws,
                              random_draws=cfg.random_draws)
        pp = f.pop("per_post")
        np.savez_compressed(os.path.join(out, f"faithfulness_{h}.npz"), index=pos, **pp)
        res[h] = {**f, "n_posts": int(len(pos))}
    return res
