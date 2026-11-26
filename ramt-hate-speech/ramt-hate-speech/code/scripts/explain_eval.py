"""Explanation evaluation on saved checkpoints (Phase 5; hypotheses H2/H3 faithfulness legs, E).

GPU job (both T4s):
    python scripts/explain_eval.py --runs <trained runs dir> --splits <splits> --out /kaggle/working/explain --launch
CPU timing job (no GPU quota):
    python scripts/explain_eval.py --runs ... --splits ... --out /kaggle/working/explain_cpu --mode cpu_timing

Faithfulness units, for every M1/M2/M3 run that kept best.pt, on test AND validation of each task:
  fixed post set = posts whose GOLD label is positive (toxic / implicit hate / hate-or-offensive),
  at most 1,000, drawn with RandomState(0) from the split file. The set does not depend on the
  model, so every model and method is scored on identical posts (paired tests, R4).
  Per post: AOPC comprehensiveness and sufficiency (ERASER bins) for attention, for the random
  baseline (5 draws) and, for M1, for Integrated Gradients (50 steps, fp32); plausibility per
  post (token AUPRC vs majority gold) on HateXplain sets.
Timing units (R7 protocol): 500 fixed HateXplain test posts (+20 warm-up discarded), batch 1,
  M1 seeds 42/123/2024; prediction alone, prediction + attention word scores, IG (50 steps),
  LIME (1,000 samples); fp16 on GPU, fp32 on CPU; CUDA synchronised.
Units are skipped when their output exists, so an interrupted job resumes.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt.data.datasets import load_items, load_tokenizer  # noqa: E402
from ramt.evaluate import aopc_method, aopc_random, per_post_ap, predict  # noqa: E402
from ramt.explain import integrated_gradients, lime_scores  # noqa: E402
from ramt.model import RAMT  # noqa: E402
from ramt.train import _make_score_fn  # noqa: E402

MODELS = ("M1_RAMT", "M2_MTL_noRat", "M3_random_matched")
TASKS = ("hx", "explicit", "implicit")
TIMING_SEEDS = ("seed42", "seed123", "seed2024")
MAX_POSTS = 1000


def run_dirs(root):
    out = []
    for ck in sorted(glob.glob(os.path.join(root, "**", "best.pt"), recursive=True)):
        d = os.path.dirname(ck)
        parts = d.split(os.sep)
        if parts[-2] in MODELS:
            out.append((parts[-3], parts[-2], parts[-1], d))         # group, model, seed, dir
    return out


def limits(splits):
    W = int(json.load(open(os.path.join(splits, "data_stats.json")))["max_words_W"])
    p = os.path.join(splits, "subword_limit.json")
    L = int(json.load(open(p))["max_subwords_L"]) if os.path.exists(p) else 256
    return W, L


def load_model(d, device):
    cfg = json.load(open(os.path.join(d, "config.json")))
    m = RAMT(cfg["encoder"], cfg["tasks"], cfg.get("pooling", "task"), cfg.get("dropout", 0.1))
    sd = torch.load(os.path.join(d, "best.pt"), map_location="cpu")
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, strict=True)
    return m.to(device).eval(), cfg


def fixed_set(items, task):
    if task == "hx":
        pos = [i for i, it in enumerate(items) if int(it["y"]) in (0, 1)]
    else:
        pos = [i for i, it in enumerate(items) if float(np.asarray(it["y"]).ravel()[0]) == 1.0]
    pos = np.asarray(pos)
    if len(pos) > MAX_POSTS:
        pos = np.sort(np.random.RandomState(0).choice(pos, MAX_POSTS, replace=False))
    return pos


def faith_unit(args, group, model_name, seed, d, task, split, device):
    out = os.path.join(args.out, "faith", group, model_name, seed, f"{task}_{split}.npz")
    if os.path.exists(out):
        return "skip"
    model, cfg = load_model(d, device)
    tok = load_tokenizer(cfg["encoder"])
    W, L = limits(args.splits)
    items = load_items(args.splits, task, split, tok, W, L, cfg.get("aggregation", "soft"), "human")
    sel = fixed_set(items, task)
    its = [items[i] for i in sel]
    words = [it["words"] for it in its]
    fp16 = device == "cuda" and cfg.get("fp16", True)
    ns = SimpleNamespace(eval_batch_size=128, fp16=fp16)
    fn = _make_score_fn(model, tok, task, W, L, device, ns)
    attn = predict(model, its, [task], device, tok.pad_token_id, 128, fp16, want_attention=True)[task]["attn_words"]
    t0 = time.time()
    base = fn(words)
    rc, rs = aopc_random(fn, words, base)
    ac, as_, _ = aopc_method(fn, words, attn, base)
    res = {"post_id": np.array([it["post_id"] for it in its]), "index": sel, "base": base,
           "comp_attn": ac, "suff_attn": as_, "comp_rand": rc, "suff_rand": rs}
    if task == "hx":
        gold = [it["gold_majority"] for it in its]
        res["plaus_attn"] = per_post_ap(attn, gold)
    if model_name == "M1_RAMT":
        ig, gaps, secs = [], [], []
        for w in words:
            s, g, t = integrated_gradients(model, tok, w, task, device, W, L, steps=50)
            ig.append(s); gaps.append(g); secs.append(t)
        ic, is_, _ = aopc_method(fn, words, ig, base)
        res.update({"comp_ig": ic, "suff_ig": is_, "ig_gap": np.asarray(gaps), "ig_sec": np.asarray(secs)})
        if task == "hx":
            res["plaus_ig"] = per_post_ap(ig, gold)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", **res)
    os.replace(out + ".tmp.npz", out)
    return f"{len(sel)} posts, {time.time() - t0:.0f}s"


def timing_unit(args, d, seed, device, tag):
    out = os.path.join(args.out, "timing", f"{tag}_{seed}.json")
    if os.path.exists(out):
        return "skip"
    torch.set_num_threads(os.cpu_count() or 1)
    model, cfg = load_model(d, device)
    tok = load_tokenizer(cfg["encoder"])
    W, L = limits(args.splits)
    items = load_items(args.splits, "hx", "test", tok, W, L, cfg.get("aggregation", "soft"), "human")
    order = np.random.RandomState(0).permutation(len(items))[:args.timing_posts + 20]
    fp16 = device == "cuda"
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    T = {"pred": [], "pred_attn": [], "ig": [], "lime": [], "ig_gap": []}
    for i in order:
        it = items[i]
        sync(); t = time.perf_counter()
        predict(model, [it], ["hx"], device, tok.pad_token_id, 1, fp16, want_attention=False); sync()
        T["pred"].append(time.perf_counter() - t)
        sync(); t = time.perf_counter()
        predict(model, [it], ["hx"], device, tok.pad_token_id, 1, fp16, want_attention=True); sync()
        T["pred_attn"].append(time.perf_counter() - t)
        sync(); t = time.perf_counter()
        _, gap, _ = integrated_gradients(model, tok, it["words"], "hx", device, W, L, steps=50, amp=fp16); sync()
        T["ig"].append(time.perf_counter() - t); T["ig_gap"].append(gap)
        if not args.skip_lime:
            sync(); t = time.perf_counter()
            lime_scores(model, tok, it["words"], "hx", device, W, L, num_samples=1000, fp16=fp16); sync()
            T["lime"].append(time.perf_counter() - t)
    summ = {}
    for k, v in T.items():
        v = np.asarray(v[20:], dtype=float)
        if len(v) and k != "ig_gap":
            summ[k] = {"median_ms": 1000 * float(np.median(v)), "p95_ms": 1000 * float(np.percentile(v, 95)),
                       "n": int(len(v))}
    summ["device"] = device
    summ["gpu_name"] = torch.cuda.get_device_name(0) if device == "cuda" else None
    summ["cpu_threads"] = torch.get_num_threads()
    summ["raw_seconds"] = {k: v for k, v in T.items()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summ, open(out, "w"))
    return {k: summ[k]["median_ms"] for k in ("pred", "pred_attn", "ig", "lime") if k in summ}


def units(args):
    args.timing_seeds = getattr(args, "timing_seeds", None)
    U = []
    if args.mode == "gpu":
        for group, model_name, seed, d in run_dirs(args.runs):
            for split in ("test", "val"):
                for task in TASKS:
                    U.append(("faith", group, model_name, seed, d, task, split))
    for group, model_name, seed, d in run_dirs(args.runs):
        if group == "main" and model_name == "M1_RAMT" and seed in (args.timing_seeds or TIMING_SEEDS):
            U.append(("timing", seed, d))
    # heavy IG units first, spread over shards
    U.sort(key=lambda u: 0 if (u[0] == "faith" and u[2] == "M1_RAMT") else 1)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="gpu", choices=["gpu", "cpu_timing"])
    ap.add_argument("--launch", action="store_true", help="one worker per visible GPU")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--timing_posts", type=int, default=500)
    ap.add_argument("--skip_lime", action="store_true")
    ap.add_argument("--timing_seeds", nargs="*", default=None, help="e.g. seed42 (default: seed42 seed123 seed2024)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.launch and a.mode == "gpu":
        n = max(1, torch.cuda.device_count())
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--runs", a.runs, "--splits", a.splits,
                                   "--out", a.out, "--shard", str(g), "--nshards", str(n),
                                   "--timing_posts", str(a.timing_posts)] + (["--skip_lime"] if a.skip_lime else []),
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))) for g in range(n)]
        rc = [p.wait() for p in procs]
        print("workers exited with", rc)
        done = len(glob.glob(os.path.join(a.out, "faith", "**", "*.npz"), recursive=True))
        print(f"faithfulness units done: {done}; timing files: "
              f"{len(glob.glob(os.path.join(a.out, 'timing', '*.json')))}")
        sys.exit(max(rc))
    device = "cuda" if (a.mode == "gpu" and torch.cuda.is_available()) else "cpu"
    U = units(a)
    mine = U[a.shard::a.nshards]
    print(f"[shard {a.shard}] {len(mine)} of {len(U)} units on {device}", flush=True)
    for u in mine:
        t0 = time.time()
        if u[0] == "faith":
            msg = faith_unit(a, *u[1:], device)
            print(f"[shard {a.shard}] faith {'/'.join(u[1:4])} {u[5]}_{u[6]}: {msg} ({time.time()-t0:.0f}s)", flush=True)
        else:
            msg = timing_unit(a, u[2], u[1], device, "gpu" if device == "cuda" else "cpu")
            print(f"[shard {a.shard}] timing {u[1]} {device}: {msg} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
