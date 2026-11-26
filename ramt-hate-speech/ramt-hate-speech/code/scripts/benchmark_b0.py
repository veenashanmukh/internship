"""Benchmark B0 (Phase 2, gate G2). Run on Kaggle GPU after prepare_data.py + check_data.py.

    python scripts/benchmark_b0.py --splits /kaggle/working/splits --out /kaggle/working/b0 \
        [--encoders GroNLP/hateBERT roberta-base microsoft/deberta-v3-base] [--parallel] [--cpu_latency]

Measures, with the real training code (ramt.train.run), never on test data:
  1. sub-word limit L = max over encoders of the 99th-percentile sub-word length of training
     posts after word truncation at W (rounded up to a multiple of 8) -> <splits>/subword_limit.json
  2. per encoder: training steps/s for one 300-step round of the full model (mu > 0, fp16),
     validation seconds, peak GPU memory; a non-finite loss triggers an fp32 re-run and the
     encoder is flagged for fp32_encoders in configs/frozen.yaml
  3. --parallel: the same run on both GPUs at once (per-GPU rate under CPU/IO contention)
  4. for --xai_encoder: eval throughput, batch-1 latency (GPU, optional CPU), IG time per post
     (50 HateXplain validation posts, 50 steps) with completeness gap, LIME time per post (10 posts)
Writes <out>/b0_results.json. Feed it to scripts/scope.py.

While IHC is blocked (D1), the benchmark trains explicit + hx only; quotas are computed with
--implicit_size (benchmark-only placeholder) so per-step cost reflects the full model's mix.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ENCODERS = ["GroNLP/hateBERT", "roberta-base", "microsoft/deberta-v3-base"]


def subword_limit(splits, encoders, W):
    from ramt.data.datasets import available, load_tokenizer
    import pandas as pd
    texts = []
    for t in ("explicit", "implicit", "hx"):
        if not available(splits, t, "train"):
            continue
        p = os.path.join(splits, f"{t}_train." + ("jsonl" if t == "hx" else "csv"))
        if t == "hx":
            texts += [w for w in pd.read_json(p, lines=True)["words"]]
        else:
            texts += [str(x).split() for x in pd.read_csv(p)["text"]]
    texts = [w[:W] if W else w for w in texts]
    res = {}
    for enc in encoders:
        tok = load_tokenizer(enc)
        lens = []
        for i in range(0, len(texts), 2048):
            e = tok(texts[i:i + 2048], is_split_into_words=True, truncation=False, add_special_tokens=True)
            lens += [len(x) for x in e["input_ids"]]
        lens = np.asarray(lens)
        res[enc] = {"p95": float(np.percentile(lens, 95)), "p99": float(np.percentile(lens, 99)),
                    "max": int(lens.max()), "n": int(len(lens))}
    L = int(8 * np.ceil(max(v["p99"] for v in res.values()) / 8))
    for enc in encoders:
        res[enc]["L_candidate"] = int(8 * np.ceil(res[enc]["p99"] / 8))
    return L, res


def train_bench(splits, enc, out_root, steps, implicit_size, fp16, gpu_tag, save_ckpt):
    from ramt.config import Config
    from ramt.train import run
    st = json.load(open(os.path.join(splits, "data_stats.json")))
    sizes = dict(st["training_sizes"])
    tasks = ["explicit", "implicit", "hx"] if sizes.get("implicit") else ["explicit", "hx"]
    if not sizes.get("implicit"):
        sizes["implicit"] = implicit_size
    name = f"{enc.replace('/', '_')}_{'fp16' if fp16 else 'fp32'}{gpu_tag}"
    cfg = Config(run_name=name, group="bench", seed=42, encoder=enc, tasks=tasks, mu=0.5, fp16=fp16,
                 rounds=1, round_len_full=steps, full_sizes=sizes, evaluate_test=False,
                 splits_dir=splits, out_root=out_root, save_ckpt=save_ckpt)
    t0 = time.time()
    try:
        run(cfg)
    except FloatingPointError as e:
        return {"nonfinite": str(e)}
    tm = json.load(open(os.path.join(out_root, name, "timing.json")))
    tm["wall_sec"] = time.time() - t0
    tm["tasks"] = tasks
    tm["run_dir"] = os.path.join(out_root, name)
    return tm


def xai_bench(splits, enc, ckpt_dir, W, L, n_ig, n_lime, cpu_latency):
    import torch
    from ramt.data.datasets import load_items, load_tokenizer
    from ramt.evaluate import predict
    from ramt.explain import integrated_gradients, lime_scores
    from ramt.model import RAMT
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RAMT(enc, cfg["tasks"], "task")
    sd = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    m.to(dev).eval()
    tok = load_tokenizer(enc)
    items = load_items(splits, "hx", "val", tok, W, L)
    R = {}
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    predict(m, items, cfg["tasks"], dev, tok.pad_token_id, 128, True)
    if dev == "cuda":
        torch.cuda.synchronize()
    R["eval_posts_per_sec_all_heads"] = len(items) / (time.time() - t0)

    def lat(device, n=100):
        mm = m.to(device)
        ts = []
        for it in items[:n]:
            t = time.time()
            predict(mm, [it], ["hx"], device, tok.pad_token_id, 1, device == "cuda")
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.time() - t)
        return {"median_ms": 1000 * float(np.median(ts[5:])), "p95_ms": 1000 * float(np.percentile(ts[5:], 95))}
    R["batch1_latency_gpu" if dev == "cuda" else "batch1_latency_cpu"] = lat(dev)
    if cpu_latency and dev == "cuda":
        R["batch1_latency_cpu"] = lat("cpu", 30)
        m.to(dev)
    tox = [it for it in items if int(it["y"]) != 2]
    ig_t, gaps = [], []
    for it in tox[:n_ig]:
        _, gap, s = integrated_gradients(m, tok, it["words"], "hx", dev, W, L, steps=50)
        ig_t.append(s); gaps.append(gap)
    R["ig_sec_per_post"] = float(np.mean(ig_t[1:])) if len(ig_t) > 1 else None
    R["ig_completeness_gap_median"] = float(np.median(gaps)) if gaps else None
    R["ig_completeness_gap_max"] = float(np.max(gaps)) if gaps else None
    lt = []
    for it in tox[:n_lime]:
        _, s = lime_scores(m, tok, it["words"], "hx", dev, W, L, num_samples=1000)
        lt.append(s)
    R["lime_sec_per_post"] = float(np.mean(lt)) if lt else None
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--xai_encoder", default="roberta-base")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--implicit_size", type=int, default=14274,
                    help="benchmark-only stand-in for the IHC training size while D1 is pending")
    ap.add_argument("--parallel", action="store_true", help="also run the xai encoder on 2 GPUs at once")
    ap.add_argument("--cpu_latency", action="store_true")
    ap.add_argument("--n_ig", type=int, default=50)
    ap.add_argument("--n_lime", type=int, default=10)
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = json.load(open(os.path.join(a.splits, "data_stats.json")))
    W = int(stats["max_words_W"])

    if a._child:                                     # worker for --parallel
        tm = train_bench(a.splits, a.xai_encoder, os.path.join(a.out, "runs"), a.steps, a.implicit_size,
                         True, f"_par{a._child}", False)
        json.dump(tm, open(os.path.join(a.out, f"par_{a._child}.json"), "w"))
        return

    import torch
    from ramt.utils import capture_env, write_json
    R = {"env": capture_env(), "W": W, "steps_per_bench_round": a.steps,
         "ihc_status": stats.get("ihc_status"), "gpus": torch.cuda.device_count(),
         "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if not torch.cuda.is_available():
        print("WARNING: no GPU visible; numbers are CPU numbers and must not be used for scope.")

    L, Lres = subword_limit(a.splits, a.encoders, W)
    write_json({"max_subwords_L": L, "rule": "max over encoders of p99 sub-word length, rounded up to 8",
                "per_encoder": Lres, "W": W}, os.path.join(a.splits, "subword_limit.json"))
    R["L"] = L; R["subword_lengths"] = Lres
    print("L =", L, json.dumps(Lres))

    R["train"] = {}
    for enc in a.encoders:
        tm = train_bench(a.splits, enc, os.path.join(a.out, "runs"), a.steps, a.implicit_size, True, "",
                         enc == a.xai_encoder)
        if "nonfinite" in tm:
            print(f"{enc}: non-finite loss in fp16 -> fp32 re-run")
            tm32 = train_bench(a.splits, enc, os.path.join(a.out, "runs"), a.steps, a.implicit_size, False, "",
                               enc == a.xai_encoder)
            tm = {"fp16_failed": tm["nonfinite"], "fp32_needed": True, **tm32}
        else:
            tm["fp32_needed"] = False
        R["train"][enc] = tm
        print(enc, {k: tm.get(k) for k in ("train_steps_per_sec", "peak_gpu_mem_gb", "fp32_needed")})

    if a.parallel and torch.cuda.device_count() >= 2:
        import gc, shutil
        gc.collect(); torch.cuda.empty_cache()          # release this process's cached GPU memory first
        for g in (0, 1):                                # both workers must start fresh and run together
            shutil.rmtree(os.path.join(a.out, "runs", f"{a.xai_encoder.replace('/', '_')}_fp16_par{g}"),
                          ignore_errors=True)
            if os.path.exists(os.path.join(a.out, f"par_{g}.json")):
                os.remove(os.path.join(a.out, f"par_{g}.json"))
        procs = []
        for g in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--splits", a.splits,
                                           "--out", a.out, "--xai_encoder", a.xai_encoder, "--steps", str(a.steps),
                                           "--implicit_size", str(a.implicit_size), "--_child", str(g)], env=env))
        for p in procs:
            p.wait()
        R["parallel"] = [json.load(open(os.path.join(a.out, f"par_{g}.json")))
                         if os.path.exists(os.path.join(a.out, f"par_{g}.json"))
                         else {"error": f"worker on GPU {g} failed; see log above"} for g in (0, 1)]
        print("parallel per-GPU steps/s:", [p.get("train_steps_per_sec") for p in R["parallel"]])

    xdir = R["train"][a.xai_encoder]["run_dir"]
    R["xai"] = xai_bench(a.splits, a.xai_encoder, xdir, W, L, a.n_ig, a.n_lime, a.cpu_latency)
    print("xai:", json.dumps(R["xai"]))
    write_json(R, os.path.join(a.out, "b0_results.json"))
    print("wrote", os.path.join(a.out, "b0_results.json"))


if __name__ == "__main__":
    main()
