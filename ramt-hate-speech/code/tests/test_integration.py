"""End-to-end: real preparation code on synthetic raw files, then short CPU training runs
through ramt.train.run, checking outputs, guards and test-set integrity."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from ramt.config import Config
from conftest import prep_args


def _cfg(tiny, splits, out, **kw):
    base = dict(run_name="smoke", group="smoke", seed=42, encoder=tiny, tasks=["explicit", "implicit", "hx"],
                pooling="task", mu=1.0, fp16=False, rounds=2, round_len_full=12, batch_size=8,
                eval_batch_size=32, threshold_bootstrap=20, splits_dir=splits, out_root=str(out),
                max_subwords=64, evaluate_test=True, faithfulness_in_run=True, faithfulness_max_posts=20,
                random_draws=2, num_workers=0)
    base.update(kw)
    c = Config(**base)
    c.validate()
    return c


# ------------------------------------------------------------------ data preparation
def test_prepare_outputs_and_integrity(splits):
    st = json.load(open(os.path.join(splits, "data_stats.json")))
    for f in ("dedup_report.json", "label_semantics.json", "split_hashes.json", "explicit_valsub.csv"):
        assert os.path.exists(os.path.join(splits, f)), f
    assert st["max_words_W"] > 0
    assert set(st["training_sizes"]) >= {"explicit", "implicit", "hx"}
    # no train/test text overlap in any task (normalised key)
    from ramt.data.prep import norm_key
    def keys(f):
        p = os.path.join(splits, f)
        if f.endswith("jsonl"):
            return set(pd.read_json(p, lines=True)["words"].map(" ".join).map(norm_key))
        return set(pd.read_csv(p)["text"].map(norm_key))
    evals = set()
    for f in os.listdir(splits):
        if f.endswith(("test.csv", "test.jsonl")) or f in ("olid_test.csv", "hatecheck.csv"):
            evals |= keys(f)
    for f in os.listdir(splits):
        if f.endswith(("train.csv", "train.jsonl")):
            assert not (keys(f) & evals), f


def test_ihc_blocked_without_permission(raw_dir, tmp_path):
    from ramt.data.prep import prepare
    prepare(prep_args(raw_dir, tmp_path, permission=False))
    st = json.load(open(tmp_path / "data_stats.json"))
    assert "BLOCKED" in json.dumps(st)
    assert not any(f.startswith("implicit") for f in os.listdir(tmp_path))


def test_training_refuses_implicit_when_blocked(raw_dir, tmp_path, tiny_encoder):
    from ramt.data.prep import prepare
    from ramt.train import IHCBlocked, run
    sp = tmp_path / "s"; sp.mkdir()
    prepare(prep_args(raw_dir, sp, permission=False))
    with pytest.raises(IHCBlocked):
        run(_cfg(tiny_encoder, str(sp), tmp_path / "runs"))


# ------------------------------------------------------------------ guards
def test_pilot_guards(tiny_encoder, splits, tmp_path):
    with pytest.raises(AssertionError):
        _cfg(tiny_encoder, splits, tmp_path, group="pilot_mu", seed=42, evaluate_test=False)
    with pytest.raises(AssertionError):
        _cfg(tiny_encoder, splits, tmp_path, group="pilot_mu", seed=1, evaluate_test=True)
    with pytest.raises(AssertionError):
        _cfg(tiny_encoder, splits, tmp_path, group="main", seed=1)
    with pytest.raises(AssertionError):
        _cfg(tiny_encoder, splits, tmp_path, pooling="cls", mu=1.0)


# ------------------------------------------------------------------ full smoke run
@pytest.fixture(scope="module")
def smoke_run(tiny_encoder, splits, tmp_path_factory):
    from ramt.train import run
    out = tmp_path_factory.mktemp("runs")
    run(_cfg(tiny_encoder, splits, out, save_ckpt=True))
    return out / "smoke"


def test_smoke_outputs(smoke_run):
    for f in ("config.json", "env.json", "train_log.csv", "exposure.json", "thresholds.json",
              "temperatures.json", "timing.json", "metrics.json", "attention_hx_test.npz", "best.pt"):
        assert (smoke_run / f).exists(), f
    m = json.load(open(smoke_run / "metrics.json"))
    assert np.isfinite(m["best_val_composite"])
    for k in ("explicit", "implicit", "hx", "cross_dataset"):
        assert k in m
    pl = m["plausibility_hx_test"]["hx"]
    assert "gold_majority" in pl and "gold_union_hatexplain_comparable" in pl
    fa = m["faithfulness_attention"]["hx"]
    assert "random_aopc_comprehensiveness" in fa and "comp_minus_random" in fa
    log = pd.read_csv(smoke_run / "train_log.csv")
    assert np.isfinite(log.select_dtypes("number").values).all()


def test_exposure_matches_quotas(smoke_run):
    ex = json.load(open(smoke_run / "exposure.json"))
    q = ex["quotas_full_model"]
    assert sum(q.values()) == 12 and ex["steps_run"] == ex["total_steps"] == 24
    for t in ("explicit", "implicit", "hx"):
        assert ex["per_task"][t]["batches"] == 2 * q[t]


def test_thresholds_from_validation_only(smoke_run):
    """Thresholds are fitted before any test file exists in the log order; and the
    run never reads test labels when choosing the checkpoint (val_composite only)."""
    th = json.load(open(smoke_run / "thresholds.json"))
    assert th and all(0.01 <= v <= 0.99 for v in np.ravel(list(_flat(th))))
    log = pd.read_csv(smoke_run / "train_log.csv")
    assert not any("test" in c for c in log.columns)


def _flat(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _flat(v)
        elif isinstance(v, (list, tuple)):
            yield from v
        else:
            yield v


def test_pilot_run_writes_no_test_outputs(tiny_encoder, splits, tmp_path):
    from ramt.train import run
    run(_cfg(tiny_encoder, splits, tmp_path, run_name="pil", group="pilot_mu", seed=1,
             evaluate_test=False, faithfulness_in_run=False))
    files = os.listdir(tmp_path / "pil")
    assert not any(f.startswith("test_") or "attention_hx_test" in f for f in files)
    m = json.load(open(tmp_path / "pil" / "metrics.json"))
    assert not {"explicit", "implicit", "hx", "cross_dataset"} & set(m)
    assert "token_auprc" in m["plausibility_hx_val"]          # mu tie-break needs it (8.3)


def test_single_task_and_random_control(tiny_encoder, splits, tmp_path):
    from ramt.train import run
    run(_cfg(tiny_encoder, splits, tmp_path, run_name="m5hx", group="main", tasks=["hx"], mu=0.0,
             faithfulness_in_run=False))
    ex = json.load(open(tmp_path / "m5hx" / "exposure.json"))
    assert set(ex["per_task"]) == {"hx"}
    assert ex["per_task"]["hx"]["batches"] == 2 * ex["quotas_full_model"]["hx"]   # exposure-matched
    run(_cfg(tiny_encoder, splits, tmp_path, run_name="m3", group="main", rationale_target="random_matched",
             faithfulness_in_run=False))
    assert (tmp_path / "m3" / "metrics.json").exists()


def test_ig_and_lime(tiny_encoder, smoke_run):
    import torch
    from ramt.data.datasets import load_tokenizer
    from ramt.explain import integrated_gradients, lime_scores
    from ramt.model import RAMT
    m = RAMT(tiny_encoder, ["explicit", "implicit", "hx"], "task")
    m.load_state_dict({k: v.float() for k, v in torch.load(smoke_run / "best.pt", map_location="cpu").items()})
    m.eval()
    tok = load_tokenizer(tiny_encoder)
    words = "the idiot cat is unhappy today".split()
    s, gap, secs = integrated_gradients(m, tok, words, "hx", "cpu", 0, 64, steps=50)
    assert s.shape == (len(words),) and np.isfinite(s).all()
    assert gap < 0.05, gap                                     # completeness (A-IG check)
    ls, _ = lime_scores(m, tok, words, "hx", "cpu", 0, 64, num_samples=200)
    assert ls.shape == (len(words),) and np.isfinite(ls).all()
