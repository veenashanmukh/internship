"""End-to-end Phase 5: tiny runs -> explain_eval (CPU) -> analyze. Checks the pipeline and
its outputs, not the scientific result."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from ramt.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def runs(tiny_encoder, splits, tmp_path_factory):
    from ramt.train import run
    out = tmp_path_factory.mktemp("p5runs")
    specs = {"M1_RAMT": dict(mu=1.0, save_ckpt=True), "M2_MTL_noRat": dict(mu=0.0, save_ckpt=True),
             "M3_random_matched": dict(mu=1.0, rationale_target="random_matched", save_ckpt=True),
             "M5_STL_explicit": dict(tasks=["explicit"], mu=0.0), "M5_STL_implicit": dict(tasks=["implicit"], mu=0.0),
             "M5_STL_hx": dict(tasks=["hx"], mu=0.0)}
    for name, kw in specs.items():
        for seed in (42, 123):
            base = dict(run_name=f"main/{name}/seed{seed}", group="main", seed=seed, encoder=tiny_encoder,
                        tasks=["explicit", "implicit", "hx"], fp16=False, rounds=2, round_len_full=12, batch_size=8,
                        eval_batch_size=32, threshold_bootstrap=10, splits_dir=splits, out_root=str(out),
                        max_subwords=64, evaluate_test=True, faithfulness_in_run=False, num_workers=0)
            base.update(kw)
            run(Config(**base))
    return str(out)


@pytest.fixture(scope="module")
def explained(runs, splits, tmp_path_factory):
    out = tmp_path_factory.mktemp("explain")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "explain_eval.py"), "--runs", runs,
                        "--splits", splits, "--out", str(out), "--timing_posts", "21"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return str(out)


def test_explain_outputs(explained):
    import numpy as np
    f = os.path.join(explained, "faith", "main", "M1_RAMT", "seed42", "hx_test.npz")
    z = np.load(f, allow_pickle=True)
    for k in ("comp_attn", "suff_attn", "comp_rand", "suff_rand", "comp_ig", "suff_ig", "ig_gap", "plaus_attn", "plaus_ig"):
        assert k in z.files, k
    z2 = np.load(f.replace("M1_RAMT", "M2_MTL_noRat"), allow_pickle=True)
    assert list(z["post_id"]) == list(z2["post_id"])             # identical fixed set across models
    assert "comp_ig" not in z2.files                             # IG only for M1
    n = sum(len(fs) for _, _, fs in os.walk(os.path.join(explained, "faith")))
    assert n == 3 * 2 * 2 * 3                                    # models x seeds x splits x tasks
    import glob as _g
    tf = _g.glob(os.path.join(explained, "timing", "*_seed42.json"))      # gpu_ or cpu_ by machine
    assert len(tf) == 1, tf
    t = json.load(open(tf[0]))
    assert {"pred", "pred_attn", "ig", "lime"} <= set(t)


def test_analysis_runs(runs, splits, explained, tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze.py"), "--runs", runs, "--splits", splits,
                        "--explain", explained, "--explain_cpu", explained, "--out", str(tmp_path), "--B", "40"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    conf = pd.read_csv(tmp_path / "confirmatory_tests.csv")
    assert len(conf) == 12, conf["test"].tolist()
    assert conf["p_holm"].notna().all()
    ok = conf["p"].notna()
    assert (conf.loc[ok, "p_holm"] >= conf.loc[ok, "p"] - 1e-12).all()
    assert (conf.loc[~ok, "p_holm"] == 1.0).all() and not conf.loc[~ok, "claim"].any()
    res = json.load(open(tmp_path / "analysis.json"))
    assert res["H3"]["verdict"] in ("supported", "partially supported", "not supported")
    crit = set(res["E"]["criteria"])
    assert {"E4", "E5"} <= crit and ({"E1_cpu", "E2_cpu"} <= crit or {"E1_gpu", "E2_gpu"} <= crit)
    s = res["sanity"]
    assert abs(s["M1_seed42_jigsaw_macro_auprc_recomputed"] - s["M1_seed42_jigsaw_macro_auprc_in_run"]) < 1e-6
    assert (tmp_path / "descriptive.csv").exists() and (tmp_path / "exploratory_tests.csv").exists()
