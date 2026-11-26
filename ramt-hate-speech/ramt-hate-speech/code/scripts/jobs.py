"""Expands experiment groups into concrete run configurations (Section 14 + 19).

Pilot groups use the dedicated pilot seed and never evaluate on test (D10, A7).
Main groups require configs/frozen.yaml to be complete (gate G3).
"""
from __future__ import annotations

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt.config import EXTRA_SEEDS, MAIN_SEEDS, PILOT_SEED  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODERS = ["GroNLP/hateBERT", "roberta-base", "microsoft/deberta-v3-base"]
ALL3 = ["explicit", "implicit", "hx"]
SHORT = {"GroNLP/hateBERT": "hatebert", "roberta-base": "roberta", "microsoft/deberta-v3-base": "deberta"}


def frozen():
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "frozen.yaml")))


def _need_frozen(keys):
    f = frozen()
    missing = [k for k in keys if f.get(k) in (None, {}) or (k == "lr_by_encoder" and None in f[k].values())]
    if missing:
        raise SystemExit(f"configs/frozen.yaml incomplete ({missing}); finish the pilot first (gate G3).")
    return f


def _fp16(enc, f=None):
    f = f or frozen()
    return enc not in (f.get("fp32_encoders") or [])


def job(prefix, name, seed, **kw):
    grp = "pilot" if prefix.startswith("pilot") else ("main" if prefix.startswith("main") else prefix)
    return {"group": grp, "run_name": f"{prefix}/{name}/seed{seed}", "seed": seed, **kw}


def jobs(group: str, scope: str = "A"):
    J = []
    if group == "smoke":
        J.append(job("smoke", "ramt", 42, encoder="roberta-base", mu=0.5, data_frac=0.02, rounds=2,
                     tasks=["explicit", "hx"],                    # IHC not needed for the smoke test (D1)
                     round_len_full=50, evaluate_test=False, group="smoke"))
    elif group == "pilot_encoder":                         # 7.2; scope A: lr grid, B/C: 2e-5 only
        lrs = (2e-5, 3e-5) if scope == "A" else (2e-5,)
        for enc in ENCODERS:
            for lr in lrs:
                J.append(job("pilot_encoder", f"mtl_{SHORT[enc]}_lr{lr:g}", PILOT_SEED, encoder=enc, lr=lr,
                             mu=0.0, evaluate_test=False, fp16=_fp16(enc)))
    elif group == "pilot_mu":                              # 3.7
        f = _need_frozen(["encoder", "lr_by_encoder"])
        for mu in (0.25, 0.5, 1.0):
            J.append(job("pilot_mu", f"ramt_mu{mu}", PILOT_SEED, encoder=f["encoder"],
                         lr=f["lr_by_encoder"][f["encoder"]], mu=mu, evaluate_test=False, fp16=_fp16(f["encoder"], f)))
    elif group == "pilot_agg":                             # D4
        f = _need_frozen(["encoder", "lr_by_encoder", "mu"])
        J.append(job("pilot_agg", "ramt_majority", PILOT_SEED, encoder=f["encoder"],
                     lr=f["lr_by_encoder"][f["encoder"]], mu=f["mu"], aggregation="majority",
                     evaluate_test=False, fp16=_fp16(f["encoder"], f)))
    elif group in ("main", "main_extra"):
        f = _need_frozen(["encoder", "lr_by_encoder", "mu", "aggregation", "scope"])
        E, lr, mu, agg = f["encoder"], f["lr_by_encoder"][f["encoder"]], f["mu"], f["aggregation"]
        base = dict(encoder=E, lr=lr, aggregation=agg, fp16=_fp16(E, f))
        if group == "main":
            seeds = MAIN_SEEDS
            pair_seeds = MAIN_SEEDS if f["scope"] in ("A", "B") else MAIN_SEEDS[:2]
            for s in seeds:
                J.append(job("main", "M1_RAMT", s, mu=mu, faithfulness_in_run=True, save_ckpt=True, **base))
                J.append(job("main", "M2_MTL_noRat", s, mu=0.0, faithfulness_in_run=True, save_ckpt=True, **base))
                J.append(job("main", "M3_random_matched", s, mu=mu, rationale_target="random_matched",
                             faithfulness_in_run=True, save_ckpt=True, **base))
                J.append(job("main", "M4_shared_pool", s, mu=mu, pooling="shared", faithfulness_in_run=True, **base))
                for t in ALL3:
                    J.append(job("main", f"M5_STL_{t}", s, tasks=[t], mu=0.0, **base))
                J.append(job("main", "M6_STL_hx_rationale", s, tasks=["hx"], mu=mu, **base))
            for s in pair_seeds:
                for pair in (["explicit", "implicit"], ["explicit", "hx"], ["implicit", "hx"]):
                    J.append(job("main", "M7_pair_" + "+".join(pair), s, tasks=pair, mu=0.0, **base))
                for enc in ENCODERS:
                    if enc != E:
                        J.append(job("main", f"M8_MTL_noRat_{SHORT[enc]}", s, encoder=enc, lr=f["lr_by_encoder"][enc],
                                     mu=0.0, aggregation=agg, fp16=_fp16(enc, f)))
        else:                                              # D11: seeds 4-5 for M1-M3
            for s in EXTRA_SEEDS:
                J.append(job("main_extra", "M1_RAMT", s, mu=mu, faithfulness_in_run=True, save_ckpt=True, **base))
                J.append(job("main_extra", "M2_MTL_noRat", s, mu=0.0, faithfulness_in_run=True, save_ckpt=True, **base))
                J.append(job("main_extra", "M3_random_matched", s, mu=mu, rationale_target="random_matched",
                             faithfulness_in_run=True, save_ckpt=True, **base))
    elif group == "optional_cls":                          # D13: 3 seeds or omitted
        f = _need_frozen(["encoder", "lr_by_encoder", "aggregation"])
        for s in MAIN_SEEDS:
            for t in ALL3:
                J.append(job("optional", f"CLS_STL_{t}", s, tasks=[t], mu=0.0, pooling="cls", encoder=f["encoder"],
                             lr=f["lr_by_encoder"][f["encoder"]], aggregation=f["aggregation"],
                             fp16=_fp16(f["encoder"], f), group="optional"))
    elif group == "optional_shared_norat":                 # 4.4 counterpart
        f = _need_frozen(["encoder", "lr_by_encoder", "aggregation"])
        for s in MAIN_SEEDS:
            J.append(job("optional", "M4b_shared_pool_noRat", s, pooling="shared", mu=0.0, encoder=f["encoder"],
                         lr=f["lr_by_encoder"][f["encoder"]], aggregation=f["aggregation"],
                         fp16=_fp16(f["encoder"], f), faithfulness_in_run=True, group="optional"))
    else:
        raise ValueError(group)
    for j in J:
        j.setdefault("group", "main")
    return J


GROUPS = ["smoke", "pilot_encoder", "pilot_mu", "pilot_agg", "main", "main_extra", "optional_cls",
          "optional_shared_norat"]
