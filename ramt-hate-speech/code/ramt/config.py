"""Run configuration. One Config = one training run (one configuration x one seed).

Every field maps to a decision in the approved methodology (Sections 1-19 of the
review document). Fields marked IMPL are implementation details the methodology
does not fix; they are listed in IMPLEMENTATION_LOG.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml

TASKS = ("explicit", "implicit", "hx")
PILOT_SEED = 1                      # D10 / A7: dedicated pilot seed, never used in main runs
MAIN_SEEDS = (42, 123, 2024)
EXTRA_SEEDS = (7, 999)              # D11: seeds 4-5 for M1-M3 if budget allows


@dataclass
class Config:
    run_name: str = "debug/run"
    group: str = "debug"                  # pilot | main | optional | smoke | debug
    seed: int = 42

    # --- model (Section 14 "Model") ---
    encoder: str = "roberta-base"
    tasks: List[str] = field(default_factory=lambda: list(TASKS))
    pooling: str = "task"                 # task | shared | cls   (cls only for the D13 baseline)
    dropout: float = 0.1                  # IMPL: dropout on pooled vector before the linear head

    # --- rationale supervision (Sections 3, 5) ---
    mu: float = 0.0
    aggregation: str = "soft"             # soft | majority | union | intersection
    rationale_target: str = "human"       # human | random_matched

    # --- optimisation (Sections 2, 14; D8, A6) ---
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    batch_size: int = 32
    eval_batch_size: int = 128
    fp16: bool = True
    rounds: int = 6
    round_len_full: int = 1000           # steps per round in the full three-task model
    sampling_temperature: float = 2.0
    # training-set sizes of the FULL three-task configuration; quotas are derived
    # from these so that subsets keep identical per-task quotas (exposure matching)
    full_sizes: Dict[str, int] = field(default_factory=dict)
    length_bucketing: bool = True        # IMPL: sort within 50-batch chunks to cut padding

    # --- input length (Section 7.1) ---
    max_words: int = 0                    # W from data_stats.json (0 = read from splits)
    max_subwords: int = 0                 # L from benchmark B0 (0 = read from splits/b0)

    # --- evaluation (Sections 8, 9) ---
    evaluate_test: bool = True            # False for every pilot run (A7, 8.1)
    threshold_bootstrap: int = 200
    faithfulness_in_run: bool = False     # IMPL: attention AOPC inside the run (M1-M4)
    faithfulness_max_posts: int = 1000    # IMPL: cap per test set, fixed sample
    random_draws: int = 5                 # Section 9: random-explanation baseline draws

    # --- paths / storage (Section 11) ---
    splits_dir: str = "/kaggle/input/tox-splits-v1"
    out_root: str = "/kaggle/working/runs"
    save_ckpt: bool = False
    data_frac: float = 1.0                # < 1 only for smoke tests
    num_workers: int = 0

    # ------------------------------------------------------------------
    def validate(self) -> "Config":
        assert all(t in TASKS for t in self.tasks) and self.tasks, self.tasks
        assert self.pooling in ("task", "shared", "cls"), self.pooling
        assert self.aggregation in ("soft", "majority", "union", "intersection")
        assert self.rationale_target in ("human", "random_matched")
        if self.pooling == "cls":
            assert self.mu == 0.0, "[CLS] pooling has no attention to supervise"
        if self.mu > 0:
            assert "hx" in self.tasks, "rationale loss needs the HateXplain task"
        if self.group.startswith("pilot"):
            assert self.seed == PILOT_SEED, "pilot runs must use the dedicated pilot seed (D10)"
            assert not self.evaluate_test, "pilot runs never compute test metrics (A7)"
        elif self.group.startswith(("main", "optional")):
            assert self.seed != PILOT_SEED, "main runs must not use the pilot seed (D10)"
        assert 0 < self.data_frac <= 1
        return self

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        c = Config()
        for k, v in d.items():
            if not hasattr(c, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(c, k, v)
        return c.validate()

    @staticmethod
    def from_yaml(path: str, overrides: Optional[dict] = None) -> "Config":
        d = yaml.safe_load(open(path)) or {}
        d.update(overrides or {})
        return Config.from_dict(d)

    def dump(self, path: str):
        json.dump(self.to_dict(), open(path, "w"), indent=2)
