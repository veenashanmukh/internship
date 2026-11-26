"""Gate G3: apply the pre-registered pilot selection rules and write configs/frozen.yaml.
Validation scores only (pilot runs never evaluate on test).

    python scripts/freeze.py --runs /kaggle/working/runs --step encoder|mu|agg|date

Rules (methodology 7.2 / 8.3 / A4):
  encoder : per encoder, lr with the highest best_val_composite; encoder = highest of those.
  mu      : highest validation composite; values within 0.005 of the best are tied and the
            tie goes to higher HateXplain validation plausibility (token AUPRC, majority gold).
  agg     : majority replaces soft only if its composite exceeds soft's (at the frozen mu) by
            more than 0.005; otherwise soft.
Each step appends its evidence to <runs>/pilot_decisions.json.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "configs", "frozen.yaml")
TIE = 0.005


def runs_of(root, group):
    out = []
    for m in sorted(glob.glob(os.path.join(root, "**", group, "*", "seed1", "metrics.json"), recursive=True)):
        d = os.path.dirname(m)
        out.append({"dir": d, "metrics": json.load(open(m)), "config": json.load(open(os.path.join(d, "config.json")))})
    if not out:
        raise SystemExit(f"no finished runs for {group} under {root}")
    return out


def set_key(text, key, value, indent=""):
    pat = re.compile(rf"^({re.escape(indent)}{re.escape(key)}:)\s*[^#\n]*(#.*)?$", re.M)
    if not pat.search(text):
        raise SystemExit(f"key {key} not found in frozen.yaml")
    return pat.sub(lambda m: f"{m.group(1)} {value}" + (f"   {m.group(2)}" if m.group(2) else ""), text, count=1)


def record(root, entry):
    p = os.path.join(root, "pilot_decisions.json")
    log = json.load(open(p)) if os.path.exists(p) else []
    log.append(entry)
    json.dump(log, open(p, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--step", required=True, choices=["encoder", "mu", "agg", "date"])
    a = ap.parse_args()
    fz = open(FROZEN).read()

    if a.step == "encoder":
        best = {}
        for r in runs_of(a.runs, "pilot_encoder"):
            e, lr, c = r["config"]["encoder"], r["config"]["lr"], r["metrics"]["best_val_composite"]
            if e not in best or c > best[e][1]:
                best[e] = (lr, c)
        enc = max(best, key=lambda e: best[e][1])
        for e, (lr, _) in best.items():
            fz = set_key(fz, e, f"{float(lr):.1e}", indent="  ")   # 2.0e-05: PyYAML reads 2e-05 as a string
        fz = set_key(fz, "encoder", enc)
        ev = {"step": "encoder", "per_encoder_best": {e: {"lr": v[0], "val_composite": v[1]} for e, v in best.items()},
              "chosen": enc}
    elif a.step == "mu":
        rs = runs_of(a.runs, "pilot_mu")
        top = max(r["metrics"]["best_val_composite"] for r in rs)
        tied = [r for r in rs if r["metrics"]["best_val_composite"] >= top - TIE]
        pick = max(tied, key=lambda r: r["metrics"]["plausibility_hx_val"]["token_auprc"])
        mu = pick["config"]["mu"]
        fz = set_key(fz, "mu", mu)
        ev = {"step": "mu", "candidates": {str(r["config"]["mu"]): {
            "val_composite": r["metrics"]["best_val_composite"],
            "val_plausibility_auprc": r["metrics"]["plausibility_hx_val"]["token_auprc"]} for r in rs},
              "tied_within_0.005": [r["config"]["mu"] for r in tied], "chosen": mu}
    elif a.step == "agg":
        import yaml
        mu = yaml.safe_load(fz)["mu"]
        soft = [r for r in runs_of(a.runs, "pilot_mu") if r["config"]["mu"] == mu]
        maj = runs_of(a.runs, "pilot_agg")
        if len(soft) != 1 or len(maj) != 1:
            raise SystemExit("need exactly one soft run at the frozen mu and one majority run")
        cs, cm = soft[0]["metrics"]["best_val_composite"], maj[0]["metrics"]["best_val_composite"]
        agg = "majority" if cm - cs > TIE else "soft"
        fz = set_key(fz, "aggregation", agg)
        ev = {"step": "agg", "soft_val_composite": cs, "majority_val_composite": cm, "chosen": agg}
    else:
        fz = set_key(fz, "frozen_on", datetime.date.today().isoformat())
        ev = {"step": "date", "frozen_on": datetime.date.today().isoformat()}

    open(FROZEN, "w").write(fz)
    record(a.runs, ev)
    print(json.dumps(ev, indent=1))


if __name__ == "__main__":
    sys.exit(main())
