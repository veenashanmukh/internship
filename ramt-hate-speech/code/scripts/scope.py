"""D2 scope rule: pick the largest scope whose required training rate is <= 85% of the
rate measured in B0.  Totals are the methodology's step-equivalent budgets (Section 19,
recommendation (c)); the budget is 24 usable Kaggle session hours.

    python scripts/scope.py --b0 /kaggle/working/b0/b0_results.json [--encoder roberta-base]
"""
import argparse
import json

TOTALS = {"A": 333_000, "B": 313_000, "C": 281_000}       # step-equivalents (doc, rec. (c))
FIVE_SEEDS_EXTRA = 45_300                                  # 36,000 steps + ~9,300 eval (D11)
SAFETY = 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", required=True)
    ap.add_argument("--encoder", default="roberta-base", help="rate used for the budget (the main encoder)")
    ap.add_argument("--hours", type=float, default=24.0)
    a = ap.parse_args()
    b0 = json.load(open(a.b0))
    if not b0.get("gpu_name"):
        raise SystemExit("b0_results.json was produced without a GPU; scope cannot be set from it.")
    single = b0["train"][a.encoder]["train_steps_per_sec"]
    par = b0.get("parallel")
    if par and all("train_steps_per_sec" in p for p in par):
        per_gpu, n = min(p["train_steps_per_sec"] for p in par), 2
    else:
        per_gpu, n = single, 1
    usable = SAFETY * per_gpu
    print(f"measured: {single:.2f} steps/s single; per-GPU used: {per_gpu:.2f} x {n} GPU(s); "
          f"85% = {usable:.2f}")
    rows, chosen = [], None
    for s in ("A", "B", "C"):
        for five in (True, False):
            tot = TOTALS[s] + (FIVE_SEEDS_EXTRA if five else 0)
            need = tot / (a.hours * 3600 * n)
            hrs = tot / (per_gpu * n * 3600)
            ok = need <= usable
            rows.append((s, five, tot, need, hrs, ok))
            print(f"scope {s}{' + 5 seeds M1-M3' if five else '':17s} total {tot:>7,}  "
                  f"needs {need:.2f}/GPU  est. {hrs:5.1f} h  {'OK' if ok else 'too slow'}")
            if ok and chosen is None:
                chosen = (s, five)
    slow = {e: v.get("fp32_needed") for e, v in b0["train"].items()}
    print("fp32 needed:", slow)
    if chosen is None:
        print("\nNO SCOPE FITS: even scope C exceeds 85% of the measured rate. This is a blocker to "
              "report (D2); do not shrink the design silently.")
    else:
        print(f"\nRECOMMENDED: scope {chosen[0]}, five seeds for M1-M3: {'yes' if chosen[1] else 'no'}"
              "\n(record in configs/frozen.yaml: scope, five_seeds_m1_m3, fp32_encoders)")
    print("Note: step-equivalents include the doc's evaluation estimate; compare b0 val/calibration "
          "seconds against it before committing.")


if __name__ == "__main__":
    main()
