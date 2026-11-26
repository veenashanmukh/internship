"""Overnight sequence after the encoder pilot: pilot_mu -> freeze mu -> pilot_agg -> freeze
aggregation -> date (gate G3) -> main -> main_extra -> optional groups. Pre-registered rules
only (scripts/freeze.py); finished runs are skipped, so a re-run resumes where it stopped.

    python scripts/run_stage3.py --splits <splits dir> --runs /kaggle/working/runs [--hours 10.5]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"STOPPED: '{cmd}' exited with {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--hours", type=float, default=10.5, help="start no new job after this many hours")
    a = ap.parse_args()
    py = sys.executable
    left = lambda: max(0.05, a.hours - (time.time() - T0) / 3600)
    run = lambda g: sh(f"{py} scripts/run_jobs.py --group {g} --scope A --splits_dir {a.splits} "
                       f"--out_root {a.runs} --max_hours {left():.2f}")
    fz = lambda: yaml.safe_load(open(os.path.join(ROOT, "configs", "frozen.yaml")))

    if fz().get("mu") is None:
        run("pilot_mu")
        sh(f"{py} scripts/freeze.py --runs {a.runs} --step mu")
    if fz().get("aggregation") is None:
        run("pilot_agg")
        sh(f"{py} scripts/freeze.py --runs {a.runs} --step agg")
    if fz().get("frozen_on") is None:
        sh(f"{py} scripts/freeze.py --runs {a.runs} --step date")
    os.makedirs(a.runs, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "configs", "frozen.yaml"), os.path.join(a.runs, "frozen.yaml"))
    print("\nGATE G3 frozen config:\n" + open(os.path.join(ROOT, "configs", "frozen.yaml")).read(), flush=True)

    groups = ["main", "main_extra"] if fz().get("five_seeds_m1_m3") else ["main"]
    if fz().get("cls_baseline"):
        groups.append("optional_cls")
    groups.append("optional_shared_norat")
    for g in groups:
        run(g)
    import glob
    print("\ncompleted runs per group (metrics.json present):")
    for g in ["pilot_mu", "pilot_agg"] + groups:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import jobs as J
        want = J.jobs(g)
        done = sum(os.path.exists(os.path.join(a.runs, j["run_name"], "metrics.json")) for j in want)
        print(f"  {g:24s} {done}/{len(want)}")
    print(f"\nSTAGE 3 FINISHED in {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
