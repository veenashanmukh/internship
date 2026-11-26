"""Run one configuration.  python scripts/train.py --overrides '{"run_name": "...", "seed": 42}'"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt.config import Config  # noqa: E402
from ramt.train import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "base.yaml"))
    ap.add_argument("--overrides", default="{}", help="JSON object of config overrides")
    a = ap.parse_args()
    run(Config.from_yaml(a.base, json.loads(a.overrides)))
