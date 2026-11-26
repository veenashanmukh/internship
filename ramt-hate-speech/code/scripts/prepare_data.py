"""Build frozen splits (Section 1, step S0). Run in a Kaggle CPU notebook.

  python scripts/prepare_data.py \
    --jigsaw_dir /kaggle/input/jigsaw-toxic-comment-classification-challenge \
    --hx_dir /kaggle/input/tox-raw/HateXplain/Data \
    --olid_dir /kaggle/input/tox-raw/olid \
    --hatecheck_file /kaggle/input/tox-raw/hatecheck-data/test_suite_cases.csv \
    --ihc_file /kaggle/input/tox-raw-private/implicit_hate_v1_stg1_posts.tsv \
    --ihc_permission_record /kaggle/input/tox-raw-private/ihc_permission_email.pdf \
    --out /kaggle/working/splits

IHC is processed ONLY when --ihc_permission_record points to an existing file
(the authors' written permission, D1). Without it, IHC is skipped and recorded as blocked.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramt.data.prep import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jigsaw_dir", required=True)
    ap.add_argument("--hx_dir", required=True)
    ap.add_argument("--olid_dir", default=None)
    ap.add_argument("--hatecheck_file", default=None)
    ap.add_argument("--ihc_file", default=None)
    ap.add_argument("--ihc_permission_record", default=None)
    ap.add_argument("--out", default="/kaggle/working/splits")
    ap.add_argument("--n_clean_train", type=int, default=30000)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--valsub_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    res = prepare(ap.parse_args())
    print(json.dumps(res, indent=1, default=str))
