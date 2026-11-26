"""Run a group of jobs, one job per visible GPU in parallel (Kaggle T4 x2 -> 2 at a time).
Finished runs (metrics.json present here or in --done_roots) are skipped, so an
interrupted Kaggle commit can simply be re-run.

  python scripts/run_jobs.py --group pilot_encoder --scope A --splits_dir /kaggle/input/tox-splits-v1 \
      --out_root /kaggle/working/runs --done_roots /kaggle/input/runs-part1/runs --max_hours 10.5
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jobs as J  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=J.GROUPS)
    ap.add_argument("--scope", default="A", choices=["A", "B", "C"])
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out_root", default="/kaggle/working/runs")
    ap.add_argument("--done_roots", nargs="*", default=[])
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--max_hours", type=float, default=10.5, help="no NEW job starts after this")
    ap.add_argument("--only", default=None, help="substring filter on run_name")
    ap.add_argument("--extra", default="{}", help="JSON overrides for every job (e.g. full_sizes for smoke)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    todo = []
    for jb in J.jobs(a.group, a.scope):
        jb = {**jb, **json.loads(a.extra), "splits_dir": a.splits_dir, "out_root": a.out_root}
        if a.only and a.only not in jb["run_name"]:
            continue
        done = any(os.path.exists(os.path.join(r, jb["run_name"], "metrics.json")) for r in [a.out_root] + a.done_roots)
        print(("SKIP " if done else "TODO ") + jb["run_name"])
        if not done:
            todo.append(jb)
    print(f"{len(todo)} job(s) to run")
    if a.dry_run or not todo:
        return

    try:
        import torch
        n_gpu = a.gpus if a.gpus is not None else max(torch.cuda.device_count(), 1)
    except Exception:
        n_gpu = 1
    q = queue.Queue()
    for jb in todo:
        q.put(jb)
    t_start = time.time()
    logdir = os.path.join(a.out_root, "_logs")
    os.makedirs(logdir, exist_ok=True)

    def worker(gpu):
        while (time.time() - t_start) / 3600 < a.max_hours:
            try:
                jb = q.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(logdir, jb["run_name"].replace("/", "__") + ".log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
            t0 = time.time()
            print(f"[gpu{gpu}] START {jb['run_name']}", flush=True)
            with open(log, "w") as f:
                rc = subprocess.call([sys.executable, os.path.join(HERE, "train.py"), "--overrides", json.dumps(jb)],
                                     stdout=f, stderr=subprocess.STDOUT, env=env)
            print(f"[gpu{gpu}] {'OK  ' if rc == 0 else 'FAIL'} {jb['run_name']} ({(time.time()-t0)/60:.1f} min) {log}",
                  flush=True)
        print(f"[gpu{gpu}] time budget reached; not starting new jobs")

    th = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f"finished in {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
