# RAMT on Kaggle: run guide (Phases 1–2)

Phases 3–5 are added to this guide only after their gates pass. Nothing here touches test
labels except the in-run evaluation of **main** runs, which are not started yet.

## 0. Kaggle setup (once)

1. Upload this folder as a Kaggle **Dataset** named `ramt-code` (or attach it as a utility script).
2. Attach these inputs to the notebook:
   - **Jigsaw Toxic Comment Classification**: the competition data (`train.csv`, `test.csv`, `test_labels.csv`).
   - **HateXplain**: `dataset.json` and `post_id_divisions.json` from github.com/hate-alert/HateXplain (`Data/`), uploaded as a private dataset.
   - **HateCheck**: `test_suite_cases.csv` from github.com/paul-rottger/hatecheck-data, uploaded as a private dataset.
   - **OLID**: `testset-levela.tsv` and `labels-levela.csv` (OffensEval 2019), uploaded as a private dataset.
   - **IHC**: **do not attach yet.** IHC is used only after the authors' written permission arrives (D1). When it does, upload the stage-1 file plus the permission e-mail saved as a PDF or TXT, as one private dataset.
3. Settings: Accelerator **GPU T4 ×2**, Internet **on** (needed to download encoders from the Hugging Face hub).

```python
!cp -r /kaggle/input/ramt-code /kaggle/working/ramt && cd /kaggle/working/ramt && pip install -q -r requirements.txt
%cd /kaggle/working/ramt
```

## 1. CPU test suite (gate G0, about 1 minute)

```python
!python -m pytest -q tests/
```
All 21 tests must pass before going further.

## 2. Prepare splits (gate G1)

```python
!python scripts/prepare_data.py \
   --jigsaw_dir /kaggle/input/jigsaw-toxic-comment-classification-challenge \
   --hx_dir /kaggle/input/<your-hatexplain-dataset> \
   --olid_dir /kaggle/input/<your-olid-dataset> \
   --hatecheck_file /kaggle/input/<your-hatecheck-dataset>/test_suite_cases.csv \
   --out /kaggle/working/splits
# later, with permission:  --ihc_file .../implicit_hate_v1_stg1_posts.tsv --ihc_permission_record .../permission.pdf
!python scripts/check_data.py --splits /kaggle/working/splits \
   --tokenizers GroNLP/hateBERT roberta-base microsoft/deberta-v3-base
```
`check_data.py` must end with `ALL CHECKS PASSED`. Then save `/kaggle/working/splits` as a new
private Kaggle dataset, `tox-splits-v1`, so every later session uses byte-identical splits
(`split_hashes.json` is checked again at each run).

## 3. Benchmark B0 (gate G2, about 45–75 minutes of GPU time; this is an assumption and B0 measures the real figure)

```python
!python scripts/benchmark_b0.py --splits /kaggle/working/splits --out /kaggle/working/b0 --parallel --cpu_latency
!python scripts/scope.py --b0 /kaggle/working/b0/b0_results.json
```
B0 writes `splits/subword_limit.json` (L). **Re-save the splits dataset after B0** so L travels with it.
Send me `b0_results.json` and the `scope.py` output. The scope decision (D2) and the fp32 list go
into `configs/frozen.yaml` together with the pilot results.

## 4. Smoke run on GPU (2% of the data, 2 short rounds, no test evaluation)

```python
!python scripts/run_jobs.py --group smoke --splits_dir /kaggle/working/splits --out_root /kaggle/working/runs
```

## 5. Pilot (Phase 3). Blocked until IHC permission (D1)

`pilot_encoder` trains the full multi-task model, which includes the implicit task, so it cannot start
without IHC. `run_jobs.py --group pilot_encoder` will stop with `IHCBlocked` rather than substitute data.

## Session hygiene
- A Kaggle session has a 12 h limit; `run_jobs.py --max_hours 10.5` starts no new job after 10.5 h. A run is complete only once `metrics.json` exists, so interrupted runs simply re-run.
- Each session: **Save Version → Save output**, then attach the previous output as an input and pass it as `--done_roots` so completed runs are skipped.
