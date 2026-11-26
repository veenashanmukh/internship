# RAMT: Rationale-supervised attention for multi-task hate-speech detection

Code, documents and results for the paper *Rationale-supervised attention for multi-task
hate-speech detection: human-aligned, faithful explanations at no accuracy or latency cost*
(draft for *Engineering Applications of Artificial Intelligence*).

Authors: Veena Shanmukh Bagildavar, Varsha Shivanand K, Yashwitha T S, Madderla Chiranjeevi
(School of Computer Science and Engineering, RV University, Bengaluru, India).

## What is in this folder

| Folder | Contents |
|---|---|
| `paper/` | LaTeX source of the paper (`main.tex`, `references.bib`, `figures/`) and the compiled draft `ramt_paper_draft.pdf`. Upload `main.tex`, `references.bib` and `figures/` to Overleaf to edit. |
| `docs/methodology.md` | The research methodology (hypotheses, metrics, controls, statistical tests), fixed before any test-set evaluation. |
| `docs/IMPLEMENTATION_LOG.md` | Everything that was done, in order, with every bug fix and every deviation from the methodology. |
| `code/` | The `ramt` Python package, the scripts that ran each stage on Kaggle, the configuration files and the unit tests. See `code/README_KAGGLE.md` for how to run it. |
| `results/` | All numbers in the paper: summary tables (`*.csv`), the confirmatory and exploratory tests (`analysis/`), the HateXplain-protocol evaluation (`hatexplain_protocol/`), GPU and CPU timings (`timing/`), per-run logs for all 61 trained models (`runs.zip`, unzip to read) and the figures. |

## Main results (test sets)

- Human rationales raise the attention's plausibility (token-AUPRC) from 0.69 (no rationale loss)
  and 0.40 (random rationales) to 0.92, and its faithfulness (AOPC comprehensiveness above random)
  from 0.28 and 0.01 to 0.46, in all five seeds, with no change in classification.
- The effect transfers to the explicit-toxicity head, which never sees rationales (plausibility 0.89 vs at most 0.55).
- The attention is as faithful as Integrated Gradients and adds no latency: IG is 4.8x (GPU) and 52x (CPU)
  slower, LIME 22.8x and 281x.
- Multi-task training does not improve accuracy over single-task models; a consistent but
  non-significant drop in the GMB-BPSN bias metric is reported.

## Reproducing

1. Download the data yourself (not redistributed here, see below) and upload it to Kaggle.
2. Follow `code/README_KAGGLE.md`: prepare data, check it, run the GPU benchmark, the pilot,
   the main runs, then `scripts/explain_eval.py`, `scripts/analyze.py` and `scripts/hatexplain_eval.py`.
3. Unit tests: `cd code && pip install -r requirements.txt && pytest -q`.

Hardware used: Kaggle, 2x NVIDIA T4 GPUs (training and GPU timing) and a 4-thread CPU session (CPU timing).

## Data (not included)

| Dataset | Source | Licence / terms |
|---|---|---|
| HateXplain | Mathew et al. (2021), github.com/hate-alert/HateXplain | MIT |
| Implicit Hate Corpus | ElSherief et al. (2021), github.com/SALT-NLP/implicit-hate | CC BY 4.0; tweet text is not redistributed |
| Jigsaw Toxic Comment Classification | Kaggle competition | Kaggle competition rules |
| HateCheck | Röttger et al. (2021) | CC BY 4.0 |

No dataset text and no trained model weights are stored in this repository. The per-run files in
`results/runs.zip` contain only configurations, metrics and training logs.
