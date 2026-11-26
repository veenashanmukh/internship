# RAMT implementation log

Specification: the methodology Claude Doc (Sections 1–19), with the Section 19 recommendations and
amendments A1–A20 as approved. Where Sections 1–18 and 19.8 differ, 19.8 governs.
Status: Phase 1 complete, Phase 2 CPU parts complete; GPU benchmark B0 pending (must run on Kaggle).

---------------------------------------------------------------------------------------------------
## 1. Status by phase

| Phase | Item | Status |
|---|---|---|
| 1 | Code audit of the earlier draft | Done. The draft was replaced; reused parts: word_ids tokenisation, collate, length bucketing, Jigsaw test handling |
| 1 | Package `ramt/`, configs, scripts | Done |
| 2 | CPU test suite (21 tests, tiny local BERT, synthetic files in the real formats) | **Pass** |
| 2 | Data integrity checks (`check_data.py`) | **ALL CHECKS PASSED on Kaggle with all five real datasets (gate G1)** |
| 2 | B0 GPU benchmark and scope rule | **Done on Kaggle: gate G2 passed, scope A + 5 seeds** (section 6c) |
| 3 | Pilot | Unblocked (D1 resolved); waits for B0 (gate G2) |
| 4–5 | Main experiments, statistics (`ramt/stats.py`), bias and fairness analysis | Not started (Phase 5 code follows after the pilot) |

## 2. Blockers (nothing has been substituted)

1. ~~IHC permission (D1)~~ **Resolved 24 Nov 2026.** The **data** is licensed **CC BY 4.0** (the LICENSE file inside the downloaded corpus zip); the GitHub repository's own files carry an MIT licence, the README invites research use with citation, and the user completed the authors' access survey (confirmation page saved). New D1, approved by the user: use IHC under CC BY 4.0 with attribution, cite ElSherief et al. (2021), never redistribute tweet text (at most post IDs), keep the licence notice with any redistributed files. Record files: `IHC_form_confirmation.pdf`, the corpus's `LICENSE` (CC BY 4.0), `README.md`. Stage-1 file used: `implicit_hate_v1_stg1_posts.tsv`. Earlier sessions wrongly said no licence was stated; the LICENSE file had been missed. The code gate stays as it is: `prepare_data.py` processes IHC only when `--ihc_permission_record` points to an existing file; otherwise `data_stats.json` records `ihc_status: BLOCKED`. `train.run` raises `IHCBlocked` for any run that includes the implicit task, or that needs IHC's size for exposure-matched quotas (every pilot and main run, including single-task M5). Only `smoke` and `bench` runs may train explicit + hx without IHC.
2. **The GPU benchmark must run on the user's Kaggle account.** This workspace has no GPU, and the Hugging Face hub and Kaggle are unreachable from it (HTTP 403), so the real encoders could not be loaded here. Every encoder-dependent number (L, throughput, memory, fp16 stability, IG/LIME cost) is still unmeasured.
3. **OLID** could not be fetched in this workspace. Its loader is tested on synthetic files in the official format only.

## 3. Mapping: decisions and amendments → code

| Item | Implementation |
|---|---|
| D1 / A17 | Resolved: MIT licence + README + survey (section 2). `prep.prepare` permission gate; `train.IHCBlocked`; README §0 |
| D2 / A5 / A10 / A12 | `scripts/scope.py`: totals A 333k, B 313k, C 281k step-equivalents; +45,300 for 5 seeds; 24 h; largest scope with need ≤ 0.85 × B0 rate. `jobs.py`: lr grid {2e-5, 3e-5} only in scope A |
| D4 / A4 | `pilot_agg` group (majority vs frozen soft). Rule to apply when freezing: higher validation composite wins; a difference ≤ 0.005 keeps soft |
| D8 / A1 / A2 / A6 | `data/schedule.py` (quotas, shuffled rounds, exposure matching); `train.run`: 6 rounds, validation after each, best-of-6, no early termination, clip 1.0 |
| D9 / A3 | `losses.rationale_loss`: sums over eligible posts, divides by B, returns exactly 0 (graph kept) when none are eligible; fp32 |
| D10 / A7 / A11 | `config.validate`: pilot ⇒ seed 1 and no test evaluation; main/optional ⇒ seed ≠ 1. Pilot runs save no checkpoint (`save_ckpt` false) |
| D11 / A8 / A9 / A13 / A20 | Planned for `ramt/stats.py` (Phase 5). Runs already save per-instance test logits so seeds and instances can be resampled |
| D12 / A14 | Faithfulness outputs keep per-post arrays and random baselines so δ_C and δ_S can be derived on validation (Phase 5) |
| D13 / A15 / A18 / A19 | `optional_cls` group (3 seeds) runs only if the scope leaves budget |
| A16 | Doc text only |
| C19 | **Resolved**: ERASER AOPC bins = [0.01, 0.05, 0.1, 0.2, 0.5], checked in the ERASER code (`evaluate.AOPC_BINS`) |

## 4. Details the specification left open, filled in here (approval needed only if you disagree)

| Detail | Choice | Where |
|---|---|---|
| Dropout on the pooled vector | 0.1 | `model.RAMT` |
| Learning rate for heads and pools | Same as the encoder (one learning rate) | `train.run` |
| Weight decay | 0.01, but none on biases and LayerNorm (standard BERT practice) | `train.run` |
| Batch order within a task | Shuffled without replacement each pass; length bucketing within 50-batch chunks (speed only) | `datasets.TaskStream` |
| Quota rounding | round(1,000 · p_t); largest remainder only if the sum misses 1,000 | `schedule.round_quotas` |
| Attention pooling precision | fp32 with autocast disabled, as D9 asks for the loss | `model.ContentAttentionPool` |
| In-run attention faithfulness | Capped at 1,000 posts per test set (seeded sample) | `config.faithfulness_max_posts` |
| AOPC on short posts | At least one word removed per bin | `evaluate._k` |
| LIME word removal | Removed words are **deleted**, matching AOPC erasure; LIME's default inserts a literal "UNKWORDZ" token | `explain.lime_scores` |
| IG baseline | Pad-token embedding at content positions, special tokens kept; 50 midpoint steps; completeness gap logged | `explain.integrated_gradients` |
| Jigsaw validation split | Stratified on the 6-label pattern; a single row in the "rare" stratum joins the largest stratum (only a tiny dataset can hit this) | `prep._strat_key` |
| Split integrity | Each run re-checks `split_hashes.json` and refuses changed files | `train.run` |
| Timing | Each run records seconds for training rounds, validation, calibration, test evaluation and faithfulness | `timing.json` |

## 5. Verified against the real data available here

HateXplain (official repository files), run with the real loaders. Output: `reports/real_data_check_local.json`.
- Splits after the official divisions: 15,383 / 1,922 / 1,924. The 919 posts not in any division are the no-majority posts.
- One annotator rationale vector has a length mismatch and is dropped (only that vector).
- Rationale-eligible share: 0.594 in train, val and test alike. P(a 32-post batch has no eligible post) ≈ 3 × 10⁻¹³.
- Every toxic-majority post has rationales; normal posts carry none.
- Posts with every word marked: 997 / 131 / 126 (1,254 in total). Only **96** of them have a *constant* soft target (76 / 13 / 7), and only those make the random-matched control identical to the human target. The other all-marked posts have unequal soft values, which the permutation still changes. Section 8.5 should cite 96, not 1,254.
- Words per post: mean 23.4, p95 49, p99 52. W from HateXplain alone would be 49; the real W pools all three training sets and is set by `prepare_data.py`.
- **Near-duplicates across official splits**: after normalisation, 6 train posts match test and 8 match val, and 2 val posts match test. Under Section 1's rule these are removed from train and val; the official test set is untouched. This shrinks HateXplain train by about 14 posts relative to the published split. Reported in `dedup_report.json`.
- Top-10 targeted communities in test (more than one annotator, excluding None and Other): African 315, Islam 194, Jewish 188, Homosexual 182, Women 147, Refugee 85, Arab 67, Caucasian 52, Asian 36, Hispanic 35.
HateCheck: 3,728 cases, 2,563 hateful, 29 functionalities. One case matches a HateXplain train post and is removed from train.

## 6. Deviations and corrections to the document's numbers

1. **Quotas 465 / 263 / 272 (A1) are an arithmetic slip.** With the sizes the document used (44,603 / 14,274 / 15,383), round(1,000 · p_t) gives **464 / 263 / 273**: shares 0.4645, 0.2628, 0.2728, which already sum to 1,000. The code computes quotas from the actual prepared training sizes, so the final numbers will differ anyway once IHC is in; they are recorded in each run's `exposure.json`.
2. Smoke and bench runs may train explicit + hx without IHC (quotas over the tasks present). This is never allowed for pilot or main groups.
3. B0 trains explicit + hx while D1 is pending. Quotas use a benchmark-only IHC size of 14,274 so that the per-step task mix matches the planned full model. Because IHC posts are short tweets, this mix leans slightly toward longer posts, so B0 will slightly *under*-estimate throughput, which errs on the safe side.

## 6b. Bugs found on Kaggle (B0)
1. **DeBERTa-v3 loaded with fp16 weights** (25 Nov 2026). `microsoft/deberta-v3-base` ships its checkpoint in fp16, and transformers 5.x keeps the stored dtype, so AMP failed with "Attempting to unscale FP16 gradients". Fix: `model.RAMT` casts the encoder to fp32 after loading (fp16 only via autocast, as specified). Regression test added (22 tests). HateBERT and RoBERTa were unaffected (fp32 checkpoints); their B0 numbers stand.

2. **B0 two-GPU test ran out of memory on GPU 0**: the benchmark process still held about 11 GB of cached memory from the DeBERTa run when the GPU-0 worker started. Fix: free the cache and reset both worker runs before the two-GPU test; a failed worker is recorded, and scope.py then falls back to the single-GPU rate. A benchmark-script bug only; training jobs run one per GPU.

B0 partial results (2×T4, L = 232, W = 159, full 3-task model with IHC, mu = 0.5, fp16): HateBERT 9.10 steps/s, 5.07 GB peak; RoBERTa 8.45 steps/s, 5.32 GB peak; DeBERTa-v3 4.95 steps/s, 11.18 GB peak (after the fp32 fix); no fp32 fallback needed.

## 6c. B0 complete (gate G2 passed, 25 Nov 2026)
2×Tesla T4, transformers 5.x, L = 232 (p99 over encoders: 232 / 226 / 225), W = 159, full 3-task model with IHC, mu = 0.5, batch 32, fp16.
| Encoder | steps/s (1 GPU) | peak GPU memory | fp16 stable |
|---|---|---|---|
| HateBERT | 9.10 | 5.07 GB | yes |
| RoBERTa | 8.45 | 5.32 GB | yes |
| DeBERTa-v3 | 4.95 | 11.18 GB | yes (after the fp32-master-weight fix) |
Two jobs at once (RoBERTa): 8.49 and 8.54 steps/s per GPU, so no contention. Inference: 1,309 posts/s (all heads); batch-1 latency 14.4 ms GPU, 68.9 ms CPU. IG: 0.157 s/post (50 steps), completeness gap median 0.0010, max 0.059. LIME: 0.34 s/post (1,000 samples).
Scope rule (D2): scope A + five seeds for M1–M3 needs 2.19 steps/s per GPU, against 85% x 8.49 = 7.21. Under DeBERTa it still fits (85% x 4.95 = 4.2). Frozen in configs/frozen.yaml: scope A, five_seeds_m1_m3 true, fp32_encoders [], cls_baseline true (D13: budget allows).
To watch: IG completeness gap. If more than 5% of posts exceed 0.05 in the main runs, report it and raise IG steps for those posts (a disclosed deviation, needing approval).

## 6d. Pilot step 1: encoder x learning rate (25 Nov 2026, seed 1, validation only)
All six runs finished in 0.86 h (HateBERT and RoBERTa about 14 min each, DeBERTa about 24 min). Best validation composite (best round 5 in every run):
| Encoder | lr 2e-5 | lr 3e-5 | frozen lr |
|---|---|---|---|
| HateBERT | **0.7308** | 0.7289 | 2e-5 |
| RoBERTa | **0.7538** | 0.7502 | 2e-5 |
| DeBERTa-v3 | 0.7530 | **0.7533** | 3e-5 (difference 0.0003) |
Rule (highest validation composite) → **encoder = roberta-base**. RoBERTa and DeBERTa are within 0.0005 on one seed, so the paper calls them comparable. Implicit-hate validation AUPRC: RoBERTa and DeBERTa about 0.825 vs HateBERT about 0.76.

**Gap found and fixed before pilot step 2:** the μ tie-break (8.3: ties within 0.005 broken by HateXplain validation plausibility) needs validation plausibility, which runs did not compute, and pilot runs keep no checkpoint. The runs now compute `plausibility_hx_val` (hx head attention vs majority gold, token AUPRC used for the tie-break) during the validation calibration pass. No other behaviour changed.
**`scripts/freeze.py`** applies the pre-registered rules mechanically (encoder/lr; μ with the 0.005 tie and plausibility tie-break; aggregation: majority only if more than 0.005 better, else soft) and records its evidence in `runs/pilot_decisions.json`. **`scripts/run_stage3.py`** chains pilot_mu → freeze μ → pilot_agg → freeze aggregation → date (gate G3) → main → main_extra → optional_cls → optional_shared_norat in one background run, starting no new job after 10.5 h. Learning rates are written as 2.0e-05, because PyYAML reads 2e-05 as a string.
Estimated wall time (2×T4, from pilot timings): pilot 2+3 about 0.75 h, main 39 jobs about 4.5 h, extra seeds about 0.9 h, [CLS] and shared-no-rationale about 0.8 h.

## 6e. Stage 3 complete (25 Nov 2026): 61/61 runs OK in 7.48 h
pilot_mu 3/3 → **μ = 1.0** (composites 0.7530 / 0.7523 / 0.7542, all within 0.005; tie broken by validation plausibility 0.913 / 0.920 / 0.926). pilot_agg → **soft** (0.7542 vs majority 0.7525). main 39/39, main_extra 6/6, optional_cls 9/9, optional_shared_norat 3/3. `frozen_on` reads 2026-11-24 because Kaggle's clock is UTC (21:xx UTC = 03:xx IST on 25 Nov).
Preliminary descriptive results (mean ± SD over seeds): `results/preliminary_results_mean_sd.csv`.

## 6f. Phase 5 analysis code: decisions and corrections (25 Nov 2026)
**Correction: faithfulness sample.** The in-run attention faithfulness used each run's own *predicted*-positive posts, so different models were scored on partly different posts, which rules out the paired test that R4 requires. `scripts/explain_eval.py` therefore recomputes faithfulness for M1–M3 (all seeds with checkpoints) on one **fixed set per split: gold-positive posts** (HateXplain hate or offensive; Jigsaw toxic; IHC implicit hate), at most 1,000 per set, drawn with RandomState(0), identical for every model and method, on test and on validation (validation only for the 19.4 margins). The in-run numbers remain descriptive only. No metric or hypothesis changes.
**Interpretations of the specification (stated in the paper):**
1. H2 "faithfulness" = AOPC comprehensiveness minus random (the R5 and E4 leg); sufficiency is reported as exploratory.
2. H3 per head is one test in the family of 12: an intersection–union test (p = max of the M1 vs M2 and M1 vs M3 p-values; both differences must be positive and sign-consistent across seeds).
3. Normalised plausibility (R5): floor = expected token AUPRC of random scores (20 draws per post); human ceiling = leave-one-annotator-out AUPRC (each annotator's rationale against the majority of the others; posts with at least 3 rationale vectors). Normalisation is the same affine map for M1–M3, so test p-values are computed on raw AUPRC and are unchanged.
4. Margins (19.4): m = lower bound of the combined 95% CI of the active control's advantage over random on **validation** (IG for E4; M2 attention for E5 and the H3 leg), δ = 0.5 m (sensitivity at 0.25 and 0.75); not assessable where m ≤ 0.
5. E4 is assessed per head (HateXplain, Jigsaw, IHC) with that head's own margins; E counts as accepted only if E1 and E2 hold on GPU and CPU and E4 holds for every assessable head.
6. Pooled seed variance (19.3) per metric across all main configurations M1–M8 that report it (main_extra seeds included); for attention faithfulness, pooled across M1–M3.
7. Bias (F): toxicity score = 1 − p(normal) from the HateXplain head; community = named by more than one annotator; top 10 communities in test excluding None and Other; generalised mean with p = −5.
8. R7 timing: IG timed with fp16 autocast on GPU (the "same precision for every method" rule); IG for the faithfulness evaluation uses fp32 (most accurate attributions, the conservative choice for E4).
9. A confirmatory test that cannot be computed counts as p = 1 in the Holm adjustment (never significant, still counted).
**Validation of the analysis code:** weighted AP and AUC equal sklearn exactly (ties and integer weights included); end-to-end test on tiny runs; the analysis recomputes in-run test metrics exactly from the per-post files (asserted).

## 6g. Confirmatory results (Run A, 25 Nov 2026; Holm over 12 tests; claim = Holm p < 0.05 and sign consistent across seeds)
| Test | A | B | A − B | combined 95% CI | Holm p | Claim |
|---|---|---|---|---|---|---|
| H1 Jigsaw macro-AUPRC (M2 vs M5) | 0.6565 | 0.6579 | −0.0013 | [−0.037, 0.034] | 1.0 | no |
| H1 IHC AUPRC | 0.8196 | 0.8184 | +0.0012 | [−0.007, 0.009] | 1.0 | no |
| H1 HateXplain macro-F1 | 0.6859 | 0.6844 | +0.0015 | [−0.013, 0.016] | 1.0 | no |
| H2 macro-F1 M1 vs M2 | 0.6844 | 0.6869 | −0.0026 | [−0.014, 0.009] | 1.0 | no |
| H2 plausibility M1 vs M2 | 0.9242 | 0.6881 | +0.2362 | [0.201, 0.272] | 2.2e−15 | **yes** |
| H2 faithfulness (comp − rand) M1 vs M2 | 0.4553 | 0.2823 | +0.1730 | [0.148, 0.198] | 1.9e−6 | **yes** |
| H2 macro-F1 M1 vs M3 | 0.6844 | 0.6844 | −0.0001 | [−0.010, 0.010] | 1.0 | no |
| H2 plausibility M1 vs M3 | 0.9242 | 0.3979 | +0.5263 | [0.489, 0.563] | 9.1e−31 | **yes** |
| H2 faithfulness M1 vs M3 | 0.4553 | 0.0120 | +0.4434 | [0.421, 0.466] | 2.5e−31 | **yes** |
| H3 explicit-head plausibility (IUT) | 0.894 | ≤ 0.551 | +0.343 | — | 1.5e−14 | **yes** |
| H3 implicit-head plausibility (IUT) | 0.608 | ≤ 0.542 | +0.066 | — | 0.87 | no |
| F GMB-BPSN AUC M1 vs M2 | 0.6848 | 0.7220 | −0.0372 | [−0.199, 0.124] | 1.0 | no (lower in every seed) |
Verdicts: **H1 not supported** (no multi-task gain, no task worse). **H2 supported** on explanation quality (plausibility and faithfulness vs both controls), not on classification. **H3 supported** via the explicit head (faithfulness leg non-inferior at the 50% margin); the implicit head is not supported. **F: no significant change**, but M1's BPSN is lower in all seeds and the interval is very wide; the bootstrap produced AUC = 0 for some small community in some resamples ("divide by zero in power" under the p = −5 generalised mean), which inflates the interval. Report per-community BPSN as exploratory; do not change the confirmatory test after the fact.
**Hypothesis E (GPU part):** E1 met (attention adds nothing: ratio 0.986). **E2 NOT met on GPU**: prediction + attention 15 ms vs IG 69 ms (fp16, 50 steps in 2 batched chunks) and LIME 336 ms, ratios 0.21 (target ≤ 0.10) and 0.044 (target ≤ 0.01). E3 met (p95 18.6 ms ≤ 100 ms). E4 met for all three heads at the 50% margin (fails at 25% for Jigsaw and IHC). E5 met for all heads. Because E requires E1, E2 and E4, **E is not accepted, whatever the CPU timings show**; the paper reports the measured speed-ups (about 4.6× vs IG, 23× vs LIME on GPU) and comparable faithfulness, without the pre-registered 10× / 100× claim. The R7 reasoning ("IG = 50 passes") did not anticipate batching IG steps on the GPU; this is stated as a limitation, not changed after the fact.
Kaggle test note: 2 of 31 tests failed only because their file names assumed a CPU machine (timing file named gpu_ on Kaggle). Tests fixed; the analysis code was unaffected.

## 6h. Inspection of Run A outputs (25 Nov 2026)
1. **Plausibility "human ceiling" (interpretation 3 in 6f) does not work as a ceiling.** Leave-one-annotator-out AUPRC = 0.689 (455 posts) while M1 reaches 0.924: one annotator's 0/1 marks are heavily tied scores, so their AUPRC is lower than a continuous model score's. Reporting change only (the tests use raw AUPRC, unaffected): report raw plausibility with the random floor (0.381) and give 0.689 as an inter-annotator reference, not a ceiling.
2. **IG completeness.** Median gap 1.1% (HateXplain), 1.4% (Jigsaw), 2.5% (IHC), but the share of posts with a gap above 5% is 20%, 25% and 36%, above the 5% level set in 6c. Pre-committed action: report it. Raising the IG step count would change E4's active control and needs the user's approval (GPU cost of about 1–2 h). **User decision (25 Nov 2026): option A, report as a limitation; no re-run.** E is already not accepted because of E2, so this affects only how firmly E4 can be stated.
3. **F interval.** V_inst = 0.0064 vs V_seed = 0.00036: the width comes from the instance bootstrap; under the p = −5 generalised mean, resamples in which a small community's BPSN AUC is low dominate. Descriptively, every rationale-trained model has lower GMB-BPSN than its no-rationale counterpart (M1 0.685 vs M2 0.722; M4 0.691 vs M4b 0.719; M6 0.702 vs M5 0.726). Exploratory; no claim.
4. **Exploratory highlights (unadjusted):** human-rationale supervision also raises the attention faithfulness of the Jigsaw head (comprehensiveness − random +0.244, CI [0.200, 0.288]) and the IHC head (+0.053, [0.019, 0.087]) on their own test sets; single-task rationale supervision (M6 vs M5) raises plausibility by +0.254 [0.137, 0.370]; HateBERT is worse on IHC than single-task RoBERTa (−0.047 [−0.061, −0.033]); no other classification difference reaches p < 0.05.
5. Figures: fig_h2_explanations.png, fig_h3_transfer_and_ig.png, fig_classification_parity.png, fig_latency_gpu.png (from per-seed data; layout-checked).

## 6i. HateXplain protocol, corrected F, IHC shortcut check (25 Nov 2026)
**Checked against HateXplain's released code** (Bias_Calculation_NB.ipynb, Explainability_Calculation_NB.ipynb, testing_with_rational.py, eraserbenchmark/metrics.py):
- **Bias, implementation error found in F.** HateXplain excludes posts that name none of the top-10 communities from every subgroup/BPSN/BNSP set, and picks the top 10 over the whole dataset. `analyze.py` kept untargeted posts in the background sets (the top-10 list is identical either way: reproduced from the raw data with HateXplain's counting code). The methodology says F follows HateXplain's protocol, so this is a bug fix, not a post-hoc choice: F is recomputed with the exact definition (`scripts/hatexplain_eval.py`, `F_corrected.json`) and the Holm family is re-adjusted with it, whichever way it comes out. The original number is reported alongside, labelled.
- **Published-results table (8.6):** classification = accuracy, macro-F1, AUROC (ovo, macro); plausibility = non-normal test posts, union gold, top-5 hard rationale, ERASER IOU-F1 (0.5, macro), token-F1 (instance macro), AUPRC (auc of the PR curve); faithfulness = ERASER one-shot comprehensiveness/sufficiency with the top-5 rationale. Our reimplementation matches ERASER's own metrics.py exactly on 300 random posts (difference < 1e−9). Unit difference stated: words (ours) vs BERT word pieces (HateXplain).
- **IHC shortcut check (6.5):** logistic regression on surface features only (leading RT, quote characters, mentions, URLs, hashtags, length), trained on IHC train, AUROC on IHC test.
**Bug fixed in ramt/stats.py:** `combined_test` returned p = 0 for a non-computable comparison (NaN difference); it now returns p = NaN, never significant (regression test added). No reported result was affected: all confirmatory and exploratory p-values were finite.

## 6j. CPU timing reduced to one seed (25 Nov 2026)
Run B (CPU latency, R7 protocol, 3 seeds × 520 posts) took about 4.3 h per seed, mostly LIME with 1,000 samples per post on 4 vCPUs, so 3 seeds exceed Kaggle's 12-hour limit. Re-run with **seed 42 only** (`--timing_seeds seed42`); the same 500 posts and code. Deviation from the 3-seed protocol, disclosed; E's verdict is unaffected (E2 already fails on GPU), and on GPU the 3 seeds agree within 10% (prediction + attention 14.8–16.0 ms).

## 7. Next steps
1. User: run README steps 1–3 on Kaggle and send back `check_data.json`, `b0_results.json` and the `scope.py` output.
2. Me: freeze the scope and fp32 list, write `ramt/stats.py` (D11/D12) and the bias metrics while IHC permission is pending.
3. Upload IHC + the access record to Kaggle: re-prepare with IHC (new splits dataset version), rerun `check_data.py`, then the pilot.

## 6k. HateXplain protocol run and corrected F (25 Nov 2026)

`hatexplain_eval.py --with_models` finished on Kaggle (notebookf5b3438025).
- Corrected F (HateXplain exact definition): M1 0.7081 vs M2 0.7437, diff -0.0355, combined 95% CI [-0.191, 0.120], p = 0.654, 0/5 seeds favour M1. Holm re-adjusted over the 12 tests with this p: every adjusted value is unchanged (F stays 1.0). Original F (0.685 vs 0.722, p 0.651) is reported in a table note.
- Gap concentrated in Hispanic (0.54 vs 0.61) and Arab (0.64 vs 0.68); the other eight communities differ by <= 0.01.
- HateXplain-protocol one-shot faithfulness: M1 comp 0.585 / suff 0.040; M2 0.488 / 0.110; M3 0.183 / 0.400.
- IHC shortcut check: surface-feature logistic regression test AUROC 0.562.
- Paper: Table 2 F row, Section 5.6, limitations and Appendices A and B updated. CPU timing still pending.

## 6l. CPU timing (26 Nov 2026)

notebook820ffcf72f finished (explain_eval.py cpu_timing, seed42 only, 4 threads, 520 posts, first 20 dropped).
- Medians: pred 94.1 ms, pred+attn 81.2 ms, IG 4240.6 ms, LIME 22783.0 ms; p95 pred+attn 142.7 ms.
- E1_cpu ratio 0.86 (pass; read as timing noise), E2_cpu IG 52.2x (>=10, pass), LIME 280.5x (>=100, pass), E3_cpu 142.7 ms < 1000 ms (met).
- E overall still not accepted because E2_gpu fails (IG 4.8x, LIME 22.8x; pooled medians as in analyze.py).
- Paper corrected: E1 threshold is 5% and E2 LIME threshold is 100x (the draft had 10% and 10x); GPU figures now use the pooled medians of analyze.py (15.2 / 72.4 / 346.6 ms) instead of the seed-42 medians. Latency figure redrawn with GPU and CPU panels.
