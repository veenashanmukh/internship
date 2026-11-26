# RAMT Research Methodology — Second-Stage Critical Review (v2, for approval)

Nov 24, 2026 · @vene

## How to read this review

**Verdict:** the v1 RAMT proposal can be submitted to EAAI only after these changes. It needs a falsifiable claim about rationale transfer, proper controls for the rationale signal, an honest treatment of label-construct mismatch between the datasets, and a compute plan measured on Kaggle rather than estimated. No code is written until you approve Section "Consolidated methodology".

Every statement carries one tag:

- **\[V\]** verified on a primary or official page (linked in Sources).
- **\[A\]** an assumption that the first Kaggle notebook must check before we rely on it.
- **\[R\]** my recommendation (a design decision, not a fact).

Every issue is marked **Essential** (a reviewer can reject the paper without it) or **Optional** (strengthens the paper if budget allows).

**What changed from v1, in one list:**

1. The main claim narrows to three testable hypotheses (H1 multi-task benefit, H2 human-rationale benefit beyond a random control, H3 transfer of rationale behaviour to heads without rationales).
2. A matched random-rationale control and an aggregation comparison are added; the mean-pooling, CLS-pooling and μ-sweep runs are cut or moved to the validation pilot.
3. Leave-one-task-out runs replace the separate "no implicit / no HateXplain" ablations and give the negative-transfer matrix at no extra cost.
4. Classification heads become a single linear layer; the extra MLP is removed.
5. Cross-dataset results are reported threshold-free (AUC) first, because the datasets measure different constructs.
6. Fairness gets a specific hypothesis: rationale supervision may *increase* identity-term false positives, because human rationales often mark identity words.
7. Runtime numbers in v1 were estimates. They are removed; a measured benchmark notebook now gates the experiment list.

## 1. Datasets: compatibility, labels, quality, licences, splits

The five datasets are usable, but they measure **four different constructs** on **three platforms**, and one of them (IHC) has no official split and an unstated data licence. The table holds what I could verify; anything not in it is marked \[A\] below.

| Dataset | Platform | Label construct | Annotation | Licence | Official split | Role |
| --- | --- | --- | --- | --- | --- | --- |
| [Jigsaw Toxic Comment](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes) | Wikipedia talk pages \[V\] | 6 binary toxicity labels; 1 = majority of raters assigned it \[V\] | Crowd raters; agreement not published on the pages I opened \[A\] | CC0; comment text under Wikipedia CC-BY-SA 3.0 \[V\] | train.csv; test.csv + test\_labels.csv, where −1 = not used for scoring \[V\] | Explicit head |
| [IHC / Latent Hatred](https://arxiv.org/pdf/2109.05322) | Twitter, US extremist groups \[V\] | explicit hate / implicit hate / not hate (stage 1) \[V\] | 3 MTurk workers per tweet; majority agreement on 95.3% \[V\] | Repo is MIT; the data licence is not stated; download requires a survey \[V\] | None published \[A: confirm in the download\] | Implicit head |
| [HateXplain](https://arxiv.org/pdf/2012.10289) | Twitter + Gab \[V\] | hate / offensive / normal by majority; 919 no-majority posts dropped \[V\] | MTurk; rationales from 2–3 annotators, only for hate/offensive posts \[V\]; agreement α = 0.42 labels, 0.49 rationales \[V, secondary source\] | GitHub repo: MIT; HF card metadata says CC-BY-4.0 \[V: conflicting\] | 8:1:1 via post\_id\_divisions.json; 15,383 / 1,922 / 1,924 on HF \[V\] | 3-class head + rationales |
| [OLID](https://arxiv.org/pdf/2203.11331) | Twitter \[V\] | offensive vs not (level A) \[V\] | Crowd \[V\] | Not verified \[A\] | 13,240 train / 860 test; test = 240 OFF, 620 NOT \[V\] | Zero-shot test only |
| [HateCheck](https://huggingface.co/datasets/Paul/hatecheck) | Synthetic templates \[V\] | hateful vs non-hateful, 29 functional tests \[V\] | Expert-written, crowd-validated \[V\] | CC-BY-4.0 \[V\] | 3,728 test cases \[V\] | Zero-shot + fairness |
| [Civil Comments identities](https://huggingface.co/datasets/jigsaw_unintended_bias/blob/main/README.md) (new) | Civil Comments platform \[V\] | toxicity score + identity mentions \[V\] | Crowd \[V\] | CC0 \[V\] | test\_public\_leaderboard 97,320 rows \[V\] | Fairness only (optional) |

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 1.1 Construct mismatch: "toxic" (Jigsaw) ≠ "hate" (HateXplain, IHC) ≠ "offensive" (OLID). | A drop in cross-dataset F1 may reflect a different definition, not a weaker model. Reviewers know HateBERT's own authors found portability depends on [compatibility of the annotated phenomena](https://preview.aclanthology.org/author-url/2021.woah-1.3/) \[V\]. | Add a label-semantics table to the paper. Make **threshold-free AUC** the primary cross-dataset metric; report F1 only as secondary, with the source-validation threshold. Never call it "generalization" without naming the construct. | Essential |
| 1.2 IHC has no published split and an unstated data licence. | Results are not comparable to prior work unless the split is published; redistributing tweets may breach terms. | Stratified 70/10/20 split with a fixed seed, after deduplication. Release **post IDs + split script only**, never text. State the access route (survey) in the paper. | Essential |
| 1.3 HateXplain licence conflict (MIT vs CC-BY-4.0). | A reviewer or editor may ask; ambiguity is easy to remove. | Cite both statements and comply with the stricter one (attribution). | Essential (one sentence) |
| 1.4 Jigsaw test rows labelled −1 are not ground truth. | Including them corrupts every test metric. | Keep only rows with all six labels ≥ 0. Report the resulting test size. | Essential |
| 1.5 Training subsample (all toxic + 30k clean) shifts the class prior. | Probabilities become miscalibrated; thresholds tuned on a natural distribution compensate, but ECE on the test set will look worse. | Validation and test keep the natural distribution. Report the subsampling ratio and include "no subsampling" as one optional ablation if time allows. | Essential (reporting) |
| 1.6 HateXplain, IHC and OLID are all Twitter; the same tweet can appear in two datasets. | A test tweet seen in training under another label inflates cross-dataset scores. | Normalise text (lower-case, strip @user / URL / punctuation) and remove exact duplicates across all datasets before splitting; drop any training row matching any validation or test row. Publish the counts. | Essential |
| 1.7 Low human agreement on HateXplain rationales (α = 0.49). | Plausibility scores have a ceiling well below 1; a reviewer will ask what "good" means. | Compute a **human ceiling**: score each annotator's rationale against the aggregate of the others on the test set, and report it beside every model's plausibility. | Essential |
| 1.8 Different placeholder conventions (@USER, URL, \<user>). | Tokens that mark a dataset become shortcuts and inflate domain separability. | Map all of them to one form (@user, http) in every dataset before tokenisation. | Essential |
| 1.9 IHC explicit-hate posts are excluded from the implicit task. | They are still useful evidence. | Use them only as a probe: report the share of IHC explicit-hate posts flagged by the explicit head and by the implicit head. | Optional |

## 2. Multi-task training strategy

v1 named the tasks but not how they share a training step. \[R\] Final design: **one task per batch, square-root task sampling, equal task weights, and negative transfer measured rather than assumed away.**

**Batch construction.** Each mini-batch of size B = 32 comes from one dataset only. A task t is drawn at every step with probability

```latex
p_t = \frac{n_t^{1/T}}{\sum_{s \in \mathcal{T}} n_s^{1/T}}, \qquad T = 2
```

where n\_t is that task's training size and 𝒯 the enabled tasks. Examples inside a task are drawn without replacement and reshuffled when exhausted. One "epoch" = S = Σ\_t ⌈n\_t / B⌉ steps, so every configuration sees the same number of updates per epoch.

**Loss of one step** (task t, batch ℬ):

```latex
\mathcal{L} = \lambda_t \, \mathcal{L}_t(\mathcal{B}) + \mathbb{1}[t = \mathrm{hx}] \; \mu \, \mathcal{L}_{\mathrm{rat}}(\mathcal{B}), \qquad \lambda_t = 1
```

- Explicit: mean binary cross-entropy over 6 labels, positive weight w\_k = clip(√(neg\_k / pos\_k), 1, 10) from training counts.
- Implicit: the same with one label.
- HateXplain: 3-class cross-entropy, no class weights.
- 𝓛\_rat is defined in Section 3.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 2.1 Mixed batches need label masks across tasks; masking bugs caused the old pipeline's errors. | Silent mask errors corrupt both training and evaluation. | One-task batches: each head receives loss only from its own dataset. The only masks left are padding and missing rationales (Section 3). | Essential |
| 2.2 Proportional sampling lets Jigsaw dominate; uniform sampling repeats the small sets many times per epoch. | The first starves the implicit and HateXplain heads; the second overfits them. | Square-root sampling (T = 2) as above. \[A\] With roughly 46k / 16k / 15k training rows, the shares become about 0.46 / 0.27 / 0.26 instead of 0.60 / 0.21 / 0.19; recompute after data prep. | Essential |
| 2.3 Loss scales differ (6-label BCE vs 3-class CE). | Fixed weights may still favour one task. | Keep λ\_t = 1 and let sampling balance tasks; log per-task loss each epoch. Learned uncertainty weighting is **not** used, because it adds a moving part that reviewers would want ablated. | Essential (fixed choice) |
| 2.4 Negative transfer is assumed not to happen. | If multi-task hurts a task, the central claim fails silently. | Measure it directly: single-task runs and leave-one-task-out runs (Section 6) give a 3 × 3 transfer matrix with bootstrap CIs. | Essential |
| 2.5 Checkpoint selection on an average score can sacrifice one task. | The best average can hide a task that got worse. | Select on the mean of per-task validation scores (explicit macro-AUPRC, implicit AUPRC, HateXplain macro-F1), **and** log every task's curve so a regression is visible. | Essential |
| 2.6 Mitigation of conflict between task gradients (e.g. gradient projection). | Only needed if 2.4 shows negative transfer. | Not in the main design. If the transfer matrix shows a significant loss, add one gradient-projection run as a follow-up. | Optional |
| 2.7 v1 froze the bottom 3 encoder layers without evidence. | An unjustified choice is an easy reviewer question. | Fine-tune all layers. Removes a hyperparameter. | Essential (simplification) |
| 2.8 Heads were 2-layer MLPs. | Extra capacity per task confounds the comparison with single-task baselines. | One linear layer per task after pooling. | Essential (simplification) |

## 3. Rationale alignment: exact formulation

The alignment is a cross-entropy between a word-level human rationale distribution, spread evenly over sub-word tokens, and the HateXplain head's attention-pooling weights over content tokens. The definitions below fix every choice v1 left open.

**Step 1 — Aggregate annotators (word level).** A post has words w₁ … wₙ. Let 𝒜\_r be the annotators who supplied a rationale vector r⁽ᵃ⁾ ∈ {0,1}ⁿ. \[V\] HateXplain collected rationales only for posts labelled hateful or offensive, from 2–3 annotators per post, and not always from the annotators who gave the class label. Default aggregation \[R\] is the soft mean:

```latex
\rho_i = \frac{1}{|\mathcal{A}_r|} \sum_{a \in \mathcal{A}_r} r^{(a)}_i \in [0,1]
```

Alternatives tested in Section 5: majority 1\[ρᵢ ≥ 0.5\], union 1\[ρᵢ > 0\], intersection 1\[ρᵢ = 1\].

**Step 2 — Align to sub-words.** The tokenizer maps sub-word position j to a word index ω(j), or to ∅ for special and padding tokens. With Sᵢ = { j : ω(j) = i }, each word's mass is split evenly over its pieces:

```latex
q_j = \begin{cases} \rho_{\omega(j)} \,/\, |S_{\omega(j)}| & \omega(j) \neq \varnothing \\ 0 & \text{otherwise} \end{cases}
\qquad \tilde q_j = \frac{q_j}{\sum_k q_k}
```

Splitting (rather than copying ρ to every piece) keeps the word-level target identical for WordPiece (HateBERT), byte-level BPE (RoBERTa) and SentencePiece (DeBERTa), which Section 7 needs.

**Step 3 — Attention over content tokens only.** Let V = { j : not padding, not special }. For encoder states hⱼ:

```latex
s_j = \mathbf{v}^\top \tanh(W h_j + b), \qquad \alpha_j = \frac{\exp s_j}{\sum_{k \in V} \exp s_k} \;(j \in V), \qquad z = \sum_{j \in V} \alpha_j h_j
```

**Step 4 — Loss.** Over the batch subset ℬ\_r of eligible posts:

```latex
\mathcal{L}_{\mathrm{rat}} = \frac{1}{|\mathcal{B}_r|} \sum_{x \in \mathcal{B}_r} \Big( - \sum_{j \in V} \tilde q_j \log(\alpha_j + \epsilon) \Big), \qquad \epsilon = 10^{-8}
```

This equals KL(q̃ ‖ α) plus the constant entropy of q̃, so the gradients are the same. A post is eligible when its majority label is hate or offensive, 𝒜\_r is non-empty, and Σ qⱼ > 0 after truncation.

**Step 5 — Word-level explanation** read from the model for evaluation: eᵢ = Σ\_{j ∈ Sᵢ} αⱼ.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 3.1 How sub-words inherit word rationales was undefined. | Copying the label to every piece gives long words more target mass, and the bias differs by tokenizer. | Even split (Step 2); word-level read-out (Step 5). | Essential |
| 3.2 Aggregation of 2–3 annotators was undefined. | Choice changes the target and plausibility scores; reviewers ask for it. | Soft mean by default; three alternatives as a small validation-only comparison (Section 5). | Essential |
| 3.3 Attention could sit on \[CLS\], \[SEP\] or padding. | Mass on special tokens is a common attention sink and makes explanations meaningless. | Exclude padding and special tokens from the softmax (Step 3); assert it in code. | Essential |
| 3.4 Posts without a rationale (normal posts, truncated rationales). | Forcing a target on them is unjustified; silently using an empty target produces NaN. | No rationale loss for them (eligibility rule in Step 4). Report how many test posts lose rationale words to truncation. | Essential |
| 3.5 Rationale annotators are not always the label annotators \[V\]. | A rationale may justify a label the majority did not give. | Use rationales only when the post's majority label is hate or offensive; mention the limitation. | Essential (reporting) |
| 3.6 log of very small attention values under fp16. | Numerical overflow or underflow destabilises training. | Compute scores, softmax and 𝓛\_rat in fp32 inside the autocast region; ε = 1e-8. | Essential |
| 3.7 μ chosen without a rule. | Picking μ on test results is leakage. | Choose μ ∈ {0.25, 0.5, 1.0} on validation only, one seed, in the pilot; freeze before the main runs. | Essential |
| 3.8 Normal posts get no constraint, so the model may attend anywhere on them. | Explanations for "normal" predictions stay unconstrained. | Accept this; evaluate explanations only for toxic predictions and say so. Uniform-attention targets for normal posts are not used, because they have no human basis. | Essential (reporting) |

## 4. Can rationale information reach Jigsaw and IHC?

Not by construction. With task-specific attention pools, the rationale loss updates only the HateXplain pool and the shared encoder; the explicit and implicit pools receive no rationale signal at all. Transfer is therefore a hypothesis (H3) that may fail, and the paper must test it rather than claim it.

**Where the gradient goes.** ∂𝓛\_rat reaches (a) the HateXplain pool parameters W, b, v and (b) every encoder state hⱼ. Only (b) is shared. The explicit head can benefit only if the encoder starts representing "rationale-like" words differently, and its own pool then learns to weight them.

**What cannot be measured.** \[V\] Jigsaw has labels only, and IHC's extra annotations are free-text target groups and implied statements, not token rationales. So there is no gold rationale for Jigsaw or IHC posts.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 4.1 v1 implied that rationales improve explanations on all tasks. | An untested transfer claim is the first thing a careful reviewer attacks. | State H3 explicitly. It is supported only if the tests below improve over **both** the no-rationale model and the random-rationale control, with 95% CIs excluding zero. Otherwise report it as a negative result. | Essential |
| 4.2 No gold rationales on Jigsaw or IHC. | Plausibility cannot be computed there. | **Cross-head plausibility:** run the explicit and implicit heads on HateXplain test posts and score *their* attention against the human rationales. If those heads align better only when the HateXplain head was rationale-supervised, the signal travelled through the encoder. | Essential |
| 4.3 Transfer might show in faithfulness, not plausibility. | A model can rely on its highlighted words more without matching human choices. | Compute comprehensiveness and sufficiency (Section 9) for explicit-head explanations on Jigsaw test and implicit-head explanations on IHC test, for each condition. No gold data needed. | Essential |
| 4.4 Task-specific pools may block transfer that a shared pool would allow. | The answer to H3 depends on this design choice. | Add the **shared-pool** variant: one attention pool for all heads, rationale-supervised. Its no-rationale counterpart completes a 2 × 2 (pool × rationale). | Shared + rationale: Essential. Shared without rationale: Optional |
| 4.5 Encoder-level evidence. | Shows *why* transfer does or does not happen. | Linear probe on frozen token states predicting human-rationale words (train on HateXplain train, test on test) for STL-explicit, MTL without rationales, and RAMT encoders. | Optional |
| 4.6 Domain confound: the explicit head trained on Wikipedia is applied to Twitter/Gab posts in 4.2. | Absolute cross-head scores will be low for every model. | Interpret only differences between conditions on the same posts, never absolute values. | Essential (reporting) |

## 5. Isolating the human-rationale effect (with counterfactual deletion)

Comparing RAMT only against μ = 0 cannot separate "human rationales help" from "any extra attention loss helps". \[R\] The fix is a matched random-rationale control trained with the same μ, plus deletion tests that ask whether the model actually relies on the words humans marked.

**Training-time controls** (all identical to RAMT except the rationale target):

| Condition | Rationale target per eligible post | What a difference from RAMT means | Priority |
| --- | --- | --- | --- |
| No rationale (μ = 0) | none | Effect of the rationale loss at all | Essential |
| **Random-matched** | the post's own soft values ρ permuted to random word positions; same count and mass, fixed per post with seed 0 | Benefit comes from *where* humans pointed, not from sparse attention per se | Essential |
| Lexicon | 1 on words in a public slur/profanity lexicon, else 0 | Human rationales add more than keyword spotting | Optional (\[A\] lexicon and its licence to be chosen) |

**Aggregation comparison** (validation only, 1 seed, chosen before the main runs): soft mean (default) vs majority; union and intersection optional. The winner on validation macro-F1 and validation plausibility is frozen; all four are reported in the appendix.

**Counterfactual deletion on HateXplain test (toxic-majority posts with rationales):**

1. **Human-rationale deletion.** Remove every word with ρᵢ > 0; record the drop in the probability of the originally predicted class.
2. **Matched random deletion.** Remove the same number of words chosen at random from the non-rationale words (5 draws, averaged).
3. **Rationale-only input.** Keep only the words with ρᵢ > 0; record the drop.
4. Effect = (1) minus (2), per post, paired across conditions; bootstrap CI over posts, averaged over seeds.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 5.1 No control for "any attention supervision". | Without it, H2 is not identifiable. | Random-matched control, 3 seeds, same μ and schedule. H2 holds only if RAMT beats it on classification or explanation metrics with CIs excluding zero. | Essential |
| 5.2 Random positions resampled every epoch would act as noise, not as a wrong but consistent signal. | The two controls answer different questions. | Fix the permutation per post (seed 0) for the whole run. | Essential |
| 5.3 Aggregation choice could be tuned on test. | Leakage. | Choose on validation, freeze, report all. | Essential (soft vs majority); Optional (union, intersection) |
| 5.4 Reliance on human-marked words is never measured directly. | Plausibility shows overlap, not dependence. | Deletion protocol 1–4 above for RAMT, no-rationale and random-matched models. | Essential |
| 5.5 Deleting words creates unnatural inputs. | The probability drop may reflect distribution shift, not reliance. | Matched random deletion (step 2) controls for this to first order. As a sensitivity check, replace instead of delete with the tokenizer's mask token. Retraining-based tests are out of budget; state that. | Mask-replace: Optional; limitation sentence: Essential |

## 6. Cross-dataset generalisation, negative transfer and domain shift

In this design every task is one dataset on one platform, so task transfer and domain transfer are perfectly confounded. \[R\] The paper should report a head × test-set AUC matrix, a task-to-task transfer matrix from leave-one-task-out runs, and a measured domain distance, and state the confound as a limitation.

**A. Head × test-set matrix (no extra training).** Every trained model scores every test set with every head it has: explicit p(toxic), implicit p(implicit), HateXplain 1 − p(normal). Columns: Jigsaw test, IHC test, HateXplain test, OLID test, HateCheck. Primary metric AUC; F1 at the source-validation threshold as secondary. The combined rule (max over heads, flag if any head exceeds its threshold) is reported as a separate row, defined before any test result is seen.

**B. Task-to-task transfer matrix.** All runs here use μ = 0 so task transfer is not mixed with the rationale effect. The effect of adding task s on task t is

```latex
E_{s \to t} = M_t(\text{all three}) - M_t(\text{all three} \setminus s), \qquad \Delta_t = M_t(\text{all three}) - M_t(\text{single-task } t)
```

with Mₜ the primary test metric of task t, paired bootstrap CIs, 3 seeds. Needed runs: 3 single-task, 3 pairs (explicit+implicit, explicit+HateXplain, implicit+HateXplain), 1 all-three. Negative transfer = a CI entirely below zero.

**C. Domain shift measurement (CPU only).**

1. Descriptives per dataset: platform, mean and 95th-percentile length, share of @user/http placeholders, vocabulary overlap (Jaccard of the 5,000 most frequent words) for every pair.
2. Separability: a TF-IDF + logistic-regression classifier predicting the *source dataset* of a post (5-fold CV, balanced sample per dataset); report accuracy and the proxy A-distance 2(1 − 2ε), with ε the pairwise error.
3. Relate (descriptively, no significance claim with so few pairs) the pairwise distance to the cross-dataset AUC drop from matrix A.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 6.1 v1 had only OLID and HateCheck as zero-shot tests. | Cross-dataset behaviour of each head is invisible. | Matrix A over all five test sets, computed at the end of every run, so it needs no extra training. | Essential |
| 6.2 v1 used "no implicit" and "no HateXplain" ablations with rationales on. | Mixes task removal with rationale removal; "no explicit" was missing. | Replace them with the 3 pairwise runs at μ = 0 (matrix B). | Essential |
| 6.3 Negative transfer only inferred from averages. | Hides which task hurts which. | Report E\_{s→t} and Δₜ with CIs. | Essential |
| 6.4 Task and domain are confounded. | A reviewer can attribute every effect to platform differences. | State it as a limitation; quantify the domain gap (C); note that the HateXplain and IHC test sets share Twitter as platform, giving one partial within-platform comparison. | Essential |
| 6.5 Shortcut features that identify a dataset. | The old implicit task was solved by recognising the source; any similar shortcut must be ruled out. | Within IHC both classes come from the same source \[V\]; check that a TF-IDF classifier's top features for implicit vs not-hate are not formatting artefacts, and report them. | Essential |
| 6.6 Embedding-based distance (MMD on frozen encoder states). | A second distance estimate strengthens C. | Add if time allows. | Optional |

## 7. Fair comparison of HateBERT, RoBERTa and DeBERTa

The three encoders differ in tokenizer, casing, vocabulary size and pretraining data, so "same hyperparameters" is not automatically fair. \[R\] Fairness here means the same words in, the same tuning budget, and the same evaluation code.

| Encoder | What it is | Size | Licence |
| --- | --- | --- | --- |
| [HateBERT](https://huggingface.co/GroNLP/hateBERT) | BERT-base-uncased further trained on more than 1M posts from banned Reddit communities \[V\] | 110M parameters \[V\] | Apache-2.0 \[V\] |
| RoBERTa-base | General-domain, cased, byte-level BPE | About 125M parameters \[A: read from the loaded model\] | \[A: check model card\] |
| [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | General-domain, ELECTRA-style pretraining, 128K vocabulary \[V\] | 86M backbone + 98M embedding parameters \[V\] | MIT \[V\] |

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 7.1 A 128-sub-word limit truncates different text for each tokenizer. | Encoders would be compared on different inputs. | Truncate at word level to W words first (W = smallest value keeping ≥ 95% of posts whole, measured on training data), then set the sub-word limit to the 99th percentile across all three tokenizers. Report the truncation rate per encoder. | Essential |
| 7.2 One learning rate for all encoders favours whichever it happens to suit. | Reviewers expect an equal tuning budget. | Same small grid per encoder: lr ∈ {2e-5, 3e-5}, μ = 0, 1 seed, validation only. Each encoder keeps its own best lr for all its runs. | Essential |
| 7.3 DeBERTa can be unstable in fp16. \[A: widely reported; not verified on our setup\] | A NaN run would be mistaken for a weak encoder. | Smoke-test DeBERTa in fp16; if loss becomes NaN, run it in fp32 and report the precision used and its runtime separately. | Essential |
| 7.4 Casing: HateBERT lower-cases, the others are cased. | Lower-casing before tokenising would handicap the cased models. | Never lower-case the text; each tokenizer applies its own rules. | Essential |
| 7.5 Encoder choice for RAMT made on test results. | Leakage. | Choose the RAMT encoder on validation (mean of per-task scores) in the pilot. | Essential |
| 7.6 Only one encoder gets the rationale loss. | H2 might hold for one encoder only. | Main table: no-rationale multi-task model for all three encoders (3 seeds) + RAMT on the chosen encoder. RAMT on the other two encoders as a robustness check. | Main: Essential. Robustness: Optional |
| 7.7 Pretraining-domain differences are part of the comparison. | Readers must know HateBERT saw abusive Reddit text. | Describe pretraining data in the table above; interpret differences as "domain-adapted vs general", not as architecture effects. | Essential (reporting) |

## 8. Threshold calibration without test leakage

&#91;R\] Thresholds are chosen per label on validation only, from a bootstrap-averaged F1 curve, while model selection uses threshold-free metrics, so the two steps never feed each other. Pilot runs never compute test metrics.

**Protocol (per run, per seed):**

1. Model selection (early stopping, encoder, lr, μ, aggregation) uses validation AUPRC / macro-F1 only, never a thresholded score.
2. For each explicit label k and for the implicit head: draw 200 bootstrap resamples of the validation set, compute F1 at every threshold τ ∈ {0.01, 0.02, …, 0.99}, average the curves, and take the argmax. Per-label thresholds are valid because macro-F1 decomposes into per-label F1.
3. Fit one temperature per label on validation negative log-likelihood, used **only** for calibration metrics (ECE, reliability diagrams). Per-label temperature is monotone, so it does not change the decision at the chosen threshold.
4. Zero-shot test sets (OLID, HateCheck, cross-dataset cells) use the source task's validation thresholds; no tuning on the target.
5. Test predictions are written once, after the configuration is frozen.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 8.1 v1 code computed test metrics in pilot runs. | Seeing test scores while choosing encoder or μ is leakage, even unintentionally. | Pilot jobs run with test evaluation disabled; only frozen-config jobs write test files. | Essential |
| 8.2 F1-optimal thresholds on rare labels are noisy. \[A\] Jigsaw threat has only a few hundred positives in total, so the validation split holds a few dozen. | A threshold fitted to a few dozen positives can swing widely between seeds. | Bootstrap-averaged F1 curve (step 2). Report each label's threshold as mean ± SD across seeds and its validation positive count. | Essential |
| 8.3 Reusing one validation set for selection and thresholds biases results upward. | Optimistic test estimates. | Selection uses only threshold-free metrics (step 1), so threshold choice cannot steer it. A split-half validation (select on one half, threshold on the other) is the stricter option. | Main: Essential. Split-half: Optional |
| 8.4 Training subsample changes the class prior (Section 1.5). | Default 0.5 thresholds become wrong. | Validation keeps the natural distribution, so tuned thresholds correct the shift. Always report metrics at both 0.5 and tuned thresholds, and threshold-free metrics as primary. | Essential |
| 8.5 Calibration and thresholding were conflated in v1. | ECE is about probabilities; thresholds are about decisions. | Separate them (steps 2 and 3); report ECE before and after temperature scaling. | Essential |

## 9. Explanation evaluation: plausibility vs faithfulness

Plausibility asks whether an explanation matches what humans marked; faithfulness asks whether the model actually depends on the marked words. They can disagree, and rationale supervision directly optimises plausibility, so **a plausibility gain alone is not evidence of a better explanation**. \[R\] Every explanation method is scored on both, against a random-explanation baseline and a human ceiling.

**Explanation methods compared** (all reduced to one score per word):

| Method | How | Cost note |
| --- | --- | --- |
| Intrinsic attention | eᵢ from the task's pooling weights (Section 3, Step 5) | Free: same forward pass |
| Integrated Gradients | Input-embedding attributions, pad-token baseline, 50 steps; report the completeness gap as a convergence check | About 50 forward/backward passes per post \[A: measure\] |
| LIME | Word-level, 1,000 samples, fixed seed, on a fixed sample of 500 posts per test set | About 1,000 forward passes per post \[A: measure\] |
| Random | Uniform random word scores, 5 draws | Reference only |

**Explained quantity.** HateXplain: toxicity score 1 − p(normal). Explicit head: p(toxic). Implicit head: p(implicit). Fixed in advance so methods are compared on the same target.

**Plausibility** (HateXplain test, toxic-majority posts with a rationale): token AUPRC against the majority rationale (threshold-free, primary) and token-F1@k with k = the number of rationale words. Reported beside the human ceiling (Section 1.7).

**Faithfulness** (every test set, posts predicted positive): comprehensiveness and sufficiency at word fractions {1, 5, 10, 20, 50}% \[A: these bins follow ERASER as I recall them; confirm in the ERASER paper\], averaged over bins (AOPC). Each value is reported as its difference from the random baseline on the same posts.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 9.1 v1 treated attention weights as explanations by default. | The faithfulness of attention is disputed in the literature (Jain & Wallace 2019; Wiegreffe & Pinter 2019 — to cite; not re-opened in this review). | Never call attention "the explanation" without its faithfulness scores next to IG and LIME. | Essential |
| 9.2 Plausibility and faithfulness were mixed in one table. | Rationale supervision can raise plausibility while lowering faithfulness. | Two separate tables; the conclusion about H2 must hold on both or say which one it holds on. | Essential |
| 9.3 A single k = 20% for comprehensiveness. | Results can depend on k. | AOPC over five bins, plus the per-bin curve in the appendix. | Essential |
| 9.4 No random reference. | Raw comprehensiveness has no meaning without a baseline. | Random-explanation baseline on the same posts; report differences. | Essential |
| 9.5 Stability across training seeds. | An explanation that changes with the seed is hard to trust. | Spearman correlation of word scores for the same post across the 3 seeds, per method. LIME also across 3 LIME seeds. | Essential |
| 9.6 Sanity check that explanations depend on the model. | Some methods produce similar maps for random weights. | Model-randomisation test: re-initialise the top encoder layers and measure how much each method's word ranking changes. | Optional |
| 9.7 Explanation cost was not measured. | It is the engineering argument for intrinsic explanations. | Seconds per explanation for each method on the same GPU and on CPU (Section 12). | Essential |

## 10. Fairness: identity-related false positives

Rationale supervision could move identity bias in either direction, so the paper should test it as an open question. \[V\] HateXplain's authors report that models trained with human rationales reduced unintended bias towards target communities. \[A\] But human rationales on hateful posts may often include the identity word itself, which could teach the model to flag any mention of a group. Both are plausible; the experiment decides.

**Measurements (no extra training):**

1. **HateXplain test, Borkan bias metrics** — Subgroup AUC, BPSN AUC (non-toxic posts that mention the community vs toxic posts that do not; low = identity false positives) and BNSP AUC, per target community and as a generalised mean. \[V\] This is the protocol HateXplain itself uses, over its 10 target communities, so results are comparable to that paper.
2. **HateCheck** — accuracy per functional test; the headline fairness number is the false-positive rate on the non-hateful contrast tests (e.g. reclaimed slurs \[V\], counter-speech, neutral identity mentions), plus accuracy per target identity.
3. **Mechanism check** — share of HateXplain training rationale words that are identity terms (from the target-community word lists), to explain any effect found in 1–2.
4. **Civil Comments identities** (CC0 \[V\]) — the same Borkan metrics zero-shot on the identity-annotated test set, explicit head p(toxic).

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 10.1 v1 listed HateCheck only as an extra test set. | No fairness claim can be made or rejected. | Pre-register the two-sided question: does rationale supervision change BPSN AUC and HateCheck non-hateful FPR relative to the no-rationale and random-matched models? | Essential |
| 10.2 Threshold-dependent FPR alone is fragile. | It moves with the threshold. | Report BPSN/BNSP (threshold-free) as primary and FPR at the tuned threshold as secondary. | Essential |
| 10.3 Count discrepancy in HateCheck documentation: the dataset card says 19 hate tests + 11 non-hate tests, the paper says 18 + 11 = 29 \[V\]. | Wrong counts in the paper are an avoidable error. | Use the paper's 29 = 18 + 11 and read functionality names from the CSV, not from memory. | Essential (one check) |
| 10.4 Small subgroups give unstable AUCs. | Large variance can look like bias. | Report subgroup sizes; bootstrap CIs; skip subgroups with fewer than 50 posts in either class \[R threshold\]. | Essential |
| 10.5 Civil Comments is a different platform. | Adds a fairness test on real data at moderate cost (inference over a large file). | Run on a fixed random sample if time is short; state the sample size. | Optional |

## 11. Reproducibility requirements

Every number in the paper must trace to one run folder, one config, one code commit and one frozen split file. \[R\] These are acceptance criteria for the implementation, checked before any result is written up.

| Item | Requirement | Where it lives |
| --- | --- | --- |
| Random seeds | Main runs: 42, 123, 2024. The seed sets weight init, task sampling, data order and dropout. GPU kernels are not bit-exact, so the paper reports variance across seeds rather than claiming exact replay. | config.json per run |
| Split files | Frozen once by the data script; MD5 hash per file; dedup and leakage counts saved. Never regenerated after the pilot. | Private Kaggle dataset, versioned |
| Configuration | One base config frozen after the pilot (date recorded); each experiment is a small override of it. | configs/ + config.json per run |
| Code version | Git commit hash written into every run folder; the code is attached to Kaggle as a versioned dataset or cloned at a fixed commit. | git\_commit.txt per run |
| Software | pip freeze, Kaggle image version, Python, PyTorch, Transformers, CUDA versions, GPU name. | env.json per run |
| Checkpoints | fp16 weights for all three seeds of RAMT, the no-rationale model and the random-matched control (needed for IG, LIME, deletion tests, explanation stability and the demo). Others keep predictions only. | Private Kaggle dataset |
| Logs | Per-epoch training log, full stdout log, timing file, per-example probabilities for validation and test of every head and every test set. | run folder |
| Results | Tables and figures generated only by the analysis script from run folders; a manifest lists each table, the runs it used and their hashes. | results/ + manifest.json |
| Pre-registration | Hypotheses H1–H3, primary metrics and the comparison list are fixed in this document before the main runs. | This document (approved version) |
| Release | Anonymised code link for review (EAAI is double-anonymised \[V\]); data released as post IDs + scripts where licences are unclear (IHC); checkpoints after acceptance. | Paper's data-availability statement |

All items are **Essential**.

## 12. Kaggle feasibility: memory and runtime

I cannot give honest runtime numbers without running on Kaggle, so v1's "20–30 minutes per run" is withdrawn. \[R\] A one-hour benchmark notebook runs first; its measurements, put into the budget formula below, decide how much of the optional list survives.

**What is verified about the platform**

- Your account shows 30 GPU hours available this week (your screenshot).
- GPU option: two T4 cards with 16 GB each; one secondary source reports the P100 option was retired in September 2026 \[V, secondary\].
- 12-hour limit per GPU session; 20 GB of saved output in the working directory \[V\].
- The weekly quota floats and is shown in account settings, not documented as a fixed number \[V\].
- **Not verified:** whether a T4×2 session uses quota at the same rate as one GPU. \[A\] Check the quota display before and after a 30-minute T4×2 session.

**Memory (arithmetic estimate, not a measurement).** Full fine-tuning with AdamW in mixed precision keeps about 16 bytes per parameter for weights, gradients and optimizer states: about 1.8 GB for HateBERT (110M), 2.0 GB for RoBERTa-base (about 125M \[A\]) and 2.9 GB for DeBERTa-v3-base (184M). Activations at batch 32 come on top and must be measured; I expect all three to fit in 16 GB at batch 32, but that is an assumption until the benchmark runs.

**Benchmark notebook B0 (about 1 GPU hour, before any experiment):** for each encoder in fp16 (and DeBERTa also in fp32 if fp16 fails):

1. 300 training steps at batch 32 on real training data with dynamic padding → steps per second and peak GPU memory.
2. Inference on 5,000 test posts → posts per second at batch 128, and median latency at batch 1.
3. Integrated Gradients on 50 posts and LIME on 10 posts → seconds per explanation.
4. The same inference measurement on CPU for the deployment table.

**Budget formula** (filled in from B0):

```latex
T_{\mathrm{run}} = \frac{E \cdot S}{r_{\mathrm{train}}} + \frac{N_{\mathrm{eval}}}{r_{\mathrm{eval}}}, \qquad T_{\mathrm{total}} = \frac{1}{g} \sum_{\mathrm{runs}} T_{\mathrm{run}} + T_{\mathrm{explain}}
```

E = epochs actually run, S = steps per epoch, N\_eval = all validation and test posts scored by all heads, r = measured rates, g = runs in parallel (2 only if the quota check above passes).

**Decision rule.** If T\_total for the Essential list exceeds 25 GPU hours (5 hours kept as margin), cut in this order: (1) DeBERTa, (2) the third seed of the pairwise transfer runs, (3) LIME (keep IG), (4) the third seed of the encoder comparison. Optional items run only if the Essential list fits.

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 12.1 Runtime was estimated, not measured. | A plan that overruns the quota leaves the paper half-finished. | Benchmark B0 + formula + cut order above. | Essential |
| 12.2 Evaluation cost was ignored. | Scoring every test set with every head after every run is a large share of the time. | Include N\_eval in the formula; score validation only during training, test once at the end. | Essential |
| 12.3 Output storage. | 20 GB fills quickly with checkpoints. | Save checkpoints only for the runs named in Section 11 (M1–M3); everything else keeps predictions and logs. | Essential |
| 12.4 CPU work charged to GPU sessions. | Wastes quota. | Data prep, TF-IDF baselines, domain-shift analysis, statistics, tables and figures all run in CPU notebooks. | Essential |
| 12.5 The 3-day deadline. | Unknown until B0 runs. | Re-plan the calendar from the B0 numbers; if the Essential list does not fit, say so before starting rather than cutting controls silently. | Essential |

## 13. Other rejection risks at EAAI

The largest remaining risk is novelty: rationale-supervised attention for hate speech already exists, and a 2025 preprint does almost exactly that. The contribution must be framed as multi-task transfer plus the controls, not as "aligning attention with rationales".

| Issue | Why it matters | Solution and exact change | Priority |
| --- | --- | --- | --- |
| 13.1 Close prior work. \[V\] BERT-HateXplain supervises the last-layer \[CLS\] attention with rationales; MRP (COLING 2022) uses masked rationale prediction as an intermediate task; [SRA (arXiv 2025)](https://arxiv.org/html/2511.07065v1) aligns model attention with human rationales on HateXplain. | A reviewer will ask what is new. | Position the contribution as (a) rationale supervision inside a multi-task explicit + implicit model, (b) a test of transfer to heads without rationales, (c) controls that isolate the human signal. Include a **single-task HateXplain + rationale** baseline (the prior-work setting inside our code). Cite all three. | Essential |
| 13.2 Published numbers on the same split are ignored. | HateXplain's official split allows direct comparison at zero cost. | Add a table of published HateXplain test results (from the original papers, cited) beside ours, marking any protocol differences. | Essential |
| 13.3 No large-language-model baseline in a 2026 submission. | Reviewers increasingly ask for one. | Zero-shot instruction-tuned open model on fixed 1,000-post samples of each test set, prompt released. Feasibility depends on B0 and model licence \[A\]. | Optional (recommended) |
| 13.4 EAAI desk-reject rules. \[V\] The abstract must separate the AI contribution from the engineering application; undefined acronyms in the title and abstract are forbidden; single-column format; double-anonymised review. | Desk rejection before review. | Checklist at the writing stage; the engineering application is an automated moderation pipeline with calibrated per-label decisions and explanation cost measured in milliseconds. | Essential |
| 13.5 Multiple comparisons. | Many pairwise tests inflate false positives. | Fix the family of planned comparisons now (Section 14); Holm correction within it; effect sizes with CIs. Everything else is labelled exploratory. | Essential |
| 13.6 Overclaiming in the title ("domain-adaptive", "explainable"). | Claims beyond the evidence are an easy reject. | Title states only what is tested, e.g. "Rationale-supervised multi-task learning for explicit and implicit toxicity detection: transfer, faithfulness and fairness". Final wording after results. | Essential |
| 13.7 Results from the earlier manuscript. | Those numbers came from a pipeline with leakage. | Reuse none of them; the new paper reports only runs from this methodology. | Essential |
| 13.8 Ethics and safety. | Hate-speech papers are expected to address harm. | Content warning; slurs masked in figures and examples; no release of raw text where licences are unclear; one paragraph on dual use. | Essential |
| 13.9 Annotator disagreement treated as noise. | Recent work studies disagreeing rationales directly \[V: arXiv 2605.31563\]. | Cite it; report the human ceiling (1.7) and the aggregation comparison (5.3). | Essential (citation + reporting) |

## 14. Consolidated methodology (for approval)

**39 Essential training runs in 8 groups**, a 10-run validation-only pilot and a 1-hour benchmark. Every group answers a stated hypothesis. Optional items run only if the measured budget allows.

**Hypotheses (pre-registered)**

- **H1 Multi-task benefit:** training explicit, implicit and HateXplain tasks jointly improves each task's primary metric over its single-task model, with no task significantly worse.
- **H2 Human-rationale benefit:** rationale supervision improves HateXplain classification or explanation quality over both the no-rationale model and the random-matched control.
- **H3 Transfer:** rationale supervision improves explanations of the explicit and implicit heads (cross-head plausibility, faithfulness), although those heads see no rationales.
- **F Fairness (two-sided):** rationale supervision changes identity-related false positives (BPSN AUC, HateCheck non-hateful FPR).
- **E Engineering:** intrinsic explanations cost far less per post than IG or LIME at comparable faithfulness.

**Model (simplified from v1)**

- One pretrained encoder (chosen on validation from HateBERT, RoBERTa-base, DeBERTa-v3-base), all layers fine-tuned.
- One additive attention pool per task over content tokens only (Section 3, Step 3).
- One linear head per task: 6 sigmoid outputs (explicit), 1 sigmoid (implicit), 3-way softmax (HateXplain).
- Removed from v1: BiLSTM, shared MLP projection, 2-layer heads, frozen layers.

**Training (Section 2 and 3):** one task per batch, square-root task sampling, λₜ = 1, rationale loss on the HateXplain pool with weight μ, AdamW, warmup 6%, weight decay 0.01, batch 32, at most 3 epochs, early stopping on the mean of per-task validation scores, mixed precision, word-level truncation W.

**Workflow with gates**

1. **S0 Data (CPU).** Build frozen splits with dedup and leakage report; domain-shift analysis; TF-IDF + logistic-regression and majority baselines. *Gate G1:* all split counts and the IHC stage-1 file confirmed.
2. **S1 Benchmark B0 (GPU, about 1 h).** *Gate G2:* budget formula filled in; cut list applied if needed.
3. **S2 Pilot (validation only, seed 42, test disabled).** 3 encoders × lr {2e-5, 3e-5} at μ = 0 (6 runs); μ ∈ {0.25, 0.5, 1.0} on the chosen encoder (3); majority aggregation at the chosen μ (1). *Gate G3:* base config frozen and dated.
4. **S3 Main runs (3 seeds, test enabled once).** Groups M1–M8 below. *Gate G4:* the first finished run is checked for NaNs, sane validation curves and complete output files before the rest are launched.
5. **S4 Post-hoc evaluation (GPU, short).** IG, LIME, deletion tests on M1–M3 checkpoints.
6. **S5 Analysis (CPU).** Every table, figure and statistic generated from run folders.

**Essential training runs**

| ID | Configuration | Answers | Runs |
| --- | --- | --- | --- |
| M1 | **RAMT:** task pools, soft human rationales, chosen μ | H2, H3, F, E | 3 |
| M2 | Multi-task, task pools, μ = 0 | H1, H2, H3 | 3 |
| M3 | Random-matched rationales, chosen μ | H2, H3 | 3 |
| M4 | Shared pool + human rationales | H3 | 3 |
| M5 | Single-task explicit / implicit / HateXplain, μ = 0 | H1 | 9 |
| M6 | Single-task HateXplain + human rationales (prior-work setting) | H2, novelty | 3 |
| M7 | Pairs at μ = 0: explicit+implicit, explicit+HateXplain, implicit+HateXplain | H1 transfer matrix | 9 |
| M8 | M2 configuration on the two other encoders | Encoder comparison | 6 |
|  | **Total** |  | **39** |

Checkpoints are kept for all seeds of M1–M3 (needed for stability of IG explanations); every other run keeps predictions and logs only.

**Optional, in priority order:** shared pool without rationales (3); RAMT on the other two encoders (6); LLM zero-shot baseline; Civil Comments fairness; lexicon control (3); union and intersection aggregation (2, pilot); no-subsampling ablation (3); linear probe; model-randomisation test; split-half validation; mask-replace deletion.

**Primary metrics:** explicit macro-AUPRC; implicit AUPRC; HateXplain macro-F1; plausibility token-AUPRC; faithfulness AOPC comprehensiveness and sufficiency minus random; fairness generalised-mean BPSN AUC and HateCheck non-hateful FPR. Secondary: F1 at tuned and 0.5 thresholds, ROC-AUC, ECE.

**Planned comparisons (Holm-corrected as one family; everything else exploratory)**

1. M1 vs M2 and M1 vs M3 on every primary metric (H2).
2. M2 vs M5 per task, and the M7 transfer matrix (H1).
3. M1 vs M6 on HateXplain metrics and plausibility (novelty over single-task rationale supervision).
4. Cross-head plausibility and faithfulness: M1 vs M2 vs M3; M1 vs M4 (H3).
5. Fairness: M1 vs M2 vs M3 (F).

**Statistics:** paired bootstrap over test instances (2,000 resamples), seed-matched differences averaged over the 3 seeds, 95% CIs, two-sided p-values with Holm correction across the family above; mean ± SD over seeds in every table.

## 15. Decisions needing your approval

Approve or change each line; implementation starts only after all seven are settled.

| # | Decision | My recommendation | Alternative |
| --- | --- | --- | --- |
| D1 | Implicit dataset | IHC stage-1, obtained through the authors' survey; implicit vs not-hate | If stage-1 cannot be obtained in time, the implicit task must be replaced by another dataset with both classes from one source; the methodology needs re-review then |
| D2 | Scope | The 39 Essential runs + pilot + benchmark; optional items only if B0 shows budget | Fewer seeds (not recommended: 3 is the minimum for seed variance) |
| D3 | Task-pool design with the shared pool as a comparison (M4) | Keep both | Drop M4 and weaken H3 to "tested for task pools only" |
| D4 | Rationale aggregation default | Soft mean, majority compared on validation | Majority as default |
| D5 | Fairness scope | HateXplain Borkan metrics + HateCheck (Essential); Civil Comments optional | Add Civil Comments to Essential |
| D6 | LLM baseline | Optional, run only if B0 leaves at least 3 GPU hours | Make it Essential and cut the M8 encoder comparison to 1 seed |
| D7 | Timeline | Re-plan after B0 with measured numbers; do not compress controls to meet the 3-day target | Keep 3 days and drop M4, M7 and M8 (weakens H1 and H3 claims) |

## 16. Round 2: responses and proposed amendments

Nothing in Sections 1–15 has been changed. Every amendment below is a **proposal**, tied to a decision in Section 17; it enters the methodology only when you approve that decision.

### R1. Task sampling and the definition of an epoch

The v2 schedule is internally inconsistent: S = Σⱼ⌈nⱼ/B⌉ steps is called an "epoch", but with square-root sampling no task completes exactly one pass in S steps. Jigsaw sees about 77% of its data per "epoch" and the two small tasks see about 130–140%.

**Sizes used** (recompute at gate G1): explicit n ≈ 44,603 = all toxic training rows + 30,000 clean \[A: assumes 16,225 toxic rows in Jigsaw train.csv\]; implicit n ≈ 14,274 = 70% of the 20,391 implicit + not-hate posts in the released file \[two secondary sources, see R6\]; HateXplain n = 15,383 \[V\]. Batch B = 32.

| Task | Batches per full pass | pₜ (T = 2) | Expected batches per v2 "epoch" (S = 2,322) | SD (binomial) | Examples per v2 "epoch" | Passes per v2 "epoch" | Passes after 3 v2 "epochs" |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Explicit | 1,394 | 0.4645 | 1,078.6 | 24.0 | 34,515 | 0.77 | 2.32 |
| Implicit | 447 | 0.2628 | 610.2 | 21.2 | 19,526 | 1.37 | 4.10 |
| HateXplain | 481 | 0.2727 | 633.2 | 21.5 | 20,262 | 1.32 | 3.95 |

**Proposed consistent schedule (decision D8): define training in steps and per-task quotas, not epochs.**

1. One **round** = 1,000 steps containing exactly round(1,000 · pₜ) batches of each task in shuffled order: 465 explicit, 263 implicit, 272 HateXplain. Deterministic quotas remove the binomial noise in the table.
2. A full run = 6 rounds = 6,000 steps. Exposure: explicit 2,790 batches = 89,280 examples = 2.00 passes; implicit 1,578 = 50,496 = 3.54 passes; HateXplain 1,632 = 52,224 = 3.39 passes.
3. Validation after every round; keep the best checkpoint; stop after 2 rounds without improvement. Linear warmup over the first 6% of the run's own steps, linear decay to zero at its last step.
4. **Exposure-matched subsets:** single-task and pairwise runs keep the same per-round quota for each task they contain and drop the others' batches. Single-task explicit = 2,790 steps, implicit = 1,578, HateXplain = 1,632; explicit+implicit = 4,368, explicit+HateXplain = 4,422, implicit+HateXplain = 3,210. Every task therefore sees the same examples-per-task in every configuration, so a multi-task gain cannot come from extra updates on that task.
5. Inside a task, examples are drawn without replacement and reshuffled at the end of each pass.

**Trade-off.** The alternative, step-matched subsets (every configuration runs 6,000 steps), gives single-task implicit and HateXplain models about 13 passes and inflates their overfitting; exposure matching is the cleaner control for H1.

### R2. A HateXplain batch with no eligible rationale posts

As written in Section 3, Step 4 divides by |ℬ\_r|, which is 0 in such a batch, so the loss is 0/0 = NaN and one NaN step corrupts the whole run. Two definitions avoid this; they differ in how much each eligible post counts.

| Option | Definition | Empty batch | Per-post weight |
| --- | --- | --- | --- |
| (a) Per-eligible mean, guarded | (1/\|ℬ\_r\|) Σ over eligible posts; term skipped when \|ℬ\_r\| = 0 | Rationale term absent; classification loss still applies | Varies: a post in a batch with few eligible posts counts more |
| **(b) Per-batch mean (proposed, D9)** | (1/B) Σ over eligible posts of the per-post cross-entropy | Sum is empty, term = 0 exactly, no gradient | Constant: every eligible post counts 1/B |

Under (b), the expected strength of the rationale signal equals μ times the eligible share q, so μ is chosen on validation with this definition and q is reported. \[A\] q is not yet known; if at least 40% of HateXplain training posts are eligible, the chance that a 32-post batch has none is at most 0.6³² ≈ 8 × 10⁻⁸, about 1 batch in 12 million. The guard is still required, and the code logs the eligible count per batch so the realised q is reported.

The random-matched control uses exactly the same eligible posts and the same definition.

### R3. Does the plan fit the GPU budget?

**Probably not as Essential-in-full on one GPU, unless the benchmark shows high throughput.** I still cannot state hours without the B0 measurement, but the workload can now be counted exactly in training steps, which turns the question into one measured number: the step rate a T4 sustains.

**Training steps (schedule from R1, 3 seeds):**

| Block | Steps |
| --- | --- |
| M1–M4 (4 configurations × 6,000) | 72,000 |
| M5 single-task (2,790 + 1,578 + 1,632 = 6,000 per seed) | 18,000 |
| M6 single-task HateXplain + rationale (1,632 per seed) | 4,896 |
| M7 pairs (4,368 + 4,422 + 3,210 = 12,000 per seed) | 36,000 |
| M8 two other encoders (6,000 each) | 36,000 |
| Pilot (10 runs × 6,000) | 60,000 |
| Reuse of 4 pilot runs as seed 42 of M1, M2 and M8 (D10) | −24,000 |
| **Total training** | **202,896** |

**Everything else, as training-step equivalents** \[A: one training step on 32 posts ≈ 96 single-post forward passes, i.e. backward ≈ 2× forward; B0 measures the true ratio\]:

- Validation: 6 rounds × about 9,000 posts (a fixed 5,000-post Jigsaw subset + all IHC and HateXplain validation) + one full-validation pass for thresholds ≈ 74,000 forward passes per run.
- Test: about 74,600 posts per run (Jigsaw test \[A: about 64,000 labelled rows\] + IHC 4,078 + HateXplain 1,924 + OLID 860 + HateCheck 3,728); none for pilot runs.
- Across 45 runs (39 with test evaluation) this is about 6.2 million forward passes, ≈ 65,000 step equivalents.
- Post-hoc explanations and deletion tests (IG on about 4,400 posts × 9 checkpoints, LIME on 500 HateXplain posts × 2 models, faithfulness and random baselines) ≈ 30,000–40,000 step equivalents \[A\].

**Total ≈ 303,000 step equivalents.** With 24 usable GPU hours (86,400 s), the plan fits only if B0 measures:

| Scope (D2) | Step equivalents | Needed rate, 1 GPU | Needed rate per GPU, 2 in parallel |
| --- | --- | --- | --- |
| A: Essential as written | ≈ 303,000 | ≥ 3.5 steps/s | ≥ 1.8 steps/s |
| B: A without the learning-rate grid (all encoders at 2e-5; pilot 7 runs) | ≈ 283,000 | ≥ 3.3 steps/s | ≥ 1.6 steps/s |
| C: B with M7 and M8 at 2 seeds | ≈ 251,000 | ≥ 2.9 steps/s | ≥ 1.5 steps/s |

The "2 in parallel" column holds only if the quota check in Section 12 shows a T4×2 session is charged like one GPU. I do not know a verified T4 step rate for this setup, so I make no claim about which row will fit; B0 decides, by the rule in D2.

### R4. Seed uncertainty vs test-sampling uncertainty

The v2 statistic (paired bootstrap over test instances, seed-averaged) measures only test-sampling uncertainty, *conditional on the three seeds*; it says nothing about whether a fourth seed would agree. The proposal (D11) estimates the two components separately and combines them.

For a comparison of systems A and B with seeds k = 1 … K (K = 3), using the **same seed for both systems** so data order and initialisation are shared:

```latex
D_k = M(A_k) - M(B_k), \qquad \bar D = \frac{1}{K} \sum_{k=1}^{K} D_k
```

```latex
V_{\mathrm{seed}} = \frac{s_D^2}{K}, \qquad V_{\mathrm{inst}} = \mathrm{Var}_b\big(\bar D^{(b)}\big), \qquad \mathrm{SE} = \sqrt{V_{\mathrm{seed}} + V_{\mathrm{inst}}}
```

- s\_D² is the sample variance of D\_k across seeds (K − 1 = 2 degrees of freedom). Because every D\_k uses the same full test set, its spread reflects training randomness, not test sampling.
- D̄⁽ᵇ⁾ is the seed-averaged difference on bootstrap resample b of the test instances (2,000 resamples), with the **same resampled indices for every seed and both systems**, so the test-sampling error shared by all seeds is counted once and not divided by K.
- Combined interval D̄ ± t\_ν · SE, with Welch–Satterthwaite degrees of freedom ν = (V\_seed + V\_inst)² · (K − 1) / V\_seed². When seed variance dominates, ν ≈ 2 and t ≈ 4.30, which honestly widens the interval.

**Reported for every planned comparison:** D̄; seed SD; instance-only 95% CI; combined 95% CI; and sign consistency (how many of the 3 seeds favour A).

**Claim rule.** A confirmatory effect needs a Holm-adjusted combined p < 0.05 **and** the same sign in 3 of 3 seeds. An effect significant on test instances but not across seeds is reported as "not robust to training randomness".

**Consequence.** With only 3 seeds, small effects will often fail the combined test. Running 5 seeds for M1–M3 would raise power for H2, at +36,000 training steps (2 extra seeds × 3 configurations × 6,000) (optional, decided in D11).

### R5. How the cross-head tests can support H3 despite domain and construct gaps

The tests support H3 only as a **controlled contrast**, never as an absolute score. The explicit head trained on Wikipedia comments and asked about Twitter/Gab posts will align poorly with HateXplain rationales in every condition; what H3 asks is whether that alignment moves *towards* the human rationales when, and only when, the HateXplain head was trained on real human rationales.

**Why the confounds cancel.** M1 (human rationales), M2 (none) and M3 (random-matched) share the same encoder, data, quotas, seed and posts. The training domain, the explicit label construct and the platform gap are identical across the three, so they shift all three scores together. Only the rationale target differs, so a difference M1 − M2 that is also present in M1 − M3 is attributable to *where* humans pointed.

**What the contrast cannot show, and how the protocol handles it:**

1. **Floor effects.** Explicit-head attention on HateXplain may sit near random for all conditions, leaving no room to detect change. Plausibility is therefore reported normalised: (model − random) / (human ceiling − random), and the floor is stated.
2. **Construct difference.** The explicit head looks for "toxic" cues, while HateXplain rationales justify "hateful/offensive". H3 is stated as *movement towards* human rationales, not agreement with them.
3. **Domain gap.** The implicit head (trained on IHC tweets) is tested on Twitter/Gab posts with a smaller platform gap; agreement between the two heads strengthens the conclusion.
4. **Plausibility without dependence.** Movement in attention alone is not enough: the second leg is faithfulness of the explicit head on Jigsaw test and of the implicit head on IHC test, where there is **no** domain or construct shift at all. Rationale supervision must not make those explanations less faithful.
5. **Shared pool (M4).** In M4 the explicit head reads the supervised attention directly, so its cross-head plausibility is high by construction; M4 is a design comparison, not evidence for H3.

**Pre-specified decision rule for H3 (task pools):**

- **Supported** if, for at least one of the explicit or implicit heads, normalised cross-head plausibility satisfies M1 > M2 and M1 > M3 under the R4 claim rule, **and** that head's faithfulness on its own test set is non-inferior to M2 (lower bound of the combined CI for AOPC comprehensiveness above −δ, δ from D12).
- **Partially supported** if plausibility moves but faithfulness is inferior, or if only M1 > M2 holds (rationale-loss effect not specific to human positions).
- **Not supported** otherwise; reported as a negative result.

### R6. IHC stage-1: availability and permitted use

**Available, but with no stated licence, and the released counts differ from the paper's.** It can stay Essential if the survey is completed on day 1 and the file is checked at gate G1.

| Question | Finding | Status |
| --- | --- | --- |
| Official access route | Complete a short Google Forms survey, then download a 2 MB zip (6 MB unpacked) from Dropbox. The README links two different Dropbox URLs. | \[V\] [dataset card](https://huggingface.co/datasets/SALT-NLP/ImplicitHate/blob/main/README.md) |
| Licence / terms | The dataset card states no licence or terms of use. The GitHub repository is MIT-licensed, but it does not say whether that covers the data. | \[V\] |
| Hugging Face copy by the authors' group | The SALT-NLP repo holds one data file, implicit\_hate.csv (708 kB); its contents and whether it is gated are not shown on the page. | \[V\] [file list](https://huggingface.co/datasets/SALT-NLP/ImplicitHate/tree/main); contents \[A\] |
| Third-party mirror | metaeval/implicit-hate-stg1 shows stage-1 posts with a 3-value class column. It is not the authors' access route. | \[V\] [mirror](https://huggingface.co/datasets/metaeval/implicit-hate-stg1) |
| Counts in the paper | Majority-vote consensus: 19,112 tweets = 933 explicit, 4,909 implicit, 13,291 not hate. | \[V\] [paper](https://arxiv.org/pdf/2109.05322) |
| Counts in the released file (as used by others) | 13,291 not hate, 7,100 implicit, 1,089 explicit; another paper uses 20,391 posts with 7,100 hateful; a third uses 18,666 after its own filtering. | \[V, secondary\] [thesis](https://home.cltl.labs.vu.nl/static/data/theses/thesis_Victoria_Im_2025.pdf), [SHIELD](https://arxiv.org/pdf/2403.12403), [2026 paper](https://arxiv.org/pdf/2606.18852) |
| Text artefacts | Sample rows in both classes contain retweet markers and stray quote characters. | \[V\] mirror preview |

**Proposed handling (D1):**

1. Use the authors' route only: one team member completes the survey today and downloads the zip. The mirror is not used, because the survey is the authors' condition of access.
2. Treat the data as **research use without redistribution**: upload it only to a *private* Kaggle dataset; release post IDs and split scripts, never text. \[A\] If the file has no ID column, release row indices of the official file instead, and email the authors before publication to confirm this is acceptable.
3. At G1, record which version the zip contains (counts per class) and cite it exactly; explain any difference from the paper's 19,112.
4. Apply the same normalisation to retweet markers and quotes in both classes, and run the shortcut check in Section 6 (issue 6.5).

### R7. Acceptance criteria for the engineering hypothesis E

E is accepted only if criteria E1, E2 and E4 all hold; E3 and E5 are reported either way. The thresholds are my recommendations (D12), fixed before any measurement; they are design targets, not predictions.

**Measurement protocol.** 500 fixed HateXplain test posts; M1 checkpoints of all 3 seeds; batch size 1; 20 warm-up posts discarded; median and 95th-percentile wall-clock time per post, with CUDA synchronisation on GPU. Hardware: one Kaggle T4 and the Kaggle CPU session; the same posts, code and precision (fp16 on GPU, fp32 on CPU) for every method. Explanation time includes everything needed to return word scores (for IG: all 50 steps; for LIME: all 1,000 samples).

| ID | Criterion | Threshold | Test |
| --- | --- | --- | --- |
| E1 | Overhead of intrinsic explanation | Median latency of prediction + attention word scores ≤ 1.05 × median latency of prediction alone, on both GPU and CPU | Direct ratio of medians |
| E2 | Speed vs post-hoc methods | Median intrinsic-explanation latency ≤ 1/10 of IG and ≤ 1/100 of LIME, same hardware | Direct ratio of medians |
| E3 | Deployment target (reported, not a pass/fail) | 95th percentile ≤ 100 ms on T4 and ≤ 1 s on CPU for prediction + explanation | Reported |
| E4 | Comparable faithfulness (non-inferiority) | Attention vs IG on M1: lower bound of the combined 95% CI of \[AOPC-comprehensiveness − random\] difference ≥ −δ, and upper bound of the \[AOPC-sufficiency − random\] difference ≤ +δ, with δ = 0.02 (probability units) | R4 combined interval |
| E5 | Supervision does not reduce attention faithfulness | Same non-inferiority test for attention of M1 vs attention of M2 | R4 combined interval |

**Why these numbers.** E1 checks that returning the attention costs essentially nothing. E2's ratios sit well below the theoretical pass counts (IG uses 50 forward-backward passes, LIME 1,000 forward passes), so failing them would reveal an implementation problem rather than a close call. δ = 0.02 means accepting at most a 2-percentage-point loss in the probability drop caused by deleting the top words; it should be confirmed before any results exist.

**If E4 fails,** the paper reports a speed–faithfulness trade-off curve instead of claiming "comparable faithfulness".

### R8. Remaining inconsistencies, missing controls and reviewer objections

Fourteen further items, none of which changes the research questions; most are one-line protocol fixes.

| # | Problem | Proposed fix | Decision | Priority |
| --- | --- | --- | --- | --- |
| 8.1 | Sections 2 and 14 still define training in epochs and "at most 3 epochs", which contradicts R1. | Replace with the round-based schedule if D8 is approved. | D8 | Essential |
| 8.2 | The periodic validation set was never defined; R3 assumed a 5,000-post Jigsaw subset. | Fixed stratified 5,000-post Jigsaw validation subset for early stopping; full validation sets for thresholds and temperatures. | D8 | Essential |
| 8.3 | The rule for choosing μ is unstated. | Highest validation composite classification score; ties within 0.005 broken by higher validation plausibility on HateXplain validation rationales. | D13 | Essential |
| 8.4 | Pilot runs reused as seed 42 carry selection bias (the winner on validation). | Disclose it, or run seed 42 again (+24,000 steps). | D10 | Essential |
| 8.5 | Random-matched control is identical to the human target on posts where every word is a rationale or the post has one word. | Count these posts and report the share; leave them in so both conditions see the same posts. | D13 | Essential |
| 8.6 | Our plausibility metrics (token-F1@k, token AUPRC) differ from those HateXplain reports, so the published-results comparison (13.2) would not be like-for-like. | Also compute HateXplain's own metrics with the evaluation code in their repository (it ships an ERASER-based folder \[V\]) for the comparison table only. | D13 | Essential |
| 8.7 | No standard \[CLS\] fine-tuning baseline. Attention pooling might itself change accuracy, and \[CLS\] fine-tuning is the community default. | Single-task \[CLS\] baseline per task (1,632–2,790 steps each); 3 seeds if B0 allows, else 1 seed, labelled. | D13 | Essential (at least 1 seed) |
| 8.8 | The planned-comparison family is large (5 comparison groups × several metrics), so Holm correction will cost power. | Confirmatory family of 12 tests: H1 Δₜ for 3 tasks; H2 M1 vs M2 and M1 vs M3 on HateXplain macro-F1, plausibility and faithfulness (6); H3 rule for 2 heads; F M1 vs M2 on generalised-mean BPSN AUC. Everything else exploratory. | D11 | Essential |
| 8.9 | Explicit-head explanations target only p(toxic); the other 5 labels are not explained. | State as a scope limit. | — | Essential (reporting) |
| 8.10 | Subgroup membership for HateXplain bias metrics is undefined. \[A\] The original paper's exact rule was not verified in this review. | Use HateXplain's own bias code if available; otherwise a community counts as targeted when at least 2 annotators name it. | D13 | Essential |
| 8.11 | Exposure-matched subsets run fewer steps, so a single-task run and the full model share a seed but not the data order. | Acceptable; pairing still fixes initialisation. Note it in the statistics section. | — | Essential (reporting) |
| 8.12 | Rationales produced by large language models for Jigsaw/IHC are an obvious reviewer suggestion; prior work extracts rationales with LLMs \[V: SHIELD, arXiv 2403.12403\]. | Cite as related work; position as future work. | — | Essential (citation) |
| 8.13 | Attention over content tokens differs from BERT-HateXplain's \[CLS\]-attention supervision. | Explain the choice (no attention sink, tokenizer-invariant target) in the method section; the M6 baseline uses our formulation, so state that it is "prior-work setting", not a reimplementation. | — | Essential (reporting) |
| 8.14 | Small IHC test set (about 4,000 posts) gives wide intervals for the implicit task. | Report CI widths; do not over-interpret implicit-task differences below them. | — | Essential (reporting) |

## 17. Decision walkthrough

Each decision lists the alternatives, the trade-off, my recommendation and what it means for the research. D1–D7 revisit Section 15; D8–D13 are new from this round.

### D1. Implicit dataset

- **Alternatives:** (a) IHC stage-1 through the authors' survey; (b) the third-party Hugging Face mirror; (c) a different implicit-hate dataset; (d) drop the implicit task.
- **Trade-offs:** (a) is the standard benchmark, both classes come from one source, but it has no stated licence and released counts differ from the paper (R6). (b) is faster but bypasses the authors' access condition. (c) changes the construct and was not reviewed here. (d) removes the "explicit + implicit" framing that motivates the paper.
- **Recommendation:** (a), with the R6 handling (private upload, IDs only, version recorded at G1).
- **Consequences:** implicit results are comparable with IHC literature once the version is stated. If the download has not arrived by the end of day 1, work pauses for a re-review rather than silently switching datasets.

### D2. Scope

- **Alternatives:** scope A (Essential as written), B (no learning-rate grid, all encoders at 2e-5), C (B plus 2 seeds for M7 and M8) from R3.
- **Trade-offs:** A gives the fairest encoder comparison. B weakens that comparison (one learning rate may suit one encoder better), which becomes a stated limitation. C also lowers power for the task-transfer matrix (H1) and the encoder comparison; H2, the core claim, keeps 3 seeds in every scope.
- **Recommendation:** decide by rule after B0: take the largest scope whose required rate is at most 85% of the measured rate (a 15% margin for slow sessions). If even C does not fit, stop and re-plan; options then are fewer clean Jigsaw rows, dropping DeBERTa, or using next week's quota.
- **Consequences:** the scope, and therefore which claims are made, is fixed by a measurement and a rule written down in advance, not by how results look.

### D3. Shared attention pool (M4)

- **Alternatives:** keep M4; drop it; keep it and add its no-rationale counterpart (completes the 2 × 2).
- **Trade-offs:** M4 costs 18,000 training steps (about 9% of training). Without it, H3 covers task-specific pools only and a reviewer may ask why sharing was not tried. The counterpart adds another 18,000 steps.
- **Recommendation:** keep M4; counterpart optional.
- **Consequences:** H3 can say whether transfer depends on where supervision is applied.

### D4. Rationale aggregation

- **Alternatives:** soft mean, majority, union, intersection.
- **Trade-offs:** soft keeps graded disagreement; majority is conventional and matches the evaluation gold; union adds noisy words; intersection leaves many posts with few or no rationale words.
- **Recommendation:** soft for training, majority compared in one validation-only pilot run (higher validation composite wins; tie → soft); evaluation gold always majority (Section 9).
- **Consequences:** one extra pilot run; aggregation is fixed before any test result is seen.

### D5. Fairness scope

- **Alternatives:** HateXplain bias metrics + HateCheck (Essential) with Civil Comments optional; or Civil Comments also Essential.
- **Trade-offs:** Civil Comments adds a large real-world identity test, at the cost of inference over a big file (a fixed sample would be used).
- **Recommendation:** keep Civil Comments optional.
- **Consequences:** fairness claims are limited to HateXplain's 10 target communities and HateCheck's functional tests, and are worded that way.

### D6. Large-language-model baseline

- **Alternatives:** none; optional; Essential.
- **Trade-offs:** reviewers increasingly expect one, but it needs GPU time and a licence check for the model, and it is a zero-shot paradigm that does not test any hypothesis here.
- **Recommendation:** optional, run only if at least 3 GPU hours remain after the Essential list; otherwise one limitations paragraph.
- **Consequences:** a moderate reviewer risk if it is missing; no effect on H1–H3.

### D7. Timeline

- **Alternatives:** hold the 3-day deadline and cut scope to fit; or re-plan after B0.
- **Trade-offs:** the work has sequential gates (data → benchmark → encoder pilot → μ pilot → main runs → post-hoc → analysis), and each Kaggle session is capped at 12 hours, so even with enough quota the calendar is not compressible below roughly that chain. Writing is extra.
- **Recommendation:** re-plan the calendar from B0. If 3 days is a hard limit, take scope C and drop the optional list entirely, and accept that the encoder comparison and transfer matrix are weaker.
- **Consequences:** the 3-day target covers experiments at best; the manuscript itself needs additional days.

### D8. Training schedule (new, from R1)

- **Alternatives:** v2 epochs with stochastic square-root sampling; round-based deterministic quotas with exposure-matched subsets; round-based quotas with step-matched subsets.
- **Trade-offs:** v2 is inconsistent (no task does one pass per "epoch"). Step-matched subsets give single-task implicit and HateXplain about 13 passes and confound H1 with over-training. Exposure matching keeps per-task examples identical across configurations.
- **Recommendation:** rounds of 1,000 steps (465 / 263 / 272 batches), 6 rounds, exposure-matched subsets, 5,000-post Jigsaw validation subset for early stopping (R8 8.1–8.2).
- **Consequences:** Sections 2 and 14 change from "epochs" to "rounds"; quotas are recomputed from the real sizes at G1 with the same formula.

### D9. Rationale-loss normalisation (new, from R2)

- **Alternatives:** guarded per-eligible mean; per-batch mean (divide by B).
- **Trade-offs:** the first makes a post's weight depend on how many eligible posts share its batch; the second gives every eligible post the same weight and is exactly zero on an empty batch.
- **Recommendation:** per-batch mean, with the eligible count logged.
- **Consequences:** μ's meaning includes the eligible share q, which is reported; μ is tuned with this definition.

### D10. Reusing pilot runs as seed 42 (new, from R3)

- **Alternatives:** reuse the 4 winning pilot runs as seed 42 of M1, M2 and M8; re-run them.
- **Trade-offs:** reuse saves 24,000 steps (about 11% of the training total before reuse) but those runs were chosen because they did well on validation, a mild optimistic bias in one of three seeds. Test data was never seen during selection.
- **Recommendation:** reuse, and disclose it. Re-run only if B0 shows ample budget.
- **Consequences:** one sentence in the paper; pilot checkpoints must be kept until their test evaluation runs.

### D11. Statistics (new, from R4 and R8 8.8)

- **Alternatives:** instance-only bootstrap (v2); combined seed + instance variance with a sign-consistency rule; 5 seeds for M1–M3 in addition.
- **Trade-offs:** instance-only intervals overstate certainty about training randomness. The combined rule is honest but conservative with 3 seeds. Five seeds cost +36,000 steps.
- **Recommendation:** combined variance (R4), 3-of-3 sign rule, confirmatory family of 12 tests; 5 seeds for M1–M3 only if B0 shows surplus budget.
- **Consequences:** fewer, more defensible confirmatory claims; small effects may be reported as "not robust to training randomness".

### D12. Engineering acceptance thresholds (new, from R7)

- **Alternatives:** absolute non-inferiority margin δ = 0.02; a relative margin (e.g. 10% of IG's AOPC); no formal criterion.
- **Trade-offs:** a relative margin scales with the metric but depends on IG's own value, which is unknown in advance; an absolute margin is simpler to pre-register. Without criteria, "comparable" is a reviewer target.
- **Recommendation:** E1 ≤ 5% overhead; E2 ≥ 10× vs IG and ≥ 100× vs LIME; E3 reported; E4–E5 non-inferiority with δ = 0.02; the same δ is used in the H3 faithfulness leg.
- **Consequences:** hypothesis E can fail cleanly; if E4 fails the paper reports a trade-off instead of a claim.

### D13. Protocol clarifications bundle (new, from R8)

- **Contents:** μ chosen by validation composite with plausibility as tie-break (8.3); degenerate random-control posts counted and kept (8.5); HateXplain's own metric code for the published-results table (8.6); single-task \[CLS\] baseline per task (8.7); subgroup membership rule for bias metrics (8.10).
- **Trade-offs:** all are small, except the \[CLS\] baseline: 6,000 steps per seed.
- **Recommendation:** approve all; \[CLS\] baseline at 3 seeds if B0 allows, otherwise 1 seed labelled as such.
- **Consequences:** closes the most predictable reviewer questions about comparability and the pooling choice.

## 18. Revised decision list for approval

Thirteen decisions; set each to Approved, Approved with changes or Rejected. Implementation starts only when all are settled, beginning with the data notebook and benchmark B0.

| # | Decision | Recommendation | Main consequence | Your decision |
| --- | --- | --- | --- | --- |
| D1 | Implicit dataset | IHC stage-1 via the authors' survey; private upload; IDs only; version recorded at G1 | Pause for re-review if not obtained on day 1 | Pending |
| D2 | Scope | Largest of scopes A / B / C whose required rate ≤ 85% of the B0-measured rate | Claims fixed by a pre-set rule | Pending |
| D3 | Shared pool M4 | Keep; no-rationale counterpart optional | H3 covers both pool designs | Pending |
| D4 | Aggregation | Soft for training, majority compared on validation; majority as evaluation gold | One extra pilot run | Pending |
| D5 | Fairness scope | HateXplain bias metrics + HateCheck; Civil Comments optional | Fairness claims limited to two settings | Pending |
| D6 | LLM baseline | Optional, only if ≥ 3 GPU hours remain | Moderate reviewer risk if absent | Pending |
| D7 | Timeline | Re-plan from B0; if 3 days is hard, scope C, no optional items | Writing needs extra days | Pending |
| D8 | Training schedule | 1,000-step rounds with fixed quotas, 6 rounds, exposure-matched subsets | Sections 2 and 14 change from epochs to rounds | Pending |
| D9 | Rationale-loss normalisation | Per-batch mean (divide by B), eligible count logged | μ includes eligible share | Pending |
| D10 | Pilot reuse | Reuse 4 pilot runs as seed 42, disclosed | Saves 24,000 steps | Pending |
| D11 | Statistics | Combined seed + instance variance, 3-of-3 sign rule, 12-test confirmatory family | Fewer, stronger claims | Pending |
| D12 | Engineering thresholds | E1 ≤ 5% overhead; E2 ≥ 10× / 100×; δ = 0.02 | E can fail cleanly | Pending |
| D13 | Protocol clarifications | Approve 8.3, 8.5, 8.6, 8.7, 8.10; \[CLS\] baseline 3 seeds if budget allows | Closes predictable reviewer questions | Pending |

## 19. Round 3: final concerns before approval

Sections 1–18 are unchanged. This round changes four recommendations: **D10** (use a dedicated pilot seed instead of reusing pilot runs), **D11** (add a pooled seed-variance floor), **D12** (replace the fixed δ = 0.02 with margins derived from IG's advantage over random) and **D13** (run the \[CLS\] baseline at 3 seeds or not at all). It also refines D1, D2 and D8. All text changes are listed as amendments A1–A20 in 19.8; all twenty were approved (Nov 2026) and govern wherever Sections 1–18 still carry the older wording; the implementation follows them (IMPLEMENTATION\_LOG §3).

Tags as before: \[V\] verified on an opened page, \[R\] my recommendation, \[A\] unresolved assumption.

### 19.1 IHC: what written permission to seek

**\[V\] Facts.** Access is gated by a survey and a Dropbox download; the dataset card states no licence or terms; the GitHub repository is MIT-licensed without saying whether that covers the data; the posts are tweets from accounts of US extremist groups (Sections 1 and R6). Several published studies use IHC \[V: R6 sources\], which shows common practice, **not** permission.

**\[R\] Request, by email from an institutional address, copied to your supervisor**, to the corresponding author listed in the EMNLP 2021 paper (the arXiv record shows Caleb Ziems as submitter \[V\]; \[A\] current addresses must be taken from the paper or the authors' current pages). Ask the authors to confirm, in writing:

1. **Version.** The file names, download date and MD5 checksum of the copy we hold are the current official release, and which label counts apply (the paper's 19,112 consensus tweets or the released file).
2. **Use.** Non-commercial academic research, including training and evaluating machine-learning models on the corpus, and publishing results in a peer-reviewed journal (Engineering Applications of Artificial Intelligence, Elsevier).
3. **Storage.** Keeping one copy in a private Kaggle dataset visible only to the named research team.
4. **Derived artefacts.** Releasing (a) split identifiers or row indices, (b) fine-tuned model weights, (c) aggregate statistics, and (d) a few short, slur-masked example quotes in the paper.
5. **Conditions.** The citation they require and any restrictions (for example no redistribution of tweet text, no attempt to identify users).
6. **Platform terms.** Whether they require us to observe any conditions arising from the original platform's terms for the tweet text. \[A\] I did not verify the platform's current terms; the authors' permission covers their annotations and compilation, not rights in third-party tweet content, so we never redistribute text in any case.

**Can IHC stay Essential?** \[R\] Yes, **if** a written reply grants at least items 2 and 3. That reply is a stronger basis than an unstated licence: it is archived with the project and cited in the data-availability statement as "used with the permission of the dataset authors". Items 4a–4d decide only what we release.

**Timing \[R\].** Gate the **submission**, not the experiments, on the reply: send the request on day 1, run experiments in parallel, and decide before submission. If the authors refuse, or have not replied by submission, the implicit-task results cannot be published as planned; the paper would need a re-review (a different implicit dataset or no implicit task). \[A\] Your institution may also require its own ethics or data-management approval for research on hateful content; check this with your supervisor.

### 19.2 D10: reusing pilot runs vs rerunning

**\[R\] Reuse is not defensible for the central comparisons; the recommendation changes to a dedicated pilot seed.**

**What reuse does statistically.** A pilot picks, among candidate configurations, the one whose seed-42 run scored best on validation. Part of that run's lead comes from the configuration, part from seed-specific luck. The luck also shows on the test set, because validation and test scores of the same run are positively correlated, so the selected run's test score tends to exceed that configuration's expected test score (a winner's-curse bias). The bias grows with the number of candidates and with seed noise relative to the true differences between candidates.

**Where the bias lands in this design:**

| Comparison | Seed-42 run of A | Seed-42 run of B | Direction of bias |
| --- | --- | --- | --- |
| H2: M1 vs M3 | selected among 3 μ values | fresh | **Favours M1 — inflates the core claim** |
| H2: M1 vs M2 | selected among μ values | selected among encoders × lr | Unknown sign |
| H1: M2 vs M5 | selected among 6 encoder × lr runs | fresh | **Favours M2 — inflates H1** |
| Encoders: M2 vs M8 | selected | selected | Roughly symmetric |

It also distorts D11: one selected seed is an atypical draw, which affects both the mean difference and the seed variance.

**Options:**

| Option | Extra cost vs reuse | Assessment |
| --- | --- | --- |
| (a) Reuse all 4 | 0 | Biases H1 and H2 upward; a careful reviewer can reject on this alone |
| (b) Rerun the 4 configurations with seed 42 | 4 runs ≈ 30,200 step-equivalents | Unbiased, but seed 42 then appears in both pilot and main runs, which needs explaining |
| **(c) Pilot on a dedicated seed (seed 1), never reported; all main runs fresh** | same as (b) | Unbiased and simplest to state |
| (d) As (c) with a shorter pilot (4 rounds instead of 6) | saves 20,000 training steps, offsetting most of the cost | Choices made at 4 rounds may not transfer to 6, especially μ; use only if the scope rule otherwise forces scope C |

**Recommendation: (c).** Cost about 30,200 step-equivalents (4 runs × 6,000 steps + their validation and test passes). Updated required rates (replacing the R3 table): scope A ≈ 333,000 step-equivalents → ≥ 3.9 steps/s on 1 GPU or ≥ 1.9 per GPU in parallel; B ≈ 313,000 → ≥ 3.6 / ≥ 1.8; C ≈ 281,000 → ≥ 3.3 / ≥ 1.6. Option (d) is the fallback only under scope C.

### 19.3 D11: is the combined-variance design valid with three seeds?

**\[R\] As written in R4 it is not adequate, because it fails worst exactly when its estimate is least reliable.** A pooled seed-variance floor fixes the main defect.

**Limitations of the R4 formula (standard statistics, my reasoning):**

1. **The seed variance is barely estimated.** With K = 3, s\_D² has 2 degrees of freedom. Its 95% interval for the true variance runs from 2/7.378 ≈ 0.27 to 2/0.0506 ≈ 39.5 times s\_D² (chi-square quantiles with 2 df), a factor of about 150.
2. **Near-zero estimates make the test anti-conservative.** If the three D\_k happen to be close, s\_D² ≈ 0, so V\_seed ≈ 0 and the Welch–Satterthwaite ν becomes very large; the interval collapses to the instance-only interval. With 3 draws this happens often by chance.
3. **When seed variance dominates, the interval is very wide** (t with 2 df = 4.30), so the design swings between too narrow and too wide depending on luck.
4. **The 3-of-3 sign rule is not a test.** If the true difference is zero and seed noise is symmetric, all three seeds favour A with probability 1/8 = 0.125. It is a consistency filter only.
5. **Satterthwaite treats the bootstrap variance as exact** (infinite df). With thousands of test posts this is reasonable.

**Proposed fix: a pooled seed-variance floor.**

- σ̂²\_pool = within-configuration variance of a metric across seeds, pooled over every main configuration evaluated on the same test set and metric. Each configuration adds K − 1 = 2 degrees of freedom; for the explicit metric, the configurations containing the explicit task (M1–M4, single-task explicit, two pairs, two M8 encoders) give about 9 × 2 = 18 df.
- For a paired difference, Var(D) = Var(A) + Var(B) − 2 Cov(A, B). Shared seeds make the covariance non-negative in expectation \[A: checked empirically by reporting the correlation of A\_k and B\_k\], so 2σ̂²\_pool is a conservative upper value.
- **Seed component:** V\_seed = max(s\_D², 2σ̂²\_pool) / K. The degrees of freedom are those of whichever term is used (2 for s\_D², about 18 for the pooled term).
- **Zero or near-zero s\_D²** is therefore never used alone: the pooled term sets the floor.
- **Homogeneity check:** report each configuration's seed SD; flag any configuration whose SD exceeds 3 × σ̂\_pool and report its comparisons with s\_D² alone (df = 2).
- The interval and Holm correction then follow R4; same-sign-in-all-seeds stays as a reported robustness condition, not a test.

**Do five seeds for M1–M3 materially help?**

|  | 3 seeds | 5 seeds |
| --- | --- | --- |
| df of s\_D² | 2 | 4 |
| 95% interval for true variance, as multiples of s\_D² | 0.27 – 39.5 | 0.36 – 8.3 |
| t quantile (0.975) when seed variance dominates | 4.30 | 2.78 |
| Seed component of the SE (same σ) | 1.00 | 0.77 |
| One-sided sign-test minimum p (all seeds favour A) | 0.125 | 0.031 |

&#91;R\] With the pooled floor, the gain from five seeds is moderate rather than decisive, but it makes the three H2 comparisons (M1 vs M2, M1 vs M3) much less dependent on the pooling assumption, and lets the sign test carry evidence on its own. Cost: 2 seeds × 3 configurations × 6,000 steps = 36,000 training steps plus about 9,300 step-equivalents of evaluation. Recommendation: five seeds for M1–M3 if the B0 rate supports scope A or B with that added; otherwise three seeds with the pooled floor, stated as a limitation.

### 19.4 D12: justifying the non-inferiority margin

**\[R\] δ = 0.02 had no basis and is withdrawn.** I found no established non-inferiority margin for explanation-faithfulness metrics. \[V\] A 2026 EMNLP study of 494 explainable-NLP papers found that only 25.30% backed their evaluation claims with statistical testing ([survey](https://aclanthology.org/2026.emnlp-main.504.pdf)), and the ERASER benchmark reports raw sufficiency and comprehensiveness without statistical testing ([ICE, Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1055.pdf)).

**A principled replacement, borrowed from clinical non-inferiority design.** \[V\] Under FDA guidance, the margin can be no larger than the whole effect of the active control over placebo (M1) and should represent the acceptable loss (M2); trials commonly require preserving 50% of the control's effect ([FDA guidance via a trial protocol](https://cdn.clinicaltrials.gov/large-docs/33/NCT04255433/SAP_001.pdf); [review of margin choice](https://pmc.ncbi.nlm.nih.gov/articles/PMC5510081)).

**Mapping to this study \[R\]:** active control = Integrated Gradients; placebo = random explanation; test method = intrinsic attention.

```latex
m_C = \mathrm{LB}_{95}\big[\mathrm{AOPC}_C(\mathrm{IG}) - \mathrm{AOPC}_C(\mathrm{rand})\big], \quad \delta_C = 0.5\, m_C
```

```latex
m_S = \mathrm{LB}_{95}\big[\mathrm{AOPC}_S(\mathrm{rand}) - \mathrm{AOPC}_S(\mathrm{IG})\big], \quad \delta_S = 0.5\, m_S
```

(C = comprehensiveness, higher is better; S = sufficiency, lower is better; LB₉₅ = lower bound of the 95% interval.)

- **Estimated on validation, fixed before test.** m\_C and m\_S are measured on validation posts (HateXplain validation, and samples of Jigsaw and IHC validation) with the M1 checkpoints, playing the role of the "historical trials". The test set never sets its own margin.
- **Separate margins for comprehensiveness and sufficiency.** They differ in direction, scale and typical distance from random, so one shared number would be too strict for one and too loose for the other.
- **Not assessable** when m ≤ 0, i.e. IG is not clearly better than random on that metric; non-inferiority to a control with no shown effect means nothing, so the metric is reported descriptively.
- **Sensitivity:** results are also reported at 25% and 75% preserved fractions.
- **Same construction for E5 and the H3 faithfulness leg**, with M2's attention as the active control.

&#91;A\] The 50% fraction is a judgement imported from another field; it is stated as such in the paper.

### 19.5 D8: consistency of the round schedule

**\[R\] Consistent in exposure, learning-rate shape and validation frequency, with two defects:** early termination breaks exposure matching and the budget, and gradient clipping is unstated in the methodology.

| Aspect | Full model | Pairwise | Single-task | Consistent? | Fix |
| --- | --- | --- | --- | --- | --- |
| Round length | 1,000 steps (465 / 263 / 272) | 728 / 737 / 535 steps | 465 / 263 / 272 steps | Yes, if a round is defined as one cycle of the included tasks' quotas, not as "1,000 steps" | A1 |
| Validation | After each of 6 rounds | After each of 6 rounds | After each of 6 rounds | Yes: same per-task exposure at each check | — |
| Early stopping | Stop after 2 rounds without improvement | Same | Same | **No:** configurations stop at different exposures, and R3's budget assumed full runs | A2: no early termination; keep the best of the 6 round checkpoints |
| Learning-rate schedule | Warmup 6% (360 steps), linear decay to 0 at step 6,000 | Same fractions of own length | Same fractions (e.g. 95 warmup steps for implicit) | Yes: each task's batches are spread evenly over rounds, so at the end of round r every run is at r/6 of its schedule | — |
| Encoder updates per unit of task exposure | All tasks update the encoder | Fewer | Fewest | Inherent to multi-task learning; this is part of what H1 measures | Note in paper |
| Composite validation score | Mean of 3 task metrics | Mean of 2 | 1 metric | Yes | — |
| Jigsaw validation subset | 5,000 posts | When explicit included | When explicit included | Yes | — |
| Gradient clipping | Not stated in Sections 1–18 | — | — | Gap | A6: clip at norm 1.0 everywhere |

### 19.6 D13: what the \[CLS\] baseline enables

**\[R\] It is an architectural and external-validity control, not a test of H1–H3, F or E.** Every central comparison is already made inside the attention-pooling family (M1–M7 all use attention pooling), so none of them changes if \[CLS\] is absent.

| Comparison enabled | Question answered | Needed for |
| --- | --- | --- |
| Single-task attention pooling (M5) vs single-task \[CLS\], per task | Does the pooling choice itself change classification? | Architecture justification |
| Single-task \[CLS\] vs published results on HateXplain's official split | Are our single-task baselines competitive with the standard fine-tuning setup used in prior work? | External validity of the 13.2 published-results table |
| Deployed RAMT vs standard \[CLS\] model | What accuracy, if any, does built-in explainability cost? | Supporting evidence for the engineering argument |

**Reviewer risk if absent.** "Your baselines use a non-standard head, so the multi-task gain might be an artefact." H1 is internally controlled (M2 and M5 share pooling), so the objection is answerable in text, but data is stronger.

**\[R\] Change from Round 2:** a 1-seed run cannot pass the D11 rules and invites the criticism it is meant to prevent. Run it at **3 seeds** (3 × 6,000 = 18,000 training steps plus about 13,900 step-equivalents of evaluation) if budget allows, **or omit it** and say so in the limitations with published \[CLS\] numbers for context. Classified as **Optional, high priority** (first in the optional list).

### 19.7 Remaining contradictions across Sections 1–18

Nineteen contradictions or gaps remain; each maps to an amendment in 19.8, except C19, now resolved: ERASER's own evaluation code uses the AOPC bins \[0.01, 0.05, 0.1, 0.2, 0.5\], which the implementation adopts.

| # | Where | Conflict | Amendment |
| --- | --- | --- | --- |
| C1 | Section 0 item 2 vs D13 | Says \[CLS\] pooling is cut; D13 adds a \[CLS\] baseline (different purpose) | A15 |
| C2 | Sections 2 and 14 vs D8 | "Epoch", "at most 3 epochs" vs rounds | A1, A6 |
| C3 | R1 step 3 / D8 vs R3 | Early termination vs a budget that assumes full runs | A2 |
| C4 | Section 3 Step 4 vs D9 | 1/\|ℬ\_r\| vs 1/B | A3 |
| C5 | Section 5 (aggregation) vs D4 | "Macro-F1 and plausibility" vs "composite score, tie → soft" | A4 |
| C6 | Section 7.2 vs D2 | Learning-rate grid Essential vs dropped in scopes B and C | A5 |
| C7 | Section 12 vs D2 | Different cut orders (DeBERTa first vs learning-rate grid first) | A10 |
| C8 | Section 12 vs R3 | 25 vs 24 usable GPU hours | A10 |
| C9 | Section 14 statistics vs R4 / D11 | Instance-only bootstrap vs combined variance with pooled floor | A8, A13 |
| C10 | Section 14 planned comparisons vs R8 8.8 | Five open-ended groups vs a 12-test confirmatory family | A9 |
| C11 | Sections 4.1 and 5.1 vs R4 / R5 | "CIs excluding zero" vs the R4 claim rule and R5 decision rule | A20 |
| C12 | R3 table vs revised D10 | Counts reuse of pilot runs | A12 |
| C13 | Sections 11, 14 vs revised D10 | Pilot checkpoints kept for reuse vs not needed | A11 |
| C14 | R5, R7 vs 19.4 | δ = 0.02 vs validation-derived margins | A14 |
| C15 | Section 15 vs Section 18 | Two decision tables with different recommendations | A16 |
| C16 | Section 1 (IHC) vs 19.1 | IHC Essential without a permission condition | A17 |
| C17 | R8 8.7 / D13 vs 19.6 | "1 seed, labelled" vs "3 seeds or omit" | A18 |
| C18 | Section 14 run list vs 19.3, 19.6 | Optional list lacks the \[CLS\] baseline and five seeds for M1–M3; pilot seed not stated | A19 |
| C19 | Section 9 | ERASER bin set {1, 5, 10, 20, 50}% still \[A\] | Verify in the ERASER paper before implementation (no text change until verified) |

### 19.8 Exact proposed amendments

Each amendment quotes the current text and gives the replacement. They are applied together, after your approval of the linked decision; nothing below is in effect yet.

| ID | Location | Current text | Proposed text | Decision |
| --- | --- | --- | --- | --- |
| A1 | Section 2, "Batch construction" | One "epoch" = S = Σ\_t ⌈n\_t / B⌉ steps, so every configuration sees the same number of updates per epoch. | Training is organised in rounds. One round is one cycle of fixed per-task quotas q\_t = round(1,000 · p\_t) batches (465 / 263 / 272 for the full model) in shuffled order. Every configuration runs 6 rounds; configurations with fewer tasks keep the same quotas for the tasks they contain (exposure matching). | D8 |
| A2 | Section 2, row 2.5; R1 step 3 | …and log every task's curve so a regression is visible. / Validation after every round; keep the best checkpoint; stop after 2 rounds without improvement. | Add to 2.5: Validation runs at the end of each of the 6 rounds; there is no early termination; the best round's checkpoint is kept. R1 step 3 becomes: Validation after every round; keep the best of the 6 round checkpoints; no early termination. | D8 |
| A3 | Section 3, Step 4 | Loss normalised by 1/\|ℬ\_r\| | Loss normalised by 1/B (batch size), summing over eligible posts only; an empty ℬ\_r gives exactly 0. | D9 |
| A4 | Section 5, aggregation paragraph | The winner on validation macro-F1 and validation plausibility is frozen; all four are reported in the appendix. | The aggregation with the higher validation composite classification score is frozen (difference within 0.005 → soft mean); all variants run are reported in the appendix. | D4 |
| A5 | Section 7, row 7.2, Priority | Essential | Essential under scope A; under scopes B and C all encoders use lr = 2e-5, stated as a limitation. | D2 |
| A6 | Section 14, "Training" | …batch 32, at most 3 epochs, early stopping on the mean of per-task validation scores, mixed precision, word-level truncation W. | …batch 32, 6 rounds of fixed per-task quotas with exposure-matched subsets, best-of-6 checkpoint selection on the mean of per-task validation scores (no early termination), gradient clipping at norm 1.0, mixed precision, word-level truncation W. | D8 |
| A7 | Section 14, workflow step 3 | (pilot seed not stated) | Append: Pilot runs use seed 1, which no main run uses; no pilot run is reported as a main result. | D10 |
| A8 | Section 14, "Statistics" | paired bootstrap over test instances (2,000 resamples), seed-matched differences averaged over the 3 seeds, 95% CIs, two-sided p-values with Holm correction across the family above; mean ± SD over seeds in every table. | Combined seed and test-instance uncertainty as in R4 with the pooled seed-variance floor of 19.3; 2,000 paired bootstrap resamples of test instances; Holm correction across the confirmatory family; mean ± SD over seeds in every table. | D11 |
| A9 | Section 14, "Planned comparisons" | Five comparison groups, Holm-corrected as one family | Confirmatory family: the 12 tests in R8 item 8.8. All other comparisons, including the five groups formerly listed here, are exploratory. Hypothesis E is judged by its own non-inferiority criteria (R7 as amended by A14), outside the Holm family. | D11, D12 |
| A10 | Section 12, "Decision rule" | If T\_total for the Essential list exceeds 25 GPU hours (5 hours kept as margin), cut in this order: (1) DeBERTa, (2) … (4) the third seed of the encoder comparison. | Usable budget = 30 h quota − 1 h benchmark − 5 h margin = 24 GPU hours. Scope is chosen by the D2 rule: the largest of scopes A, B, C whose required step rate is at most 85% of the B0-measured rate. Optional items run only if the chosen scope leaves budget. | D2 |
| A11 | Section 11, "Checkpoints" row | (no rule for pilot checkpoints) | Append: Pilot checkpoints are deleted after the pilot. | D10 |
| A12 | R3 | Row "Reuse of 4 pilot runs … −24,000"; total 202,896; "Across 45 runs (39 with test evaluation) … ≈ 65,000"; scope table A 303k / B 283k / C 251k | Delete the reuse row; total training 226,896 steps; "Across 49 runs (39 with test evaluation) this is about 6.5 million forward passes, ≈ 68,000 step equivalents"; scope table A ≈ 333,000 (≥ 3.9 / ≥ 1.9 steps/s), B ≈ 313,000 (≥ 3.6 / ≥ 1.8), C ≈ 281,000 (≥ 3.3 / ≥ 1.6). | D10 |
| A13 | R4 | V\_seed = s\_D² / K; claim needs Holm-adjusted combined p < 0.05 and the same sign in 3 of 3 seeds | V\_seed = max(s\_D², 2σ̂²\_pool) / K with the degrees of freedom of the term used (19.3); claim needs Holm-adjusted combined p < 0.05, with same sign in all seeds as a required robustness condition (explicitly not a significance test). | D11 |
| A14 | R5 rule; R7 rows E4, E5 and "Why these numbers" | …above −δ, δ from D12 / δ = 0.02 | …above −δ\_C, with δ\_C and δ\_S derived as in 19.4 (50% preservation of the active control's advantage over random, estimated on validation; sensitivity at 25% and 75%). The sentence justifying δ = 0.02 is deleted. | D12 |
| A15 | Section 0, item 2 | …the mean-pooling, CLS-pooling and μ-sweep runs are cut or moved to the validation pilot. | …the mean-pooling and \[CLS\]-pooling ablations of the full model and the μ-sweep runs are cut or moved to the validation pilot; a single-task \[CLS\] baseline is considered separately (D13). | D13 |
| A16 | Section 15, under the heading | (none) | Insert: Superseded by Sections 18 and 19.9; kept for the record. | — |
| A17 | Section 1: IHC licence cell; issue 1.2 | …download requires a survey \[V\] / …State the access route (survey) in the paper. | Append to the cell: written research permission requested from the authors (19.1). Append to 1.2: Request written permission (19.1); submission is gated on the reply. | D1 |
| A18 | R8 item 8.7; D13 recommendation | …3 seeds if B0 allows, else 1 seed, labelled. / Priority: Essential (at least 1 seed) | …3 seeds if budget allows; otherwise omitted and stated as a limitation, with published \[CLS\] results for context. Priority: Optional, high priority. | D13 |
| A19 | Section 14, "Optional, in priority order" | shared pool without rationales (3); RAMT on the other two encoders (6); … | single-task \[CLS\] baseline, 3 seeds (18,000 steps); 2 extra seeds for M1–M3 (36,000 steps); shared pool without rationales (3); RAMT on the other two encoders (6); … (rest unchanged) | D11, D13 |
| A20 | Section 4, row 4.1; Section 5, row 5.1 | …with 95% CIs excluding zero. / H2 holds only if RAMT beats it on classification or explanation metrics with CIs excluding zero. | 4.1: H3 is judged by the pre-specified decision rule in R5, under the R4 claim rule as amended by A13. 5.1: H2 is judged on the confirmatory tests of R8 item 8.8, under the same claim rule. | D11 |

**Implementation notes (Phases 1–2).** Two numbers in this document are corrected by the implementation, without changing any rule. (1) A1's quotas: with the sizes used here (44,603 / 14,274 / 15,383), round(1,000 · p\_t) gives 464 / 263 / 273, not 465 / 263 / 272; the code computes quotas from the actual prepared training sizes and records them in each run's exposure file. (2) Section 8.5: of the 1,254 HateXplain posts with every word marked, only 96 (76 / 13 / 7 across train / val / test) have a constant soft target, and only those make the random-matched control identical to the human target. Checks on the real HateXplain files also found 6 train–test and 8 train–val near-duplicates in the official splits, which the Section 1 overlap rule removes from train.

**D1 resolved (24 Nov 2026).** The IHC data is licensed CC BY 4.0 (the LICENSE file inside the downloaded corpus); the GitHub repository's own files carry an MIT licence. Earlier statements in this document that no licence is stated are superseded. Approved basis: IHC is used under CC BY 4.0 with attribution, after completing the authors' access survey, with citation of ElSherief et al. (2021). Tweet text is never redistributed (at most post IDs). The licence file, README and survey confirmation are kept as the access record. The pilot is no longer gated on a reply from the authors.

### 19.9 Updated recommendations for the 13 decisions

Four recommendations change and three are refined; mark your choices in the Section 18 dropdowns against these updated versions.

| # | Round-2 recommendation | Round-3 recommendation | Change | Amendments |
| --- | --- | --- | --- | --- |
| D1 | IHC via survey; private upload; IDs only | Same, **plus** a written permission request on day 1 (items 1–6 in 19.1); experiments proceed; submission gated on the reply | Refined | A17 |
| D2 | Largest scope A/B/C within 85% of B0 rate | Same rule with updated requirements: A ≥ 3.9, B ≥ 3.6, C ≥ 3.3 steps/s on 1 GPU (≥ 1.9 / 1.8 / 1.6 per GPU in parallel); 24 usable GPU hours | Refined | A5, A10, A12 |
| D3 | Keep M4 | Unchanged | — | — |
| D4 | Soft for training, majority on validation | Unchanged; selection criterion wording unified | — | A4 |
| D5 | HateXplain bias + HateCheck; Civil Comments optional | Unchanged | — | — |
| D6 | LLM baseline optional (≥ 3 GPU h left) | Unchanged | — | — |
| D7 | Re-plan from B0 | Unchanged | — | — |
| D8 | Rounds with quotas, early stopping after 2 rounds | Rounds with quotas, **no early termination**, best of 6 checkpoints; gradient clipping 1.0 stated | Refined | A1, A2, A6 |
| D9 | Per-batch mean | Unchanged | — | A3 |
| D10 | Reuse pilot runs as seed 42 | **Dedicated pilot seed (seed 1), never reported; all main runs fresh**; shortened pilot only as a scope-C fallback | **Changed** | A7, A11, A12 |
| D11 | Combined variance, 3-of-3 sign rule, 12-test family | **Pooled seed-variance floor** added; sign rule kept as a robustness condition, not a test; 5 seeds for M1–M3 if the scope allows | **Changed** | A8, A9, A13, A19, A20 |
| D12 | δ = 0.02 | **Margins δ\_C, δ\_S = 50% of IG's validated advantage over random**, separate for comprehensiveness and sufficiency; sensitivity at 25% / 75% | **Changed** | A9, A14 |
| D13 | Approve clarifications; \[CLS\] 3 seeds or 1 seed | Clarifications unchanged; **\[CLS\] baseline: 3 seeds or omit**, optional high priority | **Changed** | A15, A18, A19 |

**Unresolved assumptions that remain after approval:** single-T4 and dual-T4 step rates and quota charging (B0); the Jigsaw toxic-row count and the released IHC counts (G1); whether the covariance between paired seeds is non-negative (reported from the runs); the authors' reply on IHC permission.

## Sources

Pages opened or returned by search during this review (24 Nov 2026). Items marked "to cite; not re-opened" in the text are not listed here and must be checked before citing.

- [HateXplain paper (arXiv 2012.10289)](https://arxiv.org/pdf/2012.10289) — platforms, majority labels, 919 dropped posts, rationale collection, Borkan bias protocol
- [HateXplain dataset card (Hugging Face)](https://huggingface.co/datasets/hatexplain/blob/main/README.md) — split sizes, 8:1:1 divisions, licence metadata
- [HateXplain GitHub repository](https://github.com/hate-alert/HateXplain) — MIT licence
- [Disagreeing Rationales (arXiv 2605.31563)](https://arxiv.org/pdf/2605.31563) — agreement α = 0.42 labels, 0.49 rationales
- [Latent Hatred paper (arXiv 2109.05322)](https://arxiv.org/pdf/2109.05322) — stage-1 MTurk annotation, 95.3% majority agreement
- [IHC dataset card (Hugging Face)](https://huggingface.co/datasets/SALT-NLP/ImplicitHate/blob/main/README.md) — 22,056 tweets, 6,346 implicit, survey access
- [IHC GitHub repository](https://github.com/salt-nlp/implicit-hate) — MIT repo licence, data licence unstated
- [Wikipedia Toxicity Subtypes (TFDS)](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes) — Jigsaw labels, test rows not used for scoring
- [Jigsaw dataset licence (Hugging Face loader)](https://huggingface.co/datasets/jigsaw_toxicity_pred/blob/main/jigsaw_toxicity_pred.py) — CC0, comment text under Wikipedia CC-BY-SA 3.0
- [OLID split counts (arXiv 2203.11331)](https://arxiv.org/pdf/2203.11331) — 13,240 / 860; test 240 OFF, 620 NOT
- [HateCheck dataset card](https://huggingface.co/datasets/Paul/hatecheck) and [HateCheck paper](https://arxiv.org/pdf/2012.15606) — 3,728 cases, 29 tests, CC-BY-4.0, 18 vs 19 discrepancy
- [Jigsaw Unintended Bias / Civil Comments card](https://huggingface.co/datasets/jigsaw_unintended_bias/blob/main/README.md) — CC0, 97,320 public test rows
- [Nuanced bias metrics (Borkan et al., 2019)](https://research.google/pubs/nuanced-metrics-for-measuring-unintended-bias-with-real-data-for-text-classification/)
- [HateBERT model card](https://huggingface.co/GroNLP/hateBERT) and [HateBERT paper (WOAH 2021)](https://preview.aclanthology.org/author-url/2021.woah-1.3/)
- [DeBERTa-v3-base model card](https://huggingface.co/microsoft/deberta-v3-base) — 86M backbone, 98M embedding parameters, MIT
- [Masked Rationale Prediction (COLING 2022)](https://arxiv.org/pdf/2211.00243) — description of BERT-HateXplain attention supervision
- [SRA: Aligning Attention with Human Rationales (arXiv 2511.07065)](https://arxiv.org/html/2511.07065v1)
- [Exploring HateXplain with BERT (arXiv 2208.04489)](https://arxiv.org/pdf/2208.04489) — conservative, lenient and mean rationale aggregation
- [ERASER benchmark (ACL 2020)](https://aclanthology.org/2020.acl-main.408/)
- Kaggle limits: [aimultiple GPU cluster overview](https://aimultiple.com/gpu-cluster), [gpuperhour free GPUs 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- EAAI author guide: [Guide for authors](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors)

Added in round 2 (IHC verification):

- [SALT-NLP/ImplicitHate file list](https://huggingface.co/datasets/SALT-NLP/ImplicitHate/tree/main) — implicit\_hate.csv, no licence field
- [metaeval/implicit-hate-stg1 mirror](https://huggingface.co/datasets/metaeval/implicit-hate-stg1) — third-party stage-1 copy
- [Master thesis, VU Amsterdam 2025](https://home.cltl.labs.vu.nl/static/data/theses/thesis_Victoria_Im_2025.pdf) — released stage-1 counts 13,291 / 7,100 / 1,089
- [SHIELD (arXiv 2403.12403)](https://arxiv.org/pdf/2403.12403) — 20,391 posts, 7,100 hateful; LLM-extracted rationales
- [Implied-statement alignment (arXiv 2606.18852)](https://arxiv.org/pdf/2606.18852) — IHC used as 18,666 posts after filtering

Added in round 3 (statistics and margins):

- [FDA non-inferiority guidance, as summarised in a trial statistical analysis plan](https://cdn.clinicaltrials.gov/large-docs/33/NCT04255433/SAP_001.pdf) — M1 / M2 margins, 50% preservation
- [Choosing non-inferiority margins (PMC review)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5510081) — preserved fraction of M1
- [FDA guidance page, Non-Inferiority Clinical Trials (Nov 2016)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials)
- [Statistical significance in explanation evaluation (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.504.pdf) — 25.30% of 494 papers use statistical testing
- [ICE: Intervention-Consistent Explanation Evaluation (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.1055.pdf) — ERASER reports raw scores without testing
- [Latent Hatred arXiv record](https://arxiv.org/abs/2109.05322v1) — submitter and author list
