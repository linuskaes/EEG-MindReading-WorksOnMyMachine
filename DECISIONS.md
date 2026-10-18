# Why these parameters?

We provide our executable recipe in [config/final.json](config/final.json), with
every `DualConfig` default expanded for inspection. We distinguish three
categories: **measured selection**, **fixed engineering/domain choice**, and
**supplied external model setting**. A fixed value is documented, not presented
as an experimentally proven optimum. We performed no new search after locking
our submission. In [our walkthrough](JURY_WALKTHROUGH.ipynb), you can regenerate
the selection plots and inspect recorded training sweeps alongside this rationale.

## Our validation protocol and what you can reproduce

We held out 150 of training video IDs 1–1000 with `default_rng(17)` and fit on
the remaining 850 video IDs. Each subject has a subset of these recordings, so
the actual candidate count is the number of available held-out IDs for that
subject, not necessarily 150. `scripts/audit_decisions.py` writes exact split,
query counts and candidate counts. Seed 17 and 15% were fixed reproducibility
and compute choices, not tuned for their score.

The stored scores have 147, 148, 149, 143, 144 and 124 candidates respectively
for sub-01 through sub-06. We report these actual counts because 150 is the
number of globally held-out video IDs, not the number available for every subject.

The audit uses subject-macro Top-1, Top-5 and MRR. The internal family score
caches use six seeds per family; external refits use four. Full submitted
internal families use twelve seeds. Holdout scores support comparisons, not an
absolute forecast of the final result. The holdout was repeatedly reused for
tuning; it is not an untouched generalisation test. We also repeatedly inspected
DEV during development.

## Ensemble and ranking: directly recomputable selection

| Setting | Reason and numerical evidence |
|---|---|
| Five internal families | We ran greedy forward MRR searches over two candidate sets, then curated their union and pruned dependencies. `validation/family_search.json` contains the tested specs; the audit reruns both searches from numerical scores. The final five are G/O/D/H/I. We selected this set through that combined process, rather than one automatic search. |
| Feature differences | G uses Whisper PCA-32, O adds audio-rich features to that, D includes CLIP, H retains low-level visual features without low-level audio, I adds audio-rich features to PCA-64. All use DINOv2 PCA-16. The audit reports standalone and leave-one-out metrics, exposing complementarity rather than assuming every family is better alone. |
| Excluding the HuBERT extension | The extended greedy search adds it for a reported gain of about 0.0008 MRR; we chose to avoid another model dependency for this small validation gain. This is an explicit engineering tradeoff, not a statistical rejection. Scores remain in validation so it is inspectable. |
| Equal internal family weights | We used equal averaging to avoid fitting another set of weights. We have not established it as optimal. We row-z-score groups and seed members before averaging. |
| Audio share 0.20 | Maximum independent-ranking holdout MRR on the recorded grid: internal-only 0.399629 → 0.404841. Nearby 0.10/0.25 are close; precision of 0.20 is not evidence of a sharp optimum. |
| Video share 0.10, keeping audio 0.20 | Maximum independent-ranking holdout MRR over the recorded grid: 0.404841 → 0.417263; remaining internal share is 0.70. The assigned Top-1-maximising video weight differs (0.05). We retain the declared MRR-based family selection rule rather than retrospectively changing objectives. |
| CSLS k=1 | We compared k ∈ {1,5,10,20} by Top-1, then Top-5, then MRR. You can reproduce this comparison with our audit. Assigned Top-1 ties mathematically; k=1 has the best held-out Top-5. We round metrics to 12 decimals for selection so floating-point noise cannot break a mathematical tie. |
| Hungarian assignment | We use the known one-query-per-candidate structure under the recorded jury permission. No tuned weight or label lookup. It improves observed DEV Top-1, but does not improve final-mix holdout Top-1: independent 0.298793 versus assigned 0.291504. We therefore treat its generalisation benefit as uncertain. |
| Full ranking construction | We place the assigned ID first, then remaining IDs in stable descending CSLS order. This deterministic construction uses no label-dependent repair. |

For square one-to-one assignment, the CSLS row and column penalties add a
constant to the assignment objective. Thus they cannot explain a separate
Hungarian Top-1 gain. Their effect is on remaining ranks (and tie/numerical
behaviour), which matter for Top-5/MRR.

The five-family internal MRR is 0.399629. Removing G/O/D/H/I gives respectively
0.396075 / 0.397496 / 0.398908 / 0.395762 / **0.400127**. Thus removing I slightly
improves this validation metric. The five-family union is a recorded curation
decision, not the optimum of this leave-one-out test. We retained our locked v7
configuration. We include the removal comparison so you can assess this
limitation of our selection directly.

## Internal training: measured searches and fixed settings

`validation/ablations.json` preserves tested configurations, ensemble holdout MRR,
best epochs and original result-file SHA-256. It is a compact numerical record,
not a claim to have rerun those experiments. We explored configurations
sequentially, rather than conducting a fully controlled factorial study.
Because several fields can change between runs, we cannot attribute every
improvement to a single variable.

| Parameters | Status and rationale |
|---|---|
| EEG band 0.2–12 Hz | Measured search for slow stimulus-locked activity; band variants occur in the recorded configurations. Does not establish that all informative EEG is confined to this band. |
| 50 Hz internal grid; 17 channels; three seconds | 17 channels/three seconds come from the task; 50 Hz is a fixed compute/time-resolution choice after low-frequency filtering. No claim it is uniquely optimal. |
| Whisper layers 2/3/4; PCA 32 or 64; DINO/CLIP PCA-16 | Measured feature-family comparisons and curation; maps in `family_search.json` and `final.json`. PCA/scaling fit on training videos only. Public encoder names and resolved commits are explicit. |
| Multi-positive batch of 16 videos | Batch-size and objective exploration in the ablation record; groups available subjects for each video, rather than unrelated trials. `batch=256` remains a resolved field but is inactive when `multipos=16`. |
| Learning rate 0.002; dropout/input dropout 0.5; noise 0 | Appeared in empirical sweeps. Chosen as part of the final recipe, not claimed to be isolated optima across all final families. |
| Splice probability 1.0, one cut; splice interval 0.6–2.4 s | Probability/cut variations were explored; full configurations are recorded. The interval is a fixed interior window leaving useful segments on both sides. EEG splice alignment is an augmentation assumption, not a measured physiological law. |
| Schedule horizon 30 epochs, stop at 22 | A fixed-epoch schedule/stop recipe tested in the later sweeps (including `sweep-ens-e19`). The 30-epoch horizon affects OneCycle even though training stops at 22. Both are necessary for reproduction. |
| Hidden 128; latent 32; 4 temporal blocks; stimulus kernel 5; EEG kernel 9 | Compact architecture defaults, with some larger/deeper alternatives explored. We have not independently isolated evidence for every dimension/kernel. We regard untested dimensions as fixed engineering choices. |
| Weight decay 0.05; gradient clipping 5; initial temperature 0.07; temperature cap 100; OneCycle warmup 10% | Fixed stabilisation/optimisation settings, not separately optimised or causally justified by the available ablations. Temperature is learned during training. Inference does not use it. |
| 12 seeds per internal family; explicit seed offsets | Compute-budget variance reduction, not a proven optimal ensemble size. Offsets create distinct runs; the seed numbers themselves have no scientific significance. Validation used six seeds per family. |
| Subject-specific + shared spatial projection | Fixed model structure for known subjects with training EEG; no zero-shot-subject claim. |
| Row z-score; candidate embedding centring; epsilons | Fixed score-scale and numerical-stability conventions. Epsilons prevent division by zero; they were not searched against labels. Exact operations are in model/feature source. |

Remaining feature extraction constants (frame grid, audio sampling, FFT/window
sizes, PCA epsilon) are explicit in `stimulus_features.py`. They are fixed signal
processing choices of the frozen recipe, not separately validated hyperparameter
optima. The evidence supports the tested recipe and comparisons; it cannot supply
retrospective experimental support for every implementation constant.

## Supplied model settings and the final choice

We integrated the supplied audio and video models using their frozen
architecture, training and normalisation recipes, visible in
`external/*-freeze/model/config.json` and their code. We validated their contribution by refitting on our training-video
split and varying their shares. We did not independently tune or prove every
external hyperparameter. Supplied checkpoints and feature transforms are included.

You can repeat their challenge-data fitting with `--retrain-external audio`,
`video` or `both`, or use the independent switches in notebook section 11.
Audio uses the selected scratch recipe; video starts from the supplied
THINGS-EEG2 checkpoints. The new runs retain the recorded seed lists, training
settings and inner early-stopping split. This adds no new hyperparameter search.
We save the new checkpoints, fitted normalisation and actual validation histories
without replacing the locked submission. Our original training took several
hours on an HPC computing cluster with extensive GPU resources; retraining can
be expensive on a single GPU. Public-data pretraining is not repeated by this option.

We locked v7 rather than adding the extra DEV-selected v8 model. Its cleaner
comparison did not establish an assigned Top-1 gain, although Top-5/MRR improved.
That is a conservative team decision under uncertainty; it does not prove v8
would hurt the blind score. No v8 weights, scores or pipeline are included.
