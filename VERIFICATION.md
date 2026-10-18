# How we verified our submission

We distinguish exact score replay, inference from saved weights, and training.
These checks answer different questions; reproducing a submission ZIP does not
by itself establish that a training run will reproduce the same model.

Our recorded model environment is Python 3.11.9, NumPy 2.2.6 and
PyTorch 2.14.1+cu130 on Windows.

## Execution check of the distributed ZIP

We independently unpacked the release ZIP and verified it on an NVIDIA GeForce
RTX 5070 Ti with approximately 16 GiB of GPU memory. The 42-test suite, including
notebook Run All, passed from that extraction. Python source compilation,
`pip check`, and a dry run of the documented model requirements also passed in
our recorded environment.

We then completed saved-weight inference on the actual challenge EEG for all
six subjects. All 612 Top-1 assignments and both blind subjects' Top-5 sets
matched our locked predictions. We decoded a real stimulus video and ran the
pinned Whisper, DINOv2 and CLIP encoders directly to check their feature paths.
The full inference run reused existing stimulus caches.

We also exercised the combined `train --retrain-external both` path with the
actual training EEG, all 60 internal members and all four seeds of each teammate
model. A separate verification harness bounded most loops at two real optimiser
updates. The first audio and video seeds each completed one shortened epoch,
including inner validation and restoration of the best state. In total, the
run completed **209 optimiser updates across 68 members**, with finite gradients
and no GPU-memory errors. The shipped training recipe was not changed.

That bounded run saved all checkpoints and normalisation files, produced valid
102 × 102 matrices for sub-05/06, and successfully reloaded all 68 saved models.
Reloaded fused scores matched the scores calculated before saving within
absolute/relative tolerances of 0.00002. We marked these partial models as
smoke-test-only and kept them outside this release and our submission.

The machine-readable record is
[validation/execution_smoke.json](validation/execution_smoke.json), including
the tested source hashes, per-member checks, matrix validation and scope limits.
We observed no runtime errors in these checks. We did not complete full-length
training to convergence, re-extract all 1102 videos, test an empty public-model
cache download, or certify other hardware/software environments.

## Checks we completed

| Check | What we verified |
|---|---|
| Final release suite | All **42 tests passed**, covering deterministic assignment, valid rankings and ZIPs, media decoding, exact replay, all 60 checkpoint configurations, synthetic internal training followed by inference, opt-in teammate retraining, and the notebook's default Run All path. The new external-training adapter has 94% statement coverage. |
| Teammate refit smoke tests | We ran each real training implementation for one epoch per seed on synthetic EEG, across all four seeds for each model. Saved weights and fitted normalisation reloaded through the production inference adapters and reproduced the newly calculated scores. These small tests do not establish full-scale retraining equivalence. |
| Jury walkthrough | All default cells executed, seven figures rendered, the decision selections were recomputed, and replay matched the locked predictions. Inference and retraining remained off. We include executed notebook outputs. |
| Decision audit | Both recorded greedy family searches and every recorded audio/video weight-curve metric reproduced from training-video scores. The selected shares are 0.70/0.20/0.10, with CSLS k=1. |
| Inference from saved weights | We scored raw EEG for all six subjects with all 60 internal checkpoints and the supplied audio/video checkpoints. All 612 Top-1 assignments matched the locked predictions. |
| Exact score replay | We reconstructed all six ranking arrays and the submission ZIP byte-for-byte. The verifier also checks the packaged file hashes and ZIP contents. |

The decision-audit snapshot is in `validation/decision_audit.json`. You can
recalculate it with `python scripts/audit_decisions.py` or the notebook.

Our locked submission ZIP has SHA-256:

```text
4c56c66178e8e459168f94c11f02534a20a907ea1695b9e985b3b33fb70103e3
```

## Numerical differences in model inference

Our full inference check reused existing stimulus feature caches. It did not
repeat media decoding or public-model feature extraction in that run, and it
used no labels. Fused score correlations with the frozen reference exceed
0.99999994. All six subjects' Top-1 assignments and both blind subjects' Top-5
sets agree. One DEV query's Top-5 set and some lower ranks differ numerically.
We record the detailed comparison in `validation/weight_reproduction.json`.

For exact submitted arrays, use score replay. For a new inference run, the CLI
reports its own differences against the locked predictions. Small differences
across hardware or numerical environments can affect lower ranks.

## How you can check the release

Follow [README.md](README.md) to install the appropriate environment, then run
`python verify_release.py`. For a guided inspection, open
[JURY_WALKTHROUGH.ipynb](JURY_WALKTHROUGH.ipynb) and choose Run All.
Its lightweight dependencies are in `requirements-notebook.txt`.

The standalone verifier checks every original distribution hash, including the
notebook. Because saving notebook outputs or local edits changes that file,
the notebook's own verification excludes only its editable file and checks the
rest of the payload. Numerical fusion, ranking and submission checks still run.

We package final source, weights, transforms, reference scores, predictions,
numerical decision evidence and the walkthrough. We exclude raw EEG, challenge
videos, extracted frames, historical experiment runners and old submissions.
Our root manifest describes the shipped payload; external manifests retain
their source-manifest hashes and supplier declarations.

## Limits of these checks

We have not rerun full-scale training or independently recreated public-data
pretraining for this release. Our synthetic training check validates orchestration
on a small example, not the historical scientific results. The saved ablations
are records of earlier experiments, not new training runs.

The optional teammate retraining controls stay off during notebook Run All.
They use the original training recipes, with added history reporting, and require
a BF16-capable CUDA GPU. Our original audio/video training used extensive GPU
resources on an HPC computing cluster for several hours; full retraining remains
an expensive opt-in task.

The checks described here cover the local release. We have not verified a fresh
remote clone as part of them. Our scientific and provenance limitations remain
as described in [REPORT.md](REPORT.md) and [DECISIONS.md](DECISIONS.md).
