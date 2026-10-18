Our submission files are `predictions/sub-05_ranked_candidate_list.npy` and
`predictions/sub-06_ranked_candidate_list.npy`, also packaged in `predictions/submission.zip`.


# CORTIVA — final v7 EEG-to-video retrieval

We identify which of 102 three-second videos elicited an EEG recording. Our
submitted system combines **70% internal ensemble + 20% audio + 10% video**, then
applies **CSLS(k=1) + deterministic Hungarian assignment**.

DEV: **72.55% Top-1 (296/408), 93.87% Top-5 (383/408), MRR 0.819577**.
We repeatedly inspected DEV during development, so these results are not a blind
estimate. We do not know our performance on the withheld sub-05/06 correspondences.

## Generate the matrices for jury evaluation

Run the following commands from this repository's root. To generate predictions
from our saved model weights using the official challenge data:

```sh
python -m pip install -r requirements.txt
python scripts/final_v7.py infer --data-root /path/to/challenge --output outputs/infer --device cuda
```

After the command finishes successfully, evaluate these two files against your
withheld sub-05/06 test correspondences:

```text
outputs/infer/sub-05_ranked_candidate_list.npy
outputs/infer/sub-06_ranked_candidate_list.npy
```

Each is an integer **102 × 102 ranking matrix**, not a similarity or probability
matrix. Row `i` corresponds to row `i` of `tst/sub-05_tst.npy` or
`tst/sub-06_tst.npy`, respectively. Column `j` contains the video ID at rank
`j + 1`; column 0 is the first prediction. Every row contains all candidate IDs
1001–1102 exactly once. We preserve the original EEG recording order.
The program does not require or read the withheld test labels; you use those
labels separately to score the matrices.

The selected output directory also contains `submission.zip`, with exactly the
two blind-subject arrays, plus `run.json` with run metadata and a comparison to
our locked predictions. Arrays for sub-01 through sub-04 are written alongside
them. A changed output location changes the directory, not the filenames.

| Execution route | Directory containing both final `.npy` files |
|---|---|
| Command above: inference from weights | `outputs/infer/` |
| CLI exact replay: `--output outputs/replay` | `outputs/replay/` |
| CLI internal retraining: `--output outputs/train` | `outputs/train/` |
| CLI teammate retraining example below | `outputs/retrain-external/` |
| Notebook default Run All: exact replay | `outputs/jury/replay/` |
| Notebook optional `RUN_INFERENCE=True` | `outputs/jury/infer/` |
| Notebook optional `RUN_TRAINING=True` | `outputs/jury/train/` |
| Notebook optional audio/video retraining switches | `outputs/jury/retrain-external/` |
| Already packaged locked predictions | `predictions/` |

The notebook's default path recreates the locked matrices from saved scores.
Enable its inference cell to run the models on raw EEG. Both paths produce the
same file format; numerical differences from a new model run are reported in
that run's `run.json`. An output folder is ready for evaluation after the run
completes successfully.

## Reproduce and inspect the decision

We recommend starting with [our jury walkthrough](JURY_WALKTHROUGH.ipynb). Install
`python -m pip install -r requirements-notebook.txt`, then run
`python -m jupyterlab JURY_WALKTHROUGH.ipynb`. Select a Python 3.11 kernel and
choose **Run All**. The default path needs no challenge data, GPU, training, or
encoder downloads. It reproduces our locked submission and generates seven
figures from the packaged evidence. Inference and 60-member internal
retraining have separate opt-in cells. Recorded sweeps/DEV results are labelled
separately from freshly computed metrics. We include executed outputs so you can
read the figures before running the notebook yourself.

The notebook checks every payload hash except its own editable file; the command
`python verify_release.py` checks the complete original distribution, including
the notebook. Saving local notebook edits will change its distribution hash.

Python **3.11**. For score replay and the numerical decision audit only, NumPy
2.2.6 is sufficient. For model inference/training, install `requirements.txt`;
its PyTorch build matches the recorded CUDA 13.0 environment.

```sh
python -m pip install numpy==2.2.6
python verify_release.py
python scripts/final_v7.py replay --output outputs/replay
python scripts/audit_decisions.py
```

The verifier reconstructs final fusion and all six rankings and checks payload
hashes and ZIP contents. Replay requires byte-identical predictions. The decision
audit recomputes both family searches, fusion-weight curves and CSLS choice using
held-out **training** scores, without DEV labels. It also reports family-removal
comparisons and exact subject-specific query/candidate counts. Its output is
`outputs/decision-audit.json`. Numerical validation records are included specifically
to support decisions; the jury walkthrough wraps the final workflow and contains
no historical research pipeline.

## Infer from saved weights

```sh
python -m pip install -r requirements.txt
python scripts/final_v7.py infer --data-root /path/to/challenge --output outputs/infer --device cuda
```

The data directory contains `trn/`, `tst/`, `stimuli_train/` and `stimuli_test/`.
Inference does not read `tst_stim/`. Public encoders are pinned in
`config/final.json` and feature source constants. Optional `--feature-cache PATH`
reuses stimulus extraction caches; it does not change the model or fit on EEG.
Caches may contain decoded frames and must remain outside the Git payload.
CPU is supported with `--device cpu`; choose a suitable PyTorch installation.
Small floating-point differences across devices are possible.

Our GPU weight-inference check reproduced the Top-1 assignments for all six subjects and
both blind-subject Top-5 sets. Fused score correlations exceed 0.99999994. Lower
rankings differ slightly (and one DEV Top-5 set differs); use `replay` for the
exact locked arrays. Details are in `validation/weight_reproduction.json`.

We also checked a fresh extraction of the ZIP with full saved-weight inference
and bounded real-data training of all 68 model members, including checkpoint
reload and final matrix generation. See [VERIFICATION.md](VERIFICATION.md) for
the completed checks and their limits.

## Retrain

> **Compute cost:** We trained the audio and video models on an HPC computing
> cluster with extensive GPU resources for several hours. Retraining is
> computationally expensive and can take substantially longer on a single GPU.
> The teammate training loops require a CUDA GPU with BF16 support. Saved-weight
> inference and score replay avoid this retraining cost.

### Retrain either teammate model or both

To retrain both teammate models while retaining the supplied internal weights:

```sh
python scripts/final_v7.py infer --retrain-external both --data-root /path/to/challenge --output outputs/retrain-external --device cuda
```

Use `--retrain-external audio` or `--retrain-external video` for just one model.
The unselected groups keep their supplied weights. This fits all four audio
seeds from scratch and/or fine-tunes all four video seeds from the packaged
THINGS-EEG2 starting checkpoints, using the existing inner training-video split
for early stopping. It reuses the packaged stimulus features and transforms;
it does not recreate public-data pretraining or feature extraction from raw media
for these two branches. The internal branch still uses its normal feature path.

After successful completion, the final evaluation matrices are:

```text
outputs/retrain-external/sub-05_ranked_candidate_list.npy
outputs/retrain-external/sub-06_ranked_candidate_list.npy
```

The output also contains `submission.zip` and `run.json`. Its `external/` folder
contains the selected retrained `audio-freeze/` and/or `video-freeze/` payloads:
four checkpoints, fitted training-only normalisation, model code, feature data,
configuration, test similarities and `training.json` with per-seed histories and
timings. These folders are separate from our authoritative shipped checkpoints.
Choose a fresh output folder for a new retraining attempt; existing model folders
are protected against overwrite. We have checked small synthetic GPU refits and
reload, rather than rerunning the expensive full-scale training.

To reuse both completed retrained models without training them again:

```sh
python scripts/final_v7.py infer --data-root /path/to/challenge --output outputs/retrained-infer --device cuda --audio-freeze outputs/retrain-external/external/audio-freeze --video-freeze outputs/retrain-external/external/video-freeze
```

Provide just the relevant override if you retrained only one branch. In the
notebook, section 11 offers `RUN_AUDIO_RETRAIN` and `RUN_VIDEO_RETRAIN` switches;
both default to `False`, and the cell plots the actual new validation histories.

### Retrain our internal ensemble

```sh
python scripts/final_v7.py train --data-root /path/to/challenge --output outputs/train --device cuda
```

This trains the 60 internal members using the explicit frozen configuration and
then uses the supplied audio/video checkpoints. It applies the same final fusion
and assignment. All six subjects' **training** EEG is allowed. Training does not
read test labels and is not expected to produce identical weights across hardware.
We also include external fine-tuning functions in `external/audio-freeze/code/`
(`fit_ours`) and `external/video-freeze/code/` (`train_on`), with their configs,
feature transforms and pretraining checkpoints. Public-data pretraining itself
has not been independently recreated or certified by us.

To train all three branches in one run, add `--retrain-external both`:

```sh
python scripts/final_v7.py train --retrain-external both --data-root /path/to/challenge --output outputs/train-all --device cuda
```

The final sub-05/06 matrices then appear directly in `outputs/train-all/`.
This remains a challenge-data refit using supplied public pretrained components,
not a recreation of the original public-data pretraining.

## Repository contents

| Path | Purpose |
|---|---|
| `JURY_WALKTHROUGH.ipynb` | Executed guided explanation, recomputed decision plots, exact replay and optional model runs |
| `config/final.json` | Single explicit recipe, resolved model defaults, seed lists, fusion and public encoder versions |
| `src/eeg_decoder/` | Final model, features, preprocessing, external adapters and assignment |
| `scripts/final_v7.py` | Replay, inference, internal training and opt-in audio/video retraining |
| `src/eeg_decoder/external_training.py` | Opt-in teammate refits and reloadable training artifacts |
| `scripts/audit_decisions.py` | Recalculate selection evidence from validation scores |
| `checkpoints/`, `external/` | Required weights, transforms and external model code |
| `validation/` | Compact measured ablations and held-out numerical scores; no raw EEG or frames |
| `scores/final.npz` | Exact score-replay reference |
| `predictions/` | Final arrays, ZIP and DEV metrics |
| `tests/` | Final ranking, media decoding and workflow checks |
| `DECISIONS.md` | Parameter-by-parameter rationale and evidence limits |
| `REPORT.md` | Method, results, provenance and evaluation guidance |

Our submission files are `predictions/sub-05_ranked_candidate_list.npy` and
`predictions/sub-06_ranked_candidate_list.npy`, also packaged in `predictions/submission.zip`.
Each array is integer 102×102, preserves original EEG row order and contains each
video ID 1001–1102 once per row. Its locked SHA-256 is in `FINAL_CANDIDATE.json`.

If you obtain this repository through Git, run `git lfs pull` before verification:
our binary weights and score files use Git LFS. The release ZIP contains those
files directly. You can check either distribution with `python verify_release.py`.

We exclude challenge videos, extracted frames and raw EEG. External model
provenance and licence restrictions, including SparrKULee CC BY-NC 4.0 and declared
THINGS-EEG2 CC BY 4.0, are in `REPORT.md` and `PROVENANCE.json`.
