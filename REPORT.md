# Final method and submission report
For a more detailed Report read [the technical report](./technical_report.pdf).

## Approach

We rank 102 three-second audiovisual stimuli for each EEG recording. All six
subjects provide training EEG; test correspondences for subjects 05/06 are
withheld. This evaluates new stimuli for known subjects, not entirely unseen people.

Internal preprocessing filters 0.2–12 Hz, downsamples to 50 Hz, removes the
subject/split set-mean response and scales channels using that split's unlabelled
statistics. Training and test statistics are separate. The two supplied external
families use frozen training normalisation. No model is trained/fine-tuned on test EEG.

Time-resolved audiovisual features feed a causal temporal stimulus encoder;
a shared plus subject-specific spatial projection and temporal filter encode EEG.
Multi-positive contrastive training groups the available subjects viewing the
same video. Matched splice augmentation provides additional training pairs.
Five feature families × twelve seeds are combined with row-z-score averaging.

Final group shares are internal 70%, supplied audio encoder 20%, supplied video
EEG transformer 10%. CSLS with k=1 and deterministic Hungarian matching use the
one-recording-per-candidate set structure. The assigned ID goes first and the
remaining IDs follow stable CSLS order. This is transductive inference, not
independent per-query prediction. Its rule permission is recorded below.

## Evidence and limitations

We explain our parameter choices and their evidence limits in
[DECISIONS.md](DECISIONS.md). You can follow the same reasoning, with generated
plots and optional model runs, in [our walkthrough](JURY_WALKTHROUGH.ipynb).
`scripts/audit_decisions.py` recalculates the training-video validation searches
and weight curves from numerical matrices. The 150-video split was reused for
tuning, so its best scores are optimistic. DEV was repeatedly inspected during
the broader project. Neither is a final blind generalisation estimate.

| Evaluation | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Final DEV, 408 queries / 102 candidates | 72.55% (296) | 93.87% (383) | 0.819577 |
| Training-video validation, independent | 29.88% | 54.11% | 0.417263 |
| Training-video validation, final assignment | 29.15% | 58.25% | 0.425703 |
| Blind sub-05/06 | Unknown | Unknown | Unknown |

Validation holds back 150 video IDs, with subject-specific candidate counts
determined by available recordings; it trains on fewer videos and uses fewer
seeds than the final system. Its absolute accuracy cannot forecast jury accuracy.
Assignment has mixed validation support: better Top-5/MRR here, lower Top-1.

DEV Top-1 per subject: 81, 69, 75 and 71 hits out of 102. There are 112 Top-1
errors and 25 truths outside Top-5. These are observed failures; we have not
established content-specific error categories. We chose v7 over an extra
DEV-selected component because a clean assigned Top-1 gain was not established;
we do not claim it was proven harmful.

## Provenance and rule mapping

We based our submission on the challenge PDF (*Mind Reading Using Machine Learning and EEG*,
National AI Hackathon 2026, pages 3/5/6) requires Python 3.11, source, reproducible
training/inference, non-public model weights, original-order predictions and a
report/slides/presentation. It permits public data/models released before the
hackathon and unlabelled normalisation. Our recorded jury clarification
of 18 October permits reproducible CSLS and non-hardcoded one-to-one matching.
We have not included the original clarification screenshots in this repository.

- Internal public models: Whisper-base, DINOv2-base, CLIP ViT-B/32; exact names
  and resolved commits are in `config/final.json`.
- Audio family: Whisper-small (exact revision in its manifest), with PCA fitted
  on SparrKULee / ICASSP 2023 Auditory EEG Decoding Challenge data, **CC BY-NC 4.0**.
  Credit: Accou et al. (2024), *SparrKULee: A Speech-Evoked Auditory Response
  Repository from KU Leuven, Containing the EEG of 85 Participants*, Data 9(8), 94,
  DOI `10.3390/data9080094`. These materials retain their non-commercial and attribution conditions.
- Video family: supplied THINGS-EEG2 pretraining, Gifford et al. (2022), NeuroImage
  264:119754, OSF `3jk45`, declared **CC BY 4.0**. Four pretraining checkpoints and
  supplied fine-tuning code are included. Exclusive dataset use and full original
  pretraining lineage were not independently audited by us.

No raw challenge EEG, videos or extracted frames are distributed. Numerical
features required by supplied models are retained. No test response templates
are used. `PROVENANCE.json` and external manifests retain supplier declarations;
we do not grant a blanket licence over third-party material.

## Reproducing our submission

`config/final.json` is authoritative; `FINAL_CANDIDATE.json` locks the submission
checksum. `verify_release.py` verifies payload integrity and reconstructs fusion
and final rankings. `scripts/final_v7.py` provides exact score replay, raw-EEG
weight inference and internal training. `scripts/audit_decisions.py` recalculates
selection comparisons using training-video validation only.

`train` refits internal members. With `--retrain-external audio`, `video` or
`both`, either `infer` or `train` also refits the selected teammate models and
uses their new scores in the final predictions. Audio trains from scratch;
video fine-tunes the supplied THINGS-EEG2 checkpoints. We save reusable weights,
training-only normalisation and per-seed histories beneath the run's `external/`
directory. GPU retraining is not expected to be bit-exact.

We originally trained the audio and video models on an HPC computing cluster
with extensive GPU resources for several hours. Repeating this work is
computationally expensive; a single-GPU run may take substantially longer.
The optional teammate training paths require CUDA with BF16 support and reuse
packaged stimulus features. We have not independently recreated the earlier
public-data pretraining or raw feature extraction through these paths.
`VERIFICATION.md` records checks actually performed, including any limits.

Final arrays are original-order integer 102×102 permutations of 1001–1102.
We include both blind-subject arrays in `predictions/submission.zip`. For review,
start with the notebook's default Run All path or the commands in
[README.md](README.md#generate-the-matrices-for-jury-evaluation). After the
documented inference command completes, the two evaluation matrices are
`outputs/infer/sub-05_ranked_candidate_list.npy` and
`outputs/infer/sub-06_ranked_candidate_list.npy`. They contain ranked video IDs,
with one row per original test EEG recording and one column per rank. You can
score them against your withheld labels without providing those labels to our
program. The notebook writes the same filenames beneath `outputs/jury/infer/`
when its optional inference cell is enabled, or `outputs/jury/replay/` for its
default exact replay.
