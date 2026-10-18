# Pretrained transformers for single-trial EEG → video retrieval

Our ~5,650 training recordings (6 subjects, 17 posterior channels, 3 s per video) are too few to train a transformer from scratch: one trained only on our data stays at chance. This part of the submission asks whether **large public EEG datasets** make a small transformer work, separately for the two senses the videos address:

- **Video transformer**: pretrained on THINGS-EEG2 (~660,000 single trials of visual responses, recorded at the same 17 channels in the same order as ours), maps 0–0.8 s of EEG into CLIP ViT-B/32 space.
- **Audio transformer**: a time-resolved dual encoder (the sound-only version of the team's CORTIVA model, after Wang & Chen, 2026), pretrained on ~120 h of public speech EEG, matches 3 s of EEG to the sound of each video.

## Notebooks (run in this order)

| notebook | what it does | needs |
|---|---|---|
| `00_prepare_things_eeg2` | downloads THINGS-EEG2 (SHA-256 checked), CLIP ViT-B/32 of the THINGS images and of our video frames | – |
| `01_train_video_transformer` | pretrains on THINGS-EEG2 (4 seeds), validates all variants, saves a final model for every variant | 00 |
| `02_train_audio_transformer` | downloads the speech EEG, computes sound features, pretrains (2 encoders × 4 seeds), validates all variants, saves a final model for every variant | – |
| `03_evaluate_video_transformer` | test set: scores of all variants, submission of the chosen one, failure cases | 01 |
| `04_evaluate_audio_transformer` | the same for the audio transformer | 02 |

Training notebooks (01, 02) never read test data. Test data are read only in 03–05, by models and weights that were fixed on validation before.

## Method in one picture

```
                    pretraining (public data)                     fine-tuning (our training videos)
video transformer   THINGS-EEG2 images -> EEG -> CLIP space       EEG 0-0.8 s -> CLIP of the frames 0-0.7 s
audio transformer   speech EEG <-> speech sound over 3 s          EEG 0-3 s  <-> sound features of the video, 50 Hz
                                                                  ranking: similarity to the 102 candidate videos
```

**Variants** (both transformers): `scratch` (no pretraining, the baseline), `zero-shot` (pretrained model, nothing trained on our data), `frozen` (only the subject-specific channel mapping is trained), `finetune` (everything, from the pretrained weights). Controls: chance level, and for the video transformer a zero-shot run with shuffled channels. Audio: each variant with and without transformer layers on the EEG side.

## Results

Reference numbers from the best run. Mean over subjects; chance top-1 0.010, top-5 0.049.

| model | validation top-1 | validation top-5 | test top-1 | test top-5 |
|---|---|---|---|---|
| video transformer, `finetune` (chosen) | 0.064 | 0.229 | 0.103 | 0.324 |
| video transformer, `scratch` | 0.017 | 0.078 | – | – |
| audio transformer, `scratch` + transformer layers (chosen) | 0.136 | 0.330 | 0.351 | 0.606 |
| audio transformer, `finetune` + transformer layers | 0.134 | 0.327 | – | – |


Validation = 5 folds of 102 held-out training videos (the team-wide `video_splits.json`), ranked among the fold's 102 videos, like the test set. Test = sub-01 to sub-04 (the subjects with test labels).

## Findings

1. **Pretraining on THINGS-EEG2 is what makes the video transformer work**: `finetune` reaches ~3× the top-5 of `scratch`. Without adaptation (`zero-shot`) it barely transfers, so the step from 100-ms object images to natural videos needs fine-tuning of the whole network.
2. **Pretraining on speech EEG does not help the audio transformer** (`finetune` ≈ `scratch`), and the transformer layers add nothing over a convolution. Likely reasons: speech differs from video soundtracks, speech tracking is weaker at our posterior channels (~60 % of the best channels in the public data), and our data already suffice for a small, nearly linear EEG encoder. What transfers depends on whether stimuli and electrode sites match.
3. **Sound identifies the videos better than vision** in these models (test top-1 0.35 vs 0.10). Sound events evoke sharp, time-locked responses over the whole 3 s, which a time-resolved model can match; the video transformer sees only 0.8 s and compares it with one static CLIP vector.


## Failure cases

- **Silent videos**: 34 of the 1,102 videos have no audio track. Their sound features are identical, so the audio model cannot tell them apart. All ranks are computed as in the submission file (stable sort), so such ties are not broken in the model's favour. 05 shows how much the video model covers them.

## Data rules

- Every normalisation (EEG whitening/scaling, feature scaling, PCA) is fitted on the fitting recordings or training videos only; final models store these statistics and apply them to the test EEG.
- Early stopping uses an inner 10 % of the fitting videos; the held-out validation fold is never seen during training.
- Model choices use validation only (`models/chosen.json`). Test labels of sub-01 to sub-04 are used only to report scores.
- The public datasets' own held-out parts (THINGS test images, the last 10 % of each speech stimulus) select the pretraining epoch; they are not challenge data.

## Reproducibility

- Fixed seeds (4 per model, averaged, never selected), fixed folds, fixed inner split. Each run writes `config.json` with all settings, the architecture and the package versions; the evaluation notebooks read the architecture from it.
- Paths default to the GPU server's home (`~/eeg-data/…`, `~/eeg-results/…`) and can be changed with `EEG_DATA_ROOT`, `THINGS_EEG2_ROOT`, `EEG_EMBEDDING_ROOT`, `AUDEEG_ROOT`, `EEG_RESULTS_ROOT`.
- Pretraining checkpoints are stored under a hash of their settings and reused by later runs with the same settings.
- Requirements: Python 3.11, a CUDA GPU, `torch`, `transformers`, `av` (PyAV), `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`.
- The trained models in the best variants are stored in a zip file next to our final model. It uses both transformers. There you can also find the actual weights. 

## Outputs (GPU server)

```
~/eeg-results/pretrained-transformers/
    video_transformer/pretraining/<hash>/seed*.pt            THINGS-EEG2 checkpoints
    video_transformer/<run>/config.json, pretraining.csv, validation/, figures/
                           models/<variant>.pt|.json, models/chosen.json
                           test/  (03: tables, figures, similarity/, submission/)
    audio_transformer/...                                    the same layout, plus features/ (sound features of all videos)
```


## External models and data

Every external model and dataset used here, with its exact identifier. All were released before the hackathon. Nothing else external is used: no other pretrained weights, no other datasets, and from the challenge data only the files given by the organisers.

| name | exact identifier | used for (notebook) | licence |
|---|---|---|---|
| CLIP ViT-B/32 (Radford et al., 2021) | Hugging Face `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (the repository's state since 2024-02-29; 00 also records the loaded revision in `metadata.json`) | image and video-frame embeddings, the target space of the video transformer (00, 01, 03) | none stated on the model card; OpenAI's CLIP repository is MIT-licensed |
| Whisper-small (Radford et al., 2023) | Hugging Face `openai/whisper-small`, revision `973afd24965f72e36ca33b3055d56a652f456b4d` (the repository's state since 2024-02-29) | log-mel front end and encoder layer 12 as sound features (02) | Apache-2.0 (model card) |
| THINGS-EEG2 (Gifford et al., 2022), images from the THINGS database (Hebart et al., 2019) | OSF project `3jk45`: preprocessed 17-channel EEG, component `anp5v` (`sub-01.zip` … `sub-10.zip`); image set, component `y63gw` (`training_images.zip`, `test_images.zip`, `image_metadata.npy`). 00 checks every file against the SHA-256 that the OSF API reports | pretraining of the video transformer (00, 01) | CC BY 4.0 (OSF project) |
| ICASSP 2023 Auditory EEG Decoding Challenge data (KU Leuven; part of SparrKULee, Accou et al., 2024; challenge: Monesi et al., 2024) | `https://homes.esat.kuleuven.be/~spchdata/corpora/auditory_eeg_data/ICASSP-2023-eeg-decoding-challenge-dataset/`: `preprocessed_eeg.zip`, `stimuli.zip`, `preprocessed_stimuli.zip`, downloaded in October 2026 (the server publishes no version numbers or checksums) | pretraining of the audio transformer and the Whisper PCA (02) | CC BY-NC 4.0 |
| CORTIVA (Wang & Chen, 2026) | arXiv:2608.01355, version 2 (2026-10-13) | method only, no weights: row-z-scored score fusion and, via the team's CORTIVA pipeline, the dual-encoder design of the audio transformer | – |

Internal team components, not external: the validation folds `video_splits.json` and the team's implementation of CORTIVA, whose design the audio transformer reduces to sound.

### References

- Accou, B., et al. (2024). SparrKULee: A speech-evoked auditory response repository from KU Leuven, containing the EEG of 85 participants. *Data*, 9(8), 94.
- Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature*, 415, 429–433. (Motivation for the fusion in 05.)
- Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. *NeuroImage*, 264, 119754. doi:10.1016/j.neuroimage.2022.119754
- Hebart, M. N., et al. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLoS ONE*, 14(10), e0223792.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411. (Shrinkage for the EEG whitening in 01.)
- Monesi, M. J., Bollens, L., Accou, B., et al. (2024). Auditory EEG decoding challenge for ICASSP 2023. *IEEE Open Journal of Signal Processing*, 5, 652–661. doi:10.1109/OJSP.2024.3376296
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision. *ICML*.
- Wang, J., & Chen, K. (2026). CORTIVA: Candidate-score fusion of complementary visual teachers for EEG- and MEG-to-image retrieval. arXiv:2608.01355 (v2). doi:10.48550/arXiv.2608.01355


