"""External score families from other team members' frozen models.

coworker-audio: audio-only CORTIVA-style dual encoder with a small EEG transformer
(freeze `audio-dual-encoder-2026-10-18_134354-scratch-cortiva-plus-transformer`).

coworker-video: EEG transformer pretrained on THINGS-EEG2 (Gifford et al., 2022, OSF 3jk45, CC BY 4.0) and
fine-tuned on the training recordings of all six subjects; maps 0-0.8 s of EEG to the CLIP ViT-B/32 image space
(freeze `eeg-transformer-things-2026-10-18_144944-finetune-video-0-0-7-s-causal`).

Both were trained on the official training recordings only; the test EEG is normalised with that subject's
*training* statistics (normalisation.npz). Two modes:

* ``frozen_weights`` (default): recompute the test similarities from the frozen weights with the
  coworker's own code and verify them against the frozen score files.
* ``frozen_scores``: read the frozen score files ``scores/test/sub-XX_tst_similarity.npy``.

Every row is a test recording in original order, every column a candidate 1001..1102; rows are
scored independently.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .stimulus_models import zrow

CANDIDATE_IDS = np.arange(1001, 1103)
KINDS = ("coworker-audio", "coworker-video")
VIDEO_PRETRAINING_DATA = "THINGS-EEG2 (Gifford et al., 2022, NeuroImage 264:119754; OSF 3jk45; CC BY 4.0)"


def _subject_number(subject: str) -> int:
    return int(subject.split("-")[1])


def frozen_score_files(freeze_dir: str | Path, subjects: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    root = Path(freeze_dir) / "scores" / "test"
    out = {}
    for s in subjects:
        scores = np.load(root / f"{s}_tst_similarity.npy", allow_pickle=False).astype(np.float64)
        if scores.shape != (102, 102) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid frozen coworker scores for {s}: {scores.shape}")
        out[s] = scores
    return out


def _load_coworker_module(freeze_dir: Path, data_root: Path, filename: str = "audio_dual_encoder.py",
                          module_name: str = "coworker_audio_dual_encoder"):
    os.environ["EEG_DATA_ROOT"] = str(data_root)          # read by the coworker module at import time
    spec = importlib.util.spec_from_file_location(module_name, freeze_dir / "code" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_against_freeze(out: Mapping[str, NDArray], freeze_dir: Path, subjects: Sequence[str],
                           min_corr: float) -> None:
    reference = frozen_score_files(freeze_dir, subjects)
    for s in subjects:
        r = float(np.corrcoef(zrow(out[s]).ravel(), zrow(reference[s]).ravel())[0, 1])
        if r < min_corr:
            raise RuntimeError(f"Coworker scores for {s} do not reproduce the freeze (r={r:.5f})")


def frozen_weight_scores(freeze_dir: str | Path, data_root: str | Path, subjects: Sequence[str],
                         device: str = "cuda", verify: bool = True, min_corr: float = 0.999
                         ) -> dict[str, NDArray[np.float64]]:
    """Seed ensemble (mean of row z-scores) of the frozen coworker models on the official test EEG."""
    import torch

    freeze_dir, data_root = Path(freeze_dir), Path(data_root)
    ade = _load_coworker_module(freeze_dir, data_root)
    config = json.loads((freeze_dir / "model" / "config.json").read_text(encoding="utf-8"))
    norm = np.load(freeze_dir / "model" / "normalisation.npz", allow_pickle=False)
    dev = device if device != "cuda" else "cuda:0"
    mu, sd = norm["feature_mean"], norm["feature_std"]
    feats = torch.as_tensor((ade.VID_FEATS[ade.ROW_OF[CANDIDATE_IDS]] - mu) / sd, dtype=torch.float32, device=dev)
    models = []
    for seed in config["seeds"]:
        model = ade.DualEncoder(ade.N_FEATURES, ade.N_SLOTS, ade.ENCODER)
        model.load_state_dict(torch.load(freeze_dir / "model" / f"seed{seed}.pt", map_location=dev,
                                         weights_only=True))
        models.append(model.to(dev).eval())
    out = {}
    for s in subjects:
        n = _subject_number(s)
        raw = np.load(data_root / "tst" / f"sub-{n:02d}_tst.npy", allow_pickle=False)
        eeg = ade.bandpass_resample(raw.reshape(-1, raw.shape[-1]), ade.EEG_SFREQ,
                                    device=dev).reshape(len(raw), raw.shape[1], -1)
        mean, scale = norm[f"eeg_mean_sub{n:02d}"], norm[f"eeg_scale_sub{n:02d}"]
        x = torch.as_tensor((eeg - mean) / scale, dtype=torch.float32, device=dev)
        slot = torch.full((len(x),), ade.OUR_SLOT[n], device=dev)
        out[s] = np.mean([zrow(ade.scores(m, x, slot, feats).float().cpu().numpy().astype(np.float64))
                          for m in models], 0)
    if verify:
        _verify_against_freeze(out, freeze_dir, subjects, min_corr)
    return out


def video_frozen_weight_scores(freeze_dir: str | Path, data_root: str | Path, subjects: Sequence[str],
                               device: str = "cuda", verify: bool = True, min_corr: float = 0.999
                               ) -> dict[str, NDArray[np.float64]]:
    """Seed mean of the frozen video EEG-transformer similarities (CLIP ViT-B/32 space, candidate-centred
    targets) on the official test EEG, computed with the coworker's own code. The test EEG passes through the
    subject's frozen training-set whitening and channel z-score (normalisation.npz); nothing is fitted on it."""
    import torch

    freeze_dir, data_root = Path(freeze_dir), Path(data_root)
    etr = _load_coworker_module(freeze_dir, data_root, "eeg_transformer_things.py", "coworker_eeg_transformer_things")
    config = json.loads((freeze_dir / "model" / "config.json").read_text(encoding="utf-8"))
    norm = np.load(freeze_dir / "model" / "normalisation.npz", allow_pickle=False)
    dev = device if device != "cuda" else "cuda:0"
    order = etr.ORDERS[config["chosen"]["channel_order"]]
    targets = torch.as_tensor(etr.VIDEO_TARGETS[etr.WINDOW], device=dev)
    queries = {}
    for s in subjects:
        n = _subject_number(s)
        _, raw = etr.load_challenge(n, "tst")               # per-recording offset removal + FIR decimation
        eeg = torch.as_tensor(raw[:, order], device=dev)
        W = torch.as_tensor(norm[f"whitening_sub{n:02d}"], device=dev)
        mu = torch.as_tensor(norm[f"mean_sub{n:02d}"], device=dev)
        sd = torch.as_tensor(norm[f"std_sub{n:02d}"], device=dev)
        white = torch.einsum("cd,ndt->nct", W, eeg)[:, :, :etr.N_SAMPLES]
        queries[n] = (((white - mu) / sd).float(), None)
    members = []
    for seed in config["seeds"]:
        model = etr.load_frozen_model(seed, dev)
        members.append(etr.similarities(model, queries, CANDIDATE_IDS, targets, dev)["visual"])
        del model
    out = {s: np.mean([m[_subject_number(s)] for m in members], 0).astype(np.float64) for s in subjects}
    if verify:
        _verify_against_freeze(out, freeze_dir, subjects, min_corr)
    return out


def external_scores(spec: Mapping, data_root: str | Path, subjects: Sequence[str], device: str = "cuda"
                    ) -> dict[str, NDArray[np.float64]]:
    kind = spec.get("kind", "coworker-audio")
    if kind not in KINDS:
        raise ValueError(f"Unknown external family kind: {kind}")
    if not spec.get("path"):
        raise ValueError(f"External family {spec.get('name')} needs 'path' (the coworker freeze directory); "
                         "pass --coworker-freeze / --video-freeze or set CORTIVA_COWORKER_FREEZE / "
                         "CORTIVA_VIDEO_FREEZE")
    if spec.get("mode", "frozen_weights") == "frozen_scores":
        return frozen_score_files(spec["path"], subjects)
    if kind == "coworker-video":
        return video_frozen_weight_scores(spec["path"], data_root, subjects, device=device)
    return frozen_weight_scores(spec["path"], data_root, subjects, device=device)


def freeze_provenance(freeze_dir: str | Path) -> dict:
    manifest = json.loads((Path(freeze_dir) / "MANIFEST.json").read_text(encoding="utf-8"))
    features = manifest.get("features", {})
    out = {"name": manifest.get("name"), "whisper": features.get("whisper"),
           "whisper_revision": features.get("whisper_revision"), "pca_source": features.get("pca_source"),
           "seeds": manifest.get("model", {}).get("seeds")}
    if features.get("clip_targets"):                       # video EEG-transformer freeze
        out.update({"clip_targets": features["clip_targets"], "pretraining_data": VIDEO_PRETRAINING_DATA,
                    "source_run": manifest.get("source_run")})
    return out
