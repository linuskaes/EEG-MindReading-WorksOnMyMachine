"""Explicit, expensive challenge-data refits of the two teammate model families."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .external_families import _load_coworker_module
from .stimulus_models import zrow

RESOURCE_WARNING = (
    "Our original models were trained on an HPC computing cluster with extensive GPU "
    "resources for several hours. Retraining is computationally expensive and can take "
    "substantially longer on a single GPU. This run repeats challenge-data training, "
    "not public-data pretraining or raw stimulus feature extraction."
)


def require_training_device(device):
    """The original teammate loops use CUDA BF16 autocast and fused optimisers."""
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Teammate retraining requires a CUDA GPU; use --device cuda.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("Teammate retraining requires a CUDA GPU with BF16 support.")


def _normalisation(module, kind, fit_ids, device):
    """Export the same training-only transforms used by the supplied fit functions."""
    import torch
    state = {}
    for sub in module.SUBJECTS:
        ids, eeg = (module.OURS if kind == "audio" else module.RAW)[sub]
        eeg = eeg[np.isin(ids, fit_ids)]
        if kind == "audio":
            mean = eeg.mean(0, keepdims=True)
            state[f"eeg_mean_sub{sub:02d}"] = mean
            state[f"eeg_scale_sub{sub:02d}"] = (eeg - mean).std(axis=(0, 2), keepdims=True)
        else:
            eeg = torch.as_tensor(eeg[:, module.ORDERS["recorded"]], device=device)
            cov = module.ledoit_wolf(eeg.transpose(1, 2).reshape(-1, module.N_CHANNELS))
            w, v = torch.linalg.eigh(cov)
            whitening = v @ torch.diag(w.rsqrt()) @ v.T
            white = torch.einsum("cd,ndt->nct", whitening, eeg)[:, :, :module.N_SAMPLES]
            state[f"whitening_sub{sub:02d}"] = whitening.cpu().numpy()
            state[f"mean_sub{sub:02d}"] = white.mean(dim=(0, 2), keepdim=True).cpu().numpy()
            state[f"std_sub{sub:02d}"] = white.std(dim=(0, 2), keepdim=True, correction=0).cpu().numpy()
    if kind == "audio":
        features = module.VID_FEATS[module.ROW_OF[fit_ids]]
        state["feature_mean"] = features.mean((0, 1))
        state["feature_std"] = features.std((0, 1)) + 1e-6
    if not all(np.isfinite(a).all() for a in state.values()):
        raise ValueError("Non-finite fitted normalisation; check the training EEG.")
    return state


def retrain_family(kind, source, data, destination, subjects, device):
    """Refit four seeds, save a reloadable freeze, and return test similarities.

    Audio uses its scratch recipe and packaged features. Video starts from the
    supplied THINGS-EEG2 checkpoints. Neither route reads test correspondences.
    """
    import torch
    if kind not in ("audio", "video"):
        raise ValueError(f"Unknown teammate family: {kind}")
    source, data, destination = map(Path, (source, data, destination))
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    require_training_device(device)
    dev = "cuda:0" if device == "cuda" else device
    config_path = source / "model/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    filename = "audio_dual_encoder.py" if kind == "audio" else "eeg_transformer_things.py"
    module = _load_coworker_module(source, data, filename, f"retrain_{kind}")
    if kind == "audio" and config["variant"] != "scratch":
        raise ValueError("Only the selected scratch audio recipe is supported.")
    if kind == "video" and config["chosen"]["variant"] != "finetune":
        raise ValueError("Only the selected video fine-tuning recipe is supported.")
    print(RESOURCE_WARNING, flush=True)
    started = time.monotonic()
    module.load_data()
    fit_ids = module.TRAIN_VIDEO_IDS
    for sub in module.SUBJECTS:
        ids = (module.OURS if kind == "audio" else module.RAW)[sub][0]
        if not np.isin(ids, fit_ids).any():
            raise ValueError(f"No training recordings for subject {sub}")
    normalisation = _normalisation(module, kind, fit_ids, dev)
    destination.mkdir(parents=True)
    # A complete inference payload allows reuse through --audio-freeze/--video-freeze.
    for folder in ("code", "features"):
        shutil.copytree(source / folder, destination / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    model_dir = destination / "model"
    model_dir.mkdir()
    if kind == "video":
        shutil.copytree(source / "model/pretrained_things", model_dir / "pretrained_things")
    exported_config = {k: v for k, v in config.items() if k != "validation"}
    (model_dir / "config.json").write_text(json.dumps(exported_config, indent=2) + "\n", encoding="utf-8")
    np.savez(model_dir / "normalisation.npz", **normalisation)
    members, records = [], []
    for seed in config["seeds"]:
        seed_started = time.monotonic()
        print(f"Retraining {kind} seed {seed} ({len(records)+1}/{len(config['seeds'])})", flush=True)
        if kind == "audio":
            model, queries, features, info = module.fit_ours(
                module.VARIANT, module.ENCODER, seed, fit_ids, "test", dev, return_info=True)
            candidates = features[torch.as_tensor(module.ROW_OF[module.TEST_VIDEO_IDS], device=dev)]
            scores = {}
            for subject in subjects:
                sub = int(subject.split("-")[1])
                x = queries[sub][0]
                slot = torch.full((len(x),), module.OUR_SLOT[sub], device=dev)
                scores[subject] = zrow(module.scores(model, x, slot, candidates).float().cpu().numpy().astype(np.float64))
            del features, candidates, x, slot
        else:
            queries, model, info, audio_targets = module.train_on(
                module.VARIANT, seed, fit_ids, module.ORDERS["recorded"], module.WINDOW, "test", dev)
            targets = torch.as_tensor(module.VIDEO_TARGETS[module.WINDOW], device=dev)
            by_subject = module.similarities(model, queries, module.TEST_VIDEO_IDS, targets, dev, audio_targets)["visual"]
            scores = {s: by_subject[int(s.split("-")[1])].astype(np.float64) for s in subjects}
            del targets
        for subject, values in scores.items():
            if values.shape != (102, 102) or not np.isfinite(values).all():
                raise ValueError(f"Invalid retrained scores: {kind} {subject}")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_dir / f"seed{seed}.pt")
        members.append(scores)
        records.append({"seed": seed, "elapsed_seconds": time.monotonic() - seed_started, "training": info})
        del model, queries
        torch.cuda.empty_cache()
    result = {s: np.mean([m[s] for m in members], axis=0) for s in subjects}
    score_dir = destination / "scores/test"
    score_dir.mkdir(parents=True)
    for subject, values in result.items():
        np.save(score_dir / f"{subject}_tst_similarity.npy", values, allow_pickle=False)
    report = {"family": kind, "device": dev, "seeds": records, "fit_video_ids": fit_ids.tolist(),
              "test_labels_used": False, "public_pretraining_repeated": False,
              "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "source_code_sha256": hashlib.sha256((source / "code" / filename).read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - started, "resource_warning": RESOURCE_WARNING}
    # Written last: its presence indicates successful completion of this family.
    (destination / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved retrained {kind} models and scores: {destination}", flush=True)
    return result
