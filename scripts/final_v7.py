"""One final configuration: replay frozen scores, infer weights, or train internal models."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_decoder.stimulus_models import fuse
from eeg_decoder.stimulus_postprocess import build_csls_hungarian_rankings, write_submission


def score_models(config, data, output, device, *, train=False, feature_cache=None,
                 retrain_external=(), external_paths=None):
    import torch
    from eeg_decoder.cortiva_challenge import load_challenge_test_subject, load_challenge_training_subject
    from eeg_decoder.external_families import external_scores
    from eeg_decoder.stimulus_models import DualConfig, DualEncoder, prepare_eeg
    from eeg_decoder.stimulus_pipeline import build_features

    subjects, ids = config["subjects"], np.asarray(config["candidate_ids"])
    test = {s: prepare_eeg(load_challenge_test_subject(data, s).test_eeg, tuple(config["band"])) for s in subjects}
    train_eeg, train_ids = {}, {}
    if train:
        for subject in subjects:
            recording = load_challenge_training_subject(data, subject)
            train_eeg[subject] = prepare_eeg(recording.train_eeg, tuple(config["band"]))
            train_ids[subject] = recording.train_video_ids
        (output / "checkpoints").mkdir(exist_ok=True)
    families = {}
    for family in config["families"]:
        features = build_features(data, feature_cache or output / "features", family["features"], device)
        members = []
        for seed in family["seeds"]:
            cfg = DualConfig(**family["config"], seed=seed)
            model = DualEncoder(features, subjects, cfg, device)
            if train:
                model.fit(train_eeg, train_ids)
                torch.save(model.state(), output / f"checkpoints/{family['name']}-seed{seed}.pt")
            else:
                state = torch.load(ROOT / f"checkpoints/{family['name']}-seed{seed}.pt", map_location=device, weights_only=True)
                if state["subjects"] != subjects or state["config"] != model.state()["config"]:
                    raise ValueError(f"Checkpoint configuration differs from config/final.json: {family['name']} {seed}")
                model.stim.load_state_dict(state["stim"])
                model.eeg.load_state_dict(state["eeg"])
            members.append({s: model.score(test[s], s, ids) for s in subjects})
            del model
        families[family["name"]] = {s: fuse([m[s] for m in members]) for s in subjects}
        print(f"Scored {family['name']}", flush=True)
    internal = {s: fuse([families[f["name"]][s] for f in config["families"]]) for s in subjects}
    external = {}
    for kind in ("audio", "video"):
        name = f"coworker-{kind}"
        source = (external_paths or {}).get(kind) or ROOT / f"external/{kind}-freeze"
        if kind in retrain_external:
            from eeg_decoder.external_training import retrain_family
            external[name] = retrain_family(kind, source, data, output / f"external/{kind}-freeze", subjects, device)
        else:
            external[name] = external_scores({"name": name, "kind": name, "path": str(source),
                                              "mode": "frozen_weights"}, data, subjects, device=device)
    weights = config["group_weights"]
    fused = {s: fuse([internal[s], external["coworker-audio"][s], external["coworker-video"][s]],
                     [weights["internal"], weights["coworker-audio"], weights["coworker-video"]]) for s in subjects}
    return {f"fused__{s}": fused[s] for s in subjects}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "infer", "train"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-cache", type=Path, help="Optional stimulus extraction cache, never shipped in Git")
    parser.add_argument("--retrain-external", choices=("audio", "video", "both"),
                        help="Expensive CUDA/BF16 challenge-data refit; default uses supplied weights")
    parser.add_argument("--audio-freeze", type=Path, help="Reuse an alternative completed audio freeze")
    parser.add_argument("--video-freeze", type=Path, help="Reuse an alternative completed video freeze")
    args = parser.parse_args(argv)
    # Outputs must not overwrite any authoritative release payload.
    out = args.output.resolve()
    if out == ROOT or (ROOT in out.parents and out.relative_to(ROOT).parts[0] != "outputs"):
        parser.error("Inside this repository, outputs must be beneath outputs/")
    if args.mode != "replay" and args.data_root is None:
        parser.error("infer/train require --data-root")
    selected = ["audio", "video"] if args.retrain_external == "both" else ([args.retrain_external] if args.retrain_external else [])
    if args.mode == "replay" and (selected or args.audio_freeze or args.video_freeze):
        parser.error("Replay uses locked scores; model-training options require infer or train")
    for kind in selected:
        if getattr(args, f"{kind}_freeze"):
            parser.error(f"Do not combine --{kind}-freeze with retraining that family")
        if (out / f"external/{kind}-freeze").exists():
            parser.error("Retraining requires a fresh output directory; an external model folder already exists")
    if selected:
        from eeg_decoder.external_training import require_training_device, RESOURCE_WARNING
        try:
            require_training_device(args.device)
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(RESOURCE_WARNING, flush=True)
    config = json.loads((ROOT / "config/final.json").read_text())
    subjects, ids = config["subjects"], np.asarray(config["candidate_ids"], dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "replay":
        with np.load(ROOT / "scores/final.npz", allow_pickle=False) as archive:
            scores = {f"fused__{s}": archive[f"fused__{s}"] for s in subjects}
    else:
        scores = score_models(config, args.data_root, out, args.device, train=args.mode == "train", feature_cache=args.feature_cache,
                              retrain_external=selected, external_paths={"audio": args.audio_freeze, "video": args.video_freeze})
        np.savez(out / "scores.npz", **scores)
    rankings = build_csls_hungarian_rankings(scores, subjects, ids, k=config["csls_k"])
    metadata = write_submission(rankings, out, ids, subjects[-2:])
    for s in subjects[:4]:
        np.save(out / f"{s}_ranked_candidate_list.npy", rankings[s], allow_pickle=False)
    if args.mode == "replay":
        for s in subjects:
            np.testing.assert_array_equal(rankings[s], np.load(ROOT / f"predictions/{s}_ranked_candidate_list.npy"))
        assert (out / "submission.zip").read_bytes() == (ROOT / "predictions/submission.zip").read_bytes()
    comparison = {}
    for s in subjects:
        reference = np.load(ROOT / f"predictions/{s}_ranked_candidate_list.npy", allow_pickle=False)
        comparison[s] = {"top1_changes": int(np.sum(rankings[s][:, 0] != reference[:, 0])),
                         "top5_sets_changed": int(np.sum(np.any(np.sort(rankings[s][:, :5], axis=1) != np.sort(reference[:, :5], axis=1), axis=1))),
                         "ranking_entries_changed": int(np.sum(rankings[s] != reference))}
    (out / "run.json").write_text(json.dumps({"mode": args.mode, "config_sha256": hashlib.sha256((ROOT / "config/final.json").read_bytes()).hexdigest(),
                                             "retrained_external": selected,
                                             "external_paths": {kind: str((out / f"external/{kind}-freeze" if kind in selected else
                                                 getattr(args, f"{kind}_freeze") or ROOT / f"external/{kind}-freeze").resolve())
                                                 for kind in ("audio", "video")},
                                             "submission": metadata, "versus_locked_predictions": comparison}, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
