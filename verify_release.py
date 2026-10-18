"""Verify payload hashes and reconstruct final v7 fusion and predictions without labels."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from eeg_decoder.stimulus_models import fuse
from eeg_decoder.stimulus_postprocess import build_csls_hungarian_rankings


def main(*, editable_notebook=False):
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        if editable_notebook and item["path"] == "JURY_WALKTHROUGH.ipynb":
            continue
        path = ROOT / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise RuntimeError(f"Payload mismatch: {item['path']}")
    config = json.loads((ROOT / "config/final.json").read_text())
    lock = json.loads((ROOT / "FINAL_CANDIDATE.json").read_text())
    ids = np.asarray(config["candidate_ids"])
    subjects = config["subjects"]
    with np.load(ROOT / "scores/final.npz", allow_pickle=False) as scores:
        for s in subjects:
            internal = fuse([scores[f"{f['name']}__{s}"] for f in config["families"]])
            weights = config["group_weights"]
            fused = fuse([internal, scores[f"coworker-audio__{s}"], scores[f"coworker-video__{s}"]],
                         [weights[n] for n in ("internal", "coworker-audio", "coworker-video")])
            np.testing.assert_allclose(fused, scores[f"fused__{s}"], atol=1e-12, rtol=0)
        rankings = build_csls_hungarian_rankings(scores, subjects, ids, k=config["csls_k"])
    for s in subjects:
        frozen = np.load(ROOT / f"predictions/{s}_ranked_candidate_list.npy", allow_pickle=False)
        assert frozen.shape == (102, 102) and np.issubdtype(frozen.dtype, np.integer)
        np.testing.assert_array_equal(rankings[s], frozen)
        np.testing.assert_array_equal(np.sort(frozen, axis=1), np.tile(ids, (102, 1)))
        assert len(np.unique(frozen[:, 0])) == 102
    path = ROOT / "predictions/submission.zip"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lock["submission_sha256"]
    with zipfile.ZipFile(path) as archive:
        expected = {f"{s}_ranked_candidate_list.npy" for s in subjects[-2:]}
        assert set(archive.namelist()) == expected and len(archive.namelist()) == 2
        for s in subjects[-2:]:
            np.testing.assert_array_equal(np.load(io.BytesIO(archive.read(f"{s}_ranked_candidate_list.npy"))), rankings[s])
    assert len(list((ROOT / "checkpoints").glob("*.pt"))) == sum(len(f["seeds"]) for f in config["families"])
    count = sum(not (editable_notebook and x["path"] == "JURY_WALKTHROUGH.ipynb") for x in manifest["files"])
    print(f"OK: {count} payload hashes; final fusion, six rankings, and locked ZIP reproduced")
    if editable_notebook:
        print("Editable walkthrough excluded from hash check; standalone verify_release.py checks it too.")


if __name__ == "__main__":
    main()
