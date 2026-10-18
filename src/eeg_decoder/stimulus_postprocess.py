"""Reproducible, label-free post-processing for stimulus score matrices."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cortiva_assignment import csls_assignment_rankings


def build_csls_hungarian_rankings(
    score_archive: Mapping[str, NDArray[np.floating]],
    subjects: Sequence[str],
    candidate_ids: NDArray[np.integer],
    *,
    k: int = 1,
) -> dict[str, NDArray[np.int64]]:
    """Transform each fused score matrix and create complete assigned rankings."""
    result: dict[str, NDArray[np.int64]] = {}
    for subject in subjects:
        key = f"fused__{subject}"
        if key not in score_archive:
            raise KeyError(f"Missing score matrix: {key}")
        result[subject] = csls_assignment_rankings(score_archive[key], candidate_ids, k=k)
    return result


def retrieval_metrics(
    rankings: NDArray[np.integer], labels: NDArray[np.integer],
) -> dict[str, int | float]:
    """Compute exact retrieval metrics for one subject."""
    ranked = np.asarray(rankings)
    truth = np.asarray(labels)
    if ranked.ndim != 2 or truth.ndim != 1 or len(ranked) != len(truth):
        raise ValueError("rankings and labels have incompatible shapes")
    matches = ranked == truth[:, None]
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("each label must occur exactly once in its ranking")
    ranks = np.argmax(matches, axis=1) + 1
    top1_hits = int(np.sum(ranks <= 1))
    top5_hits = int(np.sum(ranks <= min(5, ranked.shape[1])))
    return {
        "queries": int(len(ranks)),
        "top1_hits": top1_hits,
        "top1": float(top1_hits / len(ranks)),
        "top5_hits": top5_hits,
        "top5": float(top5_hits / len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def write_submission(
    rankings: Mapping[str, NDArray[np.integer]],
    output_dir: Path,
    candidate_ids: NDArray[np.integer],
    subjects: Sequence[str],
) -> dict[str, object]:
    """Write checked NPY files and a byte-for-byte deterministic submission ZIP."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    hashes: dict[str, str] = {}
    members: list[str] = []
    for subject in subjects:
        if subject not in rankings:
            raise KeyError(f"Missing ranking: {subject}")
        ranked = np.asarray(rankings[subject], dtype=np.int64)
        if ranked.shape != (len(ids), len(ids)):
            raise ValueError(f"{subject} ranking has shape {ranked.shape}")
        if not np.all(np.sort(ranked, axis=1) == np.sort(ids)[None, :]):
            raise ValueError(f"{subject} rows are not complete candidate permutations")
        name = f"{subject}_ranked_candidate_list.npy"
        path = destination / name
        np.save(path, ranked, allow_pickle=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        members.append(name)

    with zipfile.ZipFile(destination / "submission.zip", "w") as archive:
        for name in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, (destination / name).read_bytes())
    return {
        "members": members,
        "sha256": hashes,
        "zip_sha256": hashlib.sha256((destination / "submission.zip").read_bytes()).hexdigest(),
    }

