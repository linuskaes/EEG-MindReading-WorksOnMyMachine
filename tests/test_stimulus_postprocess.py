from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pytest

from eeg_decoder.stimulus_postprocess import (
    build_csls_hungarian_rankings,
    retrieval_metrics,
    write_submission,
)


def test_build_rankings_uses_each_subject_fused_matrix() -> None:
    archive = {
        "fused__sub-05": np.asarray([[4., 3.], [3., 2.]]),
        "fused__sub-06": np.asarray([[0., 5.], [4., 1.]]),
    }
    ids = np.asarray([1001, 1002])
    result = build_csls_hungarian_rankings(archive, ["sub-05", "sub-06"], ids, k=1)
    assert set(result) == {"sub-05", "sub-06"}
    for ranking in result.values():
        np.testing.assert_array_equal(np.sort(ranking, axis=1), np.tile(ids, (2, 1)))
        np.testing.assert_array_equal(np.sort(ranking[:, 0]), ids)


def test_build_rankings_rejects_missing_subject_scores() -> None:
    with pytest.raises(KeyError, match="fused__sub-06"):
        build_csls_hungarian_rankings(
            {"fused__sub-05": np.eye(2)}, ["sub-05", "sub-06"], np.asarray([1, 2]), k=1,
        )


def test_retrieval_metrics_are_exact() -> None:
    rankings = np.asarray([[3, 1, 2], [2, 3, 1], [1, 2, 3]])
    labels = np.asarray([3, 1, 2])
    metrics = retrieval_metrics(rankings, labels)
    assert metrics == {
        "queries": 3,
        "top1_hits": 1,
        "top1": pytest.approx(1 / 3),
        "top5_hits": 3,
        "top5": 1.0,
        "mrr": pytest.approx((1 + 1 / 3 + 1 / 2) / 3),
        "mean_rank": pytest.approx((1 + 3 + 2) / 3),
    }


def test_write_submission_writes_exact_zip_members_and_hashes(tmp_path) -> None:
    ids = np.asarray([1001, 1002])
    rankings = {
        "sub-05": np.asarray([[1001, 1002], [1002, 1001]], dtype=np.int64),
        "sub-06": np.asarray([[1002, 1001], [1001, 1002]], dtype=np.int64),
    }
    metadata = write_submission(rankings, tmp_path, ids, ["sub-05", "sub-06"])
    expected = ["sub-05_ranked_candidate_list.npy", "sub-06_ranked_candidate_list.npy"]
    with zipfile.ZipFile(tmp_path / "submission.zip") as archive:
        assert archive.namelist() == expected
        for name in expected:
            raw = (tmp_path / name).read_bytes()
            assert archive.read(name) == raw
            assert metadata["sha256"][name] == hashlib.sha256(raw).hexdigest()

