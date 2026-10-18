from __future__ import annotations

import itertools
import numpy as np
import pytest

from eeg_decoder.cortiva_assignment import (
    assignment_rankings,
    csls_assignment_rankings,
    csls_scores,
    maximum_weight_assignment,
)


def test_csls_scores_match_hand_calculation_without_mutating_input() -> None:
    scores = np.asarray([[3., 1.], [2., 4.]])
    original = scores.copy()
    transformed = csls_scores(scores, k=1)
    np.testing.assert_allclose(transformed, [[0., -5.], [-3., 0.]])
    np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize("k", [0, 3, 1.5, True])
def test_csls_rejects_invalid_neighbour_count(k) -> None:
    with pytest.raises(ValueError):
        csls_scores(np.eye(2), k=k)


@pytest.mark.parametrize("scores", [np.zeros(2), np.empty((0, 2)), np.asarray([[np.nan]])])
def test_csls_rejects_invalid_scores(scores) -> None:
    with pytest.raises(ValueError):
        csls_scores(scores, k=1)


def test_csls_assignment_rankings_are_complete_and_have_unique_first_choices() -> None:
    scores = np.asarray([[9., 8., 1.], [8., 7., 0.], [2., 1., 6.]])
    ids = np.asarray([1001, 1002, 1003])
    rankings = csls_assignment_rankings(scores, ids, k=1)
    assert rankings.dtype == np.int64
    np.testing.assert_array_equal(np.sort(rankings, axis=1), np.tile(ids, (3, 1)))
    np.testing.assert_array_equal(np.sort(rankings[:, 0]), ids)
    np.testing.assert_array_equal(rankings, csls_assignment_rankings(scores, ids, k=1))


def test_assignment_matches_exhaustive_global_optimum() -> None:
    rng = np.random.default_rng(17)
    for size in range(2, 7):
        scores = rng.normal(size=(size, size))
        columns = maximum_weight_assignment(scores)
        expected = max(sum(scores[i, j] for i, j in enumerate(order))
                       for order in itertools.permutations(range(size)))
        assert scores[np.arange(size), columns].sum() == pytest.approx(expected)
        np.testing.assert_array_equal(np.sort(columns), np.arange(size))


def test_global_assignment_avoids_the_greedy_trap() -> None:
    scores = np.asarray([[10., 9.], [8., 0.]])
    np.testing.assert_array_equal(maximum_weight_assignment(scores), [1, 0])


def test_assignment_full_rankings_promote_one_candidate_and_keep_remaining_order() -> None:
    scores = np.asarray([[10., 9., 8.], [9., 0., 1.], [0., 7., 8.]])
    original = scores.copy()
    ids = np.asarray([1003, 1001, 1002])
    rankings = assignment_rankings(scores, ids)
    assert len(np.unique(rankings[:, 0])) == 3
    for row, ranking in zip(scores, rankings):
        ordinary = ids[np.argsort(-row, kind="stable")]
        np.testing.assert_array_equal(ranking[1:], ordinary[ordinary != ranking[0]])
        np.testing.assert_array_equal(np.sort(ranking), np.sort(ids))
    np.testing.assert_array_equal(scores, original)


def test_assignment_ties_are_reproducible_and_row_shifts_do_not_change_optimum() -> None:
    scores = np.zeros((4, 4))
    np.testing.assert_array_equal(maximum_weight_assignment(scores), np.arange(4))
    shifted = scores + np.arange(4)[:, None]
    np.testing.assert_array_equal(maximum_weight_assignment(scores), maximum_weight_assignment(shifted))


@pytest.mark.parametrize("scores", [np.zeros((2, 3)), np.empty((0, 0)),
                                      np.asarray([[np.nan]]), np.asarray([[np.inf]])])
def test_assignment_rejects_non_square_or_nonfinite_scores(scores) -> None:
    with pytest.raises(ValueError):
        maximum_weight_assignment(scores)


def test_assignment_rejects_duplicate_or_mismatched_ids() -> None:
    with pytest.raises(ValueError):
        assignment_rankings(np.eye(2), np.asarray([1, 1]))
    with pytest.raises(ValueError):
        assignment_rankings(np.eye(2), np.asarray([1]))


def test_assignment_is_equivariant_to_non_tied_query_and_candidate_permutations() -> None:
    scores = np.asarray([[3., 5., 0.], [8., 0., 1.], [0., 1., 9.]])
    ids = np.asarray([1001, 1002, 1003])
    baseline = assignment_rankings(scores, ids)
    rows = np.asarray([2, 0, 1]); columns = np.asarray([1, 2, 0])
    permuted = assignment_rankings(scores[np.ix_(rows, columns)], ids[columns])
    np.testing.assert_array_equal(baseline[rows], permuted)
