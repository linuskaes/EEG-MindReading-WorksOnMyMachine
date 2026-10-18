from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def csls_scores(
    scores: NDArray[np.floating], k: int = 1,
) -> NDArray[np.float64]:
    """Apply deterministic cross-domain similarity local scaling (CSLS).

    The row and column neighbourhood penalties are computed from the score
    matrix only.  This is unsupervised post-processing: no labels, fitting, or
    subject-specific constants are used.
    """
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or not np.isfinite(values).all()):
        raise ValueError("CSLS requires a finite nonempty 2-D score matrix")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer neighbour count")
    if k < 1 or k > min(values.shape):
        raise ValueError("k must be between 1 and the smaller matrix dimension")

    row_neighbourhood = np.partition(values, values.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
    column_neighbourhood = np.partition(values, values.shape[0] - k, axis=0)[-k:, :].mean(axis=0)
    return 2.0 * values - row_neighbourhood[:, None] - column_neighbourhood[None, :]


def maximum_weight_assignment(scores: NDArray[np.floating]) -> NDArray[np.int64]:
    """Maximize the total score under one-to-one assignment (Hungarian algorithm).

    All rows belong to one subject/batch with exactly one query per candidate.
    Ties are resolved deterministically in the existing row/column order.
    No labels or tunable parameters are used. Complexity is O(n**3).
    """
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 2 or not values.shape[0]
            or values.shape[0] != values.shape[1] or not np.isfinite(values).all()):
        raise ValueError("Assignment requires a finite nonempty square score matrix")
    cost = -values
    n = len(cost)
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    matched_row = np.zeros(n + 1, dtype=np.int64)
    predecessor = np.zeros(n + 1, dtype=np.int64)
    for row in range(1, n + 1):
        matched_row[0] = row
        column = 0
        minimum = np.full(n + 1, np.inf)
        visited = np.zeros(n + 1, dtype=bool)
        while True:
            visited[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, n + 1):
                if not visited[candidate]:
                    reduced = cost[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        predecessor[candidate] = column
                    if minimum[candidate] < delta:
                        delta = minimum[candidate]
                        next_column = candidate
            for candidate in range(n + 1):
                if visited[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    result = np.empty(n, dtype=np.int64)
    for column in range(1, n + 1):
        result[matched_row[column] - 1] = column - 1
    return result


def assignment_rankings(
    scores: NDArray[np.floating], candidate_ids: NDArray[np.integer],
) -> NDArray[np.int64]:
    """Promote the globally assigned ID to rank 1; keep all remaining ranks stable.

    Each row remains a complete candidate permutation. Uniqueness is enforced
    for the first column only; top-k metrics must be evaluated separately.
    """
    assigned = maximum_weight_assignment(scores)
    ids = np.asarray(candidate_ids)
    if (ids.ndim != 1 or len(ids) != len(assigned)
            or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("candidate_ids must contain one unique integer ID per column")
    ids = ids.astype(np.int64, copy=False)
    ordinary = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")
    result = np.empty_like(ordinary, dtype=np.int64)
    for index, (columns, chosen) in enumerate(zip(ordinary, assigned)):
        result[index, 0] = ids[chosen]
        result[index, 1:] = ids[columns[columns != chosen]]
    return result


def csls_assignment_rankings(
    scores: NDArray[np.floating], candidate_ids: NDArray[np.integer], k: int = 1,
) -> NDArray[np.int64]:
    """Build full CSLS rankings with a unique Hungarian rank-1 assignment."""
    return assignment_rankings(csls_scores(scores, k=k), candidate_ids)
