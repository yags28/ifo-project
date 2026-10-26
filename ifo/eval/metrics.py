"""Evaluation metrics. Owner: Manu."""

from __future__ import annotations

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic-time-warping distance between two trajectories (T1, D) and (T2, D), Euclidean step cost.

    Used for path fidelity: compare the achieved object trajectory (e.g. kettle xyz) to the demo's.
    Normalized by the warping-path length so trajectories of different lengths are comparable.
    """
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min((acc[i - 1, j], (i - 1, j)), (acc[i, j - 1], (i, j - 1)), (acc[i - 1, j - 1], (i - 1, j - 1)))
            acc[i, j] = cost[i - 1, j - 1] + prev[0]
            steps[i, j] = steps[prev[1]] + 1
    return float(acc[n, m] / steps[n, m])


def success_rate(successes: list[bool] | np.ndarray) -> tuple[float, float]:
    """Mean success and its standard error."""
    s = np.asarray(successes, dtype=np.float64)
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
