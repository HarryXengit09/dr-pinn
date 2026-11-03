"""Tie-aware within-eye ranking utilities.

Symmetric or near-symmetric vessel trees produce many exactly equal values;
floating-point noise would otherwise reorder ties arbitrarily (seen in the M1
demo). Values within `rtol * max|values|` of their neighbour in sorted order
are treated as tied and share their average rank.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_RTOL = 1e-9


def tie_aware_ranks(values: ArrayLike, rtol: float = DEFAULT_RTOL) -> NDArray[np.float64]:
    """Ascending ranks 1..n; tied values share the average rank."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("values must be a non-empty array of finite numbers")
    order = np.argsort(v, kind="stable")
    sv = v[order]
    tol = rtol * max(float(np.abs(sv).max()), np.finfo(float).tiny)
    group_start = np.concatenate([[True], np.diff(sv) > tol])
    group_id = np.cumsum(group_start) - 1
    positions = np.arange(1, v.size + 1, dtype=np.float64)
    avg = np.bincount(group_id, weights=positions) / np.bincount(group_id)
    ranks = np.empty(v.size)
    ranks[order] = avg[group_id]
    return ranks


def percentile_rank(values: ArrayLike, rtol: float = DEFAULT_RTOL) -> NDArray[np.float64]:
    """Within-eye percentile in [0, 1] (1 = highest). All-tied input gives 0.5."""
    ranks = tie_aware_ranks(values, rtol)
    n = ranks.size
    return np.full(n, 0.5) if n == 1 else (ranks - 1.0) / (n - 1.0)


def top_fraction_mask(values: ArrayLike, fraction: float = 0.1,
                      rtol: float = DEFAULT_RTOL) -> NDArray[np.bool_]:
    """Segments whose tie-aware rank is in the top `fraction` (ties at the cut are included)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ranks = tie_aware_ranks(values, rtol)
    k = max(1, int(round(fraction * ranks.size)))
    threshold = np.sort(ranks)[-k]
    return ranks >= threshold


def spearman(a: ArrayLike, b: ArrayLike, rtol: float = DEFAULT_RTOL) -> float:
    """Spearman correlation on tie-aware ranks. NaN if either input is constant."""
    ra, rb = tie_aware_ranks(a, rtol), tie_aware_ranks(b, rtol)
    if ra.shape != rb.shape:
        raise ValueError("inputs must have the same length")
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_set_overlap(reference: ArrayLike, other: ArrayLike, fraction: float = 0.1,
                    rtol: float = DEFAULT_RTOL) -> float:
    """Fraction of the reference top set that is still in the other top set."""
    ref, oth = top_fraction_mask(reference, fraction, rtol), top_fraction_mask(other, fraction, rtol)
    return float((ref & oth).sum() / ref.sum())
