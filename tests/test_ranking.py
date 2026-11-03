import math

import numpy as np
import pytest

from drpinn.uncertainty import percentile_rank, spearman, tie_aware_ranks, top_fraction_mask, top_set_overlap


def test_ranks_and_ties():
    np.testing.assert_array_equal(tie_aware_ranks([3.0, 1.0, 2.0]), [3, 1, 2])
    np.testing.assert_array_equal(tie_aware_ranks([1.0, 1.0, 2.0]), [1.5, 1.5, 3])
    # floating-point near-ties are grouped
    np.testing.assert_array_equal(tie_aware_ranks([1.0, 1.0 + 1e-14, 2.0]), [1.5, 1.5, 3])


def test_percentile_rank():
    np.testing.assert_allclose(percentile_rank([10.0, 30.0, 20.0]), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(percentile_rank([5.0, 5.0]), [0.5, 0.5])


def test_top_fraction_includes_ties_at_the_cut():
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9.0])
    assert top_fraction_mask(v, 0.1).sum() == 2  # both 9s
    assert top_fraction_mask(np.arange(20.0), 0.1).tolist() == [False] * 18 + [True] * 2


def test_spearman_and_overlap():
    x = np.random.default_rng(0).normal(size=50)
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(spearman(x, np.ones(50)))
    assert top_set_overlap(x, np.exp(x)) == 1.0
    assert top_set_overlap(x, -x) == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        tie_aware_ranks([])
    with pytest.raises(ValueError):
        tie_aware_ranks([1.0, math.nan])
    with pytest.raises(ValueError):
        top_fraction_mask([1.0, 2.0], 0.0)
