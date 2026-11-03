"""Smoke test: the M2 rank-sensitivity study runs and its controls behave as proved."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m2_rank_sensitivity.py"


@pytest.fixture(scope="module")
def results():
    spec = importlib.util.spec_from_file_location("m2_rank_sensitivity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(trees=2, samples=3, generations=5, seed=1)


def test_controls_are_exactly_invariant(results):
    controls = [r for r in results["results"] if r["group"].startswith("control")]
    assert len(controls) == 4
    for r in controls:
        assert r["spearman_p05"] == pytest.approx(1.0, abs=1e-12)
        assert r["top10_overlap_p05"] == 1.0


def test_all_metrics_are_valid(results):
    for r in results["results"]:
        assert -1.0 <= r["spearman_p05"] <= r["spearman_mean"] <= 1.0
        assert 0.0 <= r["top10_overlap_p05"] <= r["top10_overlap_mean"] <= 1.0
    assert "SYNTHETIC" in results["note"]
