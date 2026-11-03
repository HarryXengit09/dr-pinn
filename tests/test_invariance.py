"""Scaling invariances of the Newtonian two-pressure model, and what breaks them.

Result: with uniform Newtonian viscosity, one pressure at all inlets and one at
all outlets, every segment's WSS = (P_in - P_out) * f_e(shape). The within-eye
ranking therefore cannot depend on blood pressure, viscosity, image scale, or a
uniform multiplicative radius bias. Monte Carlo over those parameters alone
would report perfect rank stability by construction, not as evidence.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from drpinn.hemodynamics import BoundaryConditions, random_tree, solve_network

MU = 3.5e-3


@pytest.fixture(scope="module")
def syn():
    return random_tree(8, seed=7, n_loops=8)


def _solve(syn, p_in=6000.0, p_out=2000.0, mu=MU, net=None):
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, p_in, syn.outlet_nodes, p_out)
    return solve_network(net or syn.network, bc, mu)


def test_wss_is_linear_in_driving_pressure(syn):
    base = _solve(syn, 6000.0, 2000.0)
    other = _solve(syn, 9000.0, 1000.0)  # dP 4000 -> 8000
    np.testing.assert_allclose(other.wss_pa, 2.0 * base.wss_pa, rtol=1e-9)


def test_wss_independent_of_outlet_offset(syn):
    np.testing.assert_allclose(_solve(syn, 6000.0, 2000.0).wss_pa,
                               _solve(syn, 7000.0, 3000.0).wss_pa, rtol=1e-9)


def test_wss_and_pressure_independent_of_uniform_viscosity(syn):
    a, b = _solve(syn, mu=2e-3), _solve(syn, mu=5e-3)
    np.testing.assert_allclose(a.pressure_pa, b.pressure_pa, rtol=1e-9)
    np.testing.assert_allclose(a.wss_pa, b.wss_pa, rtol=1e-9)
    np.testing.assert_allclose(a.flow_m3_s, 2.5 * b.flow_m3_s, rtol=1e-9)


def test_wss_independent_of_image_scale(syn):
    base = _solve(syn)
    scaled = _solve(syn, net=syn.network.scaled(0.7))
    np.testing.assert_allclose(scaled.wss_pa, base.wss_pa, rtol=1e-9)
    np.testing.assert_allclose(scaled.pressure_pa, base.pressure_pa, rtol=1e-9)


def test_uniform_multiplicative_radius_bias_preserves_ranking(syn):
    base = _solve(syn)
    biased = _solve(syn, net=syn.network.with_radius(1.2 * syn.network.radius_m))
    np.testing.assert_allclose(biased.wss_pa, 1.2 * base.wss_pa, rtol=1e-9)


def _rank_corr(a, b):
    return spearmanr(a, b).statistic


def test_heterogeneous_perturbations_do_change_ranking(syn):
    """These are the uncertainties a meaningful UQ study must sample."""
    base = _solve(syn).wss_pa
    rng = np.random.default_rng(1)

    outlet_p = {int(o): float(rng.uniform(1000, 3000)) for o in syn.outlet_nodes}
    bc = BoundaryConditions(pressure_pa={int(syn.inlet_nodes[0]): 6000.0, **outlet_p})
    hetero_outlets = solve_network(syn.network, bc, MU).wss_pa

    additive = _solve(syn, net=syn.network.with_radius(syn.network.radius_m + 2e-6)).wss_pa
    noisy_r = syn.network.radius_m * rng.lognormal(0.0, 0.1, syn.network.n_edges)
    noisy = _solve(syn, net=syn.network.with_radius(noisy_r)).wss_pa

    for perturbed in (hetero_outlets, additive, noisy):
        assert _rank_corr(base, perturbed) < 1.0 - 1e-6
