"""Mass conservation, checked with code independent of the solver."""

import math

import numpy as np
import pytest

from drpinn.hemodynamics import (
    BoundaryConditions,
    evaluate_pressure_field,
    random_tree,
    solve_network,
)

MU = 3.5e-3


def _independent_node_balance(net, p, mu=MU):
    out = np.zeros(net.n_nodes)
    for (u, v), r, length in zip(net.edges, net.radius_m, net.length_m):
        q = math.pi * r**4 / (8 * mu * length) * (p[u] - p[v])
        out[u] += q
        out[v] -= q
    return out


@pytest.mark.parametrize("seed", range(20))
def test_random_networks_with_loops_conserve_mass(seed):
    syn = random_tree(8, seed=seed, n_loops=12)
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, 6000.0, syn.outlet_nodes, 2000.0)
    sol = solve_network(syn.network, bc, MU)
    assert sol.max_rel_conservation_error < 1e-10
    assert sol.linear_residual < 1e-10

    balance = _independent_node_balance(syn.network, sol.pressure_pa)
    free = np.setdiff1d(np.arange(syn.network.n_nodes), list(bc.pressure_pa))
    assert np.abs(balance[free]).max() < 1e-10 * sol.total_inflow_m3_s
    assert balance[syn.outlet_nodes].sum() == pytest.approx(-sol.total_inflow_m3_s, rel=1e-10)


def test_matches_independent_dense_solver():
    syn = random_tree(6, seed=3, n_loops=5)
    net = syn.network
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, 6000.0, syn.outlet_nodes, 2000.0)
    sol = solve_network(net, bc, MU)

    n = net.n_nodes
    lap = np.zeros((n, n))
    for (u, v), r, length in zip(net.edges, net.radius_m, net.length_m):
        g = math.pi * r**4 / (8 * MU * length)
        lap[[u, v], [u, v]] += g
        lap[u, v] -= g
        lap[v, u] -= g
    rhs = np.zeros(n)
    for i, val in bc.pressure_pa.items():  # row replacement for Dirichlet nodes
        lap[i] = 0.0
        lap[i, i] = 1.0
        rhs[i] = val
    np.testing.assert_allclose(sol.pressure_pa, np.linalg.solve(lap, rhs), rtol=1e-9)


def test_violation_metrics_detect_bad_pressure_fields():
    """The metrics used to score a PINN must flag an incorrect pressure field."""
    syn = random_tree(7, seed=2, n_loops=3)
    net = syn.network
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, 6000.0, syn.outlet_nodes, 2000.0)
    good = solve_network(net, bc, MU)
    assert good.max_dirichlet_error_pa == 0.0

    k = 5  # an interior junction
    assert k not in bc.pressure_pa
    p = good.pressure_pa.copy()
    p[k] += 10.0
    bad = evaluate_pressure_field(net, bc, MU, p)
    incident = np.flatnonzero((net.edges == k).any(axis=1))
    assert bad.conservation_residual_m3_s[k] == pytest.approx(10.0 * good.conductance[incident].sum())
    assert bad.max_rel_conservation_error > 1e-4
    assert math.isnan(bad.linear_residual)

    p2 = good.pressure_pa.copy()
    p2[0] += 5.0
    assert evaluate_pressure_field(net, bc, MU, p2).max_dirichlet_error_pa == pytest.approx(5.0)


def test_mixed_neumann_and_robin_boundaries_balance():
    syn = random_tree(7, seed=5, n_loops=4)
    rng = np.random.default_rng(0)
    q_in = 5e-11
    robin = {int(o): (float(rng.uniform(1e12, 1e13)), float(rng.uniform(1500, 2500)))
             for o in syn.outlet_nodes}
    bc = BoundaryConditions(inflow_m3_s={0: q_in}, terminal_resistance=robin)
    sol = solve_network(syn.network, bc, MU)
    out = sum((sol.pressure_pa[i] - p_ref) / r for i, (r, p_ref) in robin.items())
    assert out == pytest.approx(q_in, rel=1e-9)
    assert sol.max_rel_conservation_error < 1e-10
