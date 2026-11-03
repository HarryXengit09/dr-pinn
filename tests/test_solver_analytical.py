"""Solver vs closed-form Hagen-Poiseuille network results."""

import math

import numpy as np
import pytest

from drpinn.hemodynamics import (
    BoundaryConditions,
    VesselNetwork,
    murray_tree,
    random_tree,
    series_chain,
    single_pipe,
    solve_network,
)
from drpinn.hemodynamics.units import mmhg_to_pa

MU = 3.5e-3  # Pa s, illustrative Newtonian value for tests only


def resistance(r, length, mu=MU):
    return 8 * mu * length / (math.pi * r**4)


def test_single_pipe_matches_hagen_poiseuille():
    length, r = 1e-3, 50e-6
    dp = mmhg_to_pa(10)
    sol = solve_network(single_pipe(length, r).network,
                        BoundaryConditions.inlet_outlet_pressure([0], dp, [1], 0.0), MU)
    q = math.pi * r**4 * dp / (8 * MU * length)
    # Independent hand calculation guards against a typo shared by code and test.
    assert q == pytest.approx(9.3492e-10, rel=1e-4)
    assert sol.flow_m3_s[0] == pytest.approx(q, rel=1e-12)
    assert sol.wss_pa[0] == pytest.approx(r * dp / (2 * length), rel=1e-12)  # 33.33 Pa
    assert sol.mean_velocity_m_s[0] == pytest.approx(q / (math.pi * r**2), rel=1e-12)
    assert sol.wall_shear_rate_s[0] == pytest.approx(sol.wss_pa[0] / MU, rel=1e-12)


def test_series_resistances_add():
    lengths, radii = [1e-3, 2e-3, 0.5e-3], [60e-6, 40e-6, 50e-6]
    syn = series_chain(lengths, radii)
    p_in, p_out = 5000.0, 2000.0
    sol = solve_network(syn.network,
                        BoundaryConditions.inlet_outlet_pressure([0], p_in, [3], p_out), MU)
    rs = [resistance(r, length) for r, length in zip(radii, lengths)]
    q = (p_in - p_out) / sum(rs)
    np.testing.assert_allclose(sol.flow_m3_s, q, rtol=1e-12)
    expected_p = p_in - q * np.concatenate([[0.0], np.cumsum(rs)])
    np.testing.assert_allclose(sol.pressure_pa, expected_p, rtol=1e-12)


def test_parallel_multi_edges_share_pressure_drop():
    radii = np.array([30e-6, 45e-6, 60e-6])
    net = VesselNetwork(2, [[0, 1]] * 3, [1e-3] * 3, radii)
    dp = 1500.0
    sol = solve_network(net, BoundaryConditions(pressure_pa={0: dp, 1: 0.0}), MU)
    np.testing.assert_allclose(sol.flow_m3_s, dp / resistance(radii, 1e-3), rtol=1e-12)


def test_y_junction_pressure_is_conductance_weighted_mean():
    net = VesselNetwork(4, [[0, 1], [1, 2], [1, 3]], [1e-3, 0.8e-3, 1.5e-3],
                        [60e-6, 45e-6, 40e-6])
    p0, p2, p3 = 6000.0, 2500.0, 2000.0
    sol = solve_network(net, BoundaryConditions(pressure_pa={0: p0, 2: p2, 3: p3}), MU)
    g = 1 / resistance(net.radius_m, net.length_m)
    p1 = (g[0] * p0 + g[1] * p2 + g[2] * p3) / g.sum()
    assert sol.pressure_pa[1] == pytest.approx(p1, rel=1e-12)
    assert sol.flow_m3_s[0] == pytest.approx(sol.flow_m3_s[1] + sol.flow_m3_s[2], rel=1e-12)


def test_prescribed_inflow_neumann():
    length, r, q = 1e-3, 50e-6, 2e-11
    sol = solve_network(single_pipe(length, r).network,
                        BoundaryConditions(inflow_m3_s={0: q}, pressure_pa={1: 1000.0}), MU)
    assert sol.flow_m3_s[0] == pytest.approx(q, rel=1e-12)
    assert sol.pressure_pa[0] == pytest.approx(1000.0 + q * resistance(r, length), rel=1e-12)


def test_terminal_resistance_robin():
    length, r = 1e-3, 50e-6
    r_pipe = resistance(r, length)
    r_term, p_ref, p_in = 3 * r_pipe, 2000.0, 6000.0
    bc = BoundaryConditions(pressure_pa={0: p_in}, terminal_resistance={1: (r_term, p_ref)})
    sol = solve_network(single_pipe(length, r).network, bc, MU)
    assert sol.flow_m3_s[0] == pytest.approx((p_in - p_ref) / (r_pipe + r_term), rel=1e-12)

    # Robin at both ends, no Dirichlet node: still well posed.
    bc2 = BoundaryConditions(terminal_resistance={0: (r_pipe, p_in), 1: (r_term, p_ref)})
    sol2 = solve_network(single_pipe(length, r).network, bc2, MU)
    assert sol2.flow_m3_s[0] == pytest.approx((p_in - p_ref) / (2 * r_pipe + r_term), rel=1e-12)


def test_murray_tree_has_uniform_wss_and_halving_flow():
    gens = 7
    syn = murray_tree(gens, root_radius_m=60e-6)
    net = syn.network
    dp = 4000.0
    sol = solve_network(net, BoundaryConditions.inlet_outlet_pressure(
        syn.inlet_nodes, dp, syn.outlet_nodes, 0.0), MU)

    # Generator sanity: Murray's law r_parent^3 = sum r_child^3 at every split.
    for e in range(net.n_edges):
        kids = np.flatnonzero(syn.parent_edge == e)
        if kids.size:
            assert net.radius_m[e] ** 3 == pytest.approx((net.radius_m[kids] ** 3).sum())

    q_root = dp / sum(resistance(net.radius_m[syn.edge_generation == g][0],
                                 net.length_m[syn.edge_generation == g][0]) / 2**g
                      for g in range(gens))
    np.testing.assert_allclose(sol.flow_m3_s, q_root / 2.0**syn.edge_generation, rtol=1e-10)
    assert np.ptp(sol.wss_pa) / sol.wss_pa.mean() < 1e-10


def test_stenosis_raises_local_wss_and_lowers_downstream_wss():
    """Narrowing gives high shear in the narrowed segment, low shear below it."""
    syn = murray_tree(7, root_radius_m=60e-6)
    bc = BoundaryConditions.inlet_outlet_pressure(syn.inlet_nodes, 4000.0, syn.outlet_nodes, 0.0)
    base = solve_network(syn.network, bc, MU)
    e = int(np.flatnonzero(syn.edge_generation == 2)[0])
    radius = syn.network.radius_m.copy()
    radius[e] *= 0.5
    sten = solve_network(syn.network.with_radius(radius), bc, MU)
    down = syn.descendants(e)
    assert sten.wss_pa[e] > 1.5 * base.wss_pa[e]
    assert np.all(sten.wss_pa[down] < base.wss_pa[down])
    assert sten.total_inflow_m3_s < base.total_inflow_m3_s


def test_poiseuille_identity_holds_on_every_edge():
    syn = random_tree(7, seed=11, n_loops=6)
    sol = solve_network(syn.network, BoundaryConditions.inlet_outlet_pressure(
        syn.inlet_nodes, 5000.0, syn.outlet_nodes, 1500.0), MU)
    net = syn.network
    np.testing.assert_allclose(
        sol.wss_pa, net.radius_m * np.abs(sol.pressure_drop_pa) / (2 * net.length_m), rtol=1e-10)
