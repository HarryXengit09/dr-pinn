import math

import numpy as np
import pytest

from drpinn.hemodynamics import (
    BoundaryConditions,
    VesselNetwork,
    murray_level_resistance,
    murray_terminal_resistance,
    murray_tree,
    random_tree,
    solve_network,
)
from drpinn.vessel_graph import FEATURE_NAMES, geometry_features, tortuosity

MU = 3.5e-3


# --- Murray-scaled terminal resistance --------------------------------------------

def test_murray_level_resistance_formula():
    r, lam = 20e-6, 20.0
    assert murray_level_resistance(r, MU, lam) == pytest.approx(8 * MU * lam / (math.pi * r**3))


def test_terminal_resistance_reproduces_the_truncated_murray_tree_exactly():
    """k unseen generations as a Robin BC must give the same flows as drawing them."""
    lam, p_in, p_out, k = 20.0, 6000.0, 2000.0, 3
    full = murray_tree(8, 60e-6, length_to_radius=lam)
    cut = murray_tree(8 - k, 60e-6, length_to_radius=lam)
    sol_full = solve_network(full.network, BoundaryConditions.inlet_outlet_pressure(
        full.inlet_nodes, p_in, full.outlet_nodes, p_out), MU)
    robin = murray_terminal_resistance(cut.network, cut.outlet_nodes, MU, k, lam, p_out)
    sol_cut = solve_network(cut.network, BoundaryConditions(
        pressure_pa={0: p_in}, terminal_resistance=robin), MU)
    m = cut.network.n_edges  # same BFS numbering for the shared generations
    np.testing.assert_allclose(sol_cut.flow_m3_s, sol_full.flow_m3_s[:m], rtol=1e-10)
    np.testing.assert_allclose(sol_cut.pressure_pa, sol_full.pressure_pa[:cut.network.n_nodes], rtol=1e-10)


def test_terminal_resistance_requires_endpoints():
    syn = murray_tree(4, 60e-6)
    with pytest.raises(ValueError, match="degree"):
        murray_terminal_resistance(syn.network, [1], MU, 2, 20.0, 2000.0)
    with pytest.raises(ValueError):
        murray_terminal_resistance(syn.network, syn.outlet_nodes, MU, 0.0, 20.0, 2000.0)


# --- geometry-only features ---------------------------------------------------------

def test_murray_tree_features():
    syn = murray_tree(6, 60e-6)
    f = geometry_features(syn.network, syn.inlet_nodes)
    assert set(f) == set(FEATURE_NAMES)
    gen = syn.edge_generation
    np.testing.assert_array_equal(f["branch_order"], gen)
    root, rest = gen == 0, gen > 0
    assert np.all(np.isnan(f["murray_deviation"][root]))
    np.testing.assert_allclose(f["murray_deviation"][rest], 0.0, atol=1e-12)
    np.testing.assert_allclose(f["sibling_radius_ratio"][rest], 1.0)
    np.testing.assert_allclose(f["parent_radius_ratio"][rest], 2 ** (-1 / 3))
    np.testing.assert_array_equal(f["is_terminal"], (gen == gen.max()).astype(float))
    assert np.all(np.isnan(f["tortuosity"]))  # unknown without centrelines, never a silent 1.0
    lengths = syn.network.length_m
    path = np.zeros(syn.network.n_edges)  # sum of ancestor lengths, walked independently
    for e in range(syn.network.n_edges):
        p = syn.parent_edge[e]
        while p >= 0:
            path[e] += lengths[p]
            p = syn.parent_edge[p]
    np.testing.assert_allclose(f["path_length_from_inlet_um"], path * 1e6, rtol=1e-12)


def test_branch_angle_and_murray_violation_on_hand_built_y():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    xy = [[0, 0], [1e-3, 0], [1e-3 + 1e-3 * c, 1e-3 * s], [1e-3 + 1e-3 * c, -1e-3 * s]]
    net = VesselNetwork(4, [[0, 1], [1, 2], [1, 3]], [1e-3] * 3, [50e-6, 40e-6, 30e-6], xy)
    f = geometry_features(net, [0])
    np.testing.assert_allclose(f["branch_angle_deg"][1:], 30.0, atol=1e-9)
    expected_dev = (50**3 - 40**3 - 30**3) / 50**3
    np.testing.assert_allclose(f["murray_deviation"][1:], expected_dev, rtol=1e-12)
    np.testing.assert_allclose(f["sibling_radius_ratio"][1:], 30 / 40)


def test_parent_is_the_widest_feeder_at_a_loop_junction():
    # 0->1 (r50), 0->2 (r30), 1->3 (r40), 2->3 (r20), 3->4 (r35): node 3 has two feeders
    net = VesselNetwork(5, [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]], [1e-3] * 5,
                        np.array([50, 30, 40, 20, 35]) * 1e-6)
    f = geometry_features(net, [0])
    assert f["parent_radius_ratio"][4] == pytest.approx(35 / 40)


def test_tortuosity_of_a_semicircle():
    t = np.linspace(0, math.pi, 2001)
    semicircle = np.column_stack([np.cos(t), np.sin(t)])
    assert tortuosity([semicircle, [[0, 0], [3, 4]]]) == pytest.approx([math.pi / 2, 1.0], rel=1e-6)


def test_features_reject_unreachable_nodes_and_bad_paths():
    net = VesselNetwork(4, [[0, 1], [2, 3]], [1e-3, 1e-3], [5e-5, 5e-5])
    with pytest.raises(ValueError, match="not connected"):
        geometry_features(net, [0])
    syn = random_tree(4, seed=0)
    with pytest.raises(ValueError):
        geometry_features(syn.network, syn.inlet_nodes, edge_paths=[[[0, 0], [1, 1]]])
