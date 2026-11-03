"""Mask -> vessel graph, checked against synthetic masks with known geometry."""

import math

import numpy as np
import pytest

from drpinn.hemodynamics import BoundaryConditions, solve_network
from drpinn.vessel_graph import geometry_features, tortuosity
from drpinn.vessel_graph.extract import ExtractionConfig, extract_vessel_graph
from drpinn.vessel_graph.synthetic_masks import (
    curved_tube, fundus_like, make_image, straight_tube, x_crossing, y_bifurcation,
)
from drpinn.vessel_graph.validation import compare_centrelines, densify


def _nearest_truth(g, img):
    """For each extracted segment, index of the true path closest to its midpoint."""
    mid = lambda p: (d := densify(p))[len(d) // 2]  # noqa: E731
    mids = np.array([mid(p) for p in img.paths_rc])
    return [int(np.linalg.norm(mids - mid(p), axis=1).argmin()) for p in g.paths_rc]


@pytest.mark.parametrize("angle", [0.0, 26.57, 45.0, 70.0])
def test_straight_tube_length_and_radius(angle):
    img = straight_tube(shape=(240, 260), radius=4.0, angle_deg=angle)
    g = extract_vessel_graph(img.binary, probability=img.soft)
    assert g.n_edges == 1 and sorted(g.node_kind) == ["tip", "tip"]
    assert g.length_px[0] == pytest.approx(img.true_lengths_px[0], rel=0.02)
    assert g.radius_px[0] == pytest.approx(4.0, rel=0.02)
    assert g.radius_reliable[0]


def test_binary_area_radius_is_unbiased_over_subpixel_offsets():
    rng = np.random.default_rng(0)
    est = [extract_vessel_graph(straight_tube(radius=3.0, offset=tuple(rng.uniform(-0.5, 0.5, 2))).binary
                                ).radius_px[0] for _ in range(12)]
    assert np.mean(est) == pytest.approx(3.0, rel=0.03)


def test_curved_tube_length_and_tortuosity():
    img = curved_tube()
    g = extract_vessel_graph(img.binary, probability=img.soft)
    assert g.n_edges == 1
    assert g.length_px[0] == pytest.approx(img.true_lengths_px[0], rel=0.02)
    assert tortuosity(g.paths_rc)[0] == pytest.approx(tortuosity(img.paths_rc)[0], rel=0.02)


def test_y_bifurcation_topology_and_radii():
    img = y_bifurcation()
    g = extract_vessel_graph(img.binary, probability=img.soft)
    assert g.n_edges == 3
    assert sorted(g.degree().tolist()) == [1, 1, 1, 3]
    truth = _nearest_truth(g, img)
    assert sorted(truth) == [0, 1, 2]
    np.testing.assert_allclose(g.radius_px, img.radii_px[truth], rtol=0.03)


@pytest.mark.parametrize("angle", [30.0, 40.0, 70.0, 90.0])
def test_crossing_is_split_into_two_unconnected_vessels(angle):
    img = x_crossing(angle_deg=angle)
    g = extract_vessel_graph(img.binary, probability=img.soft)
    assert g.n_edges == 2 and g.stats["crossings_resolved"] == 1
    assert np.all(g.degree() == 1) and np.all(g.through_crossing)
    np.testing.assert_allclose(sorted(g.radius_px), sorted(img.radii_px), rtol=0.05)
    np.testing.assert_allclose(sorted(g.length_px), sorted(img.true_lengths_px), rtol=0.05)


@pytest.mark.parametrize("angles,radius,is_crossing", [
    ((0, 45, 90, 135), 3.0, False),   # skeleton: two 3-way junctions + bridge
    ((0, 60, 120, 240), 2.0, False),  # skeleton: one 4-way junction
    ((0, 90, 150, 240), 3.0, False),  # pairs only 150 deg apart: not straight enough
    ((0, 100, 180, 260), 2.0, True),  # 180 and 160 deg pairs: a crossing
])
def test_four_arm_junctions_are_split_only_when_straight_through(angles, radius, is_crossing):
    c = np.array([120.0, 120.0])
    arms = [np.stack([c, c + 100 * np.array([-math.sin(math.radians(a)), math.cos(math.radians(a))])])
            for a in angles]
    g = extract_vessel_graph(make_image((240, 240), arms, [radius] * 4, "star").binary)
    assert g.stats["crossings_resolved"] == int(is_crossing)
    assert (g.degree() == 1).sum() == 4
    if not is_crossing:
        assert g.stats["possible_crossings"] >= 1


def test_pixel_graph_removes_redundant_diagonal_links():
    from drpinn.vessel_graph.extract import _pixel_graph
    skel = np.zeros((6, 6), bool)
    skel[2, 1], skel[2, 2], skel[2, 3], skel[3, 3], skel[4, 3] = True, True, True, True, True  # L corner
    _, _, _, adj = _pixel_graph(skel)
    deg = np.diff(adj.indptr)
    assert sorted(deg.tolist()) == [1, 1, 2, 2, 2]  # a simple path, not a triangle at the corner


def test_crossing_resolution_can_be_switched_off():
    g = extract_vessel_graph(x_crossing().binary, config=ExtractionConfig(resolve_crossings=False))
    assert g.stats["crossings_resolved"] == 0 and g.degree().max() >= 3
    assert not g.through_crossing.any()


def test_spur_is_pruned_but_real_branch_is_kept():
    shape = (120, 240)
    main = np.array([[60.0, 20.0], [60.0, 220.0]])
    stub = np.array([[60.0, 120.0], [53.0, 120.0]])     # tiny bump: 7 px from the axis of an r=4 tube
    branch = np.array([[60.0, 120.0], [15.0, 150.0]])   # genuine side branch
    g_stub = extract_vessel_graph(make_image(shape, [main, stub], [4.0, 2.0], "stub").binary)
    g_branch = extract_vessel_graph(make_image(shape, [main, branch], [4.0, 2.5], "branch").binary)
    assert g_stub.n_edges == 1
    assert g_branch.n_edges == 3 and g_branch.degree().max() == 3


def test_cleaning_removes_specks_and_fills_holes():
    img = straight_tube(radius=4.0)
    m = img.binary.copy()
    m[5:7, 5:7] = True        # speck
    m[60, 100] = False        # 1-pixel hole inside the vessel
    g = extract_vessel_graph(m)
    assert g.n_edges == 1
    assert g.stats["components_removed"] == 1 and g.stats["holes_filled"] == 1


def test_vessels_cut_by_the_image_or_fov_border_are_flagged():
    edge = make_image((100, 200), [np.array([[50.0, -10.0], [50.0, 150.0]])], [3.0], "runs off left edge")
    g = extract_vessel_graph(edge.binary)
    tips = np.flatnonzero(g.degree() == 1)
    assert g.node_truncated[tips].tolist().count(True) == 1
    # circular field of view smaller than the vessel
    yy, xx = np.mgrid[:200, :200]
    fov = (yy - 100) ** 2 + (xx - 100) ** 2 <= 80**2
    long = make_image((200, 200), [np.array([[100.0, 10.0], [100.0, 120.0]])], [3.0], "crosses FOV edge")
    g2 = extract_vessel_graph(long.binary, fov=fov)
    assert g2.node_truncated.sum() == 1


def test_ring_vessel_has_no_self_loops():
    t = np.linspace(0, 2 * math.pi, 400)
    ring = np.column_stack([100 + 60 * np.sin(t), 100 + 60 * np.cos(t)])
    g = extract_vessel_graph(make_image((200, 200), [ring], [3.0], "ring").binary)
    assert g.n_edges == 2 and np.all(g.edges[:, 0] != g.edges[:, 1])
    assert g.to_network(10.0).n_edges == 2


@pytest.mark.parametrize("seed", range(4))
def test_fundus_like_images_recover_centrelines_and_radii(seed):
    img = fundus_like(seed=seed)
    g = extract_vessel_graph(img.binary, probability=img.soft)
    c = compare_centrelines(g.paths_rc, g.radius_px, img.paths_rc, img.radii_px, mask=g.radius_reliable)
    assert c.recall > 0.95 and c.precision > 0.95
    assert c.radius_median_abs_error < 0.02
    assert c.matched >= 0.8 * (c.matched + c.unmatched)
    assert g.stats["components_removed"] > 0  # speckles were present and removed


def test_to_network_units_and_end_to_end_physics():
    img = fundus_like(seed=1)
    g = extract_vessel_graph(img.binary, probability=img.soft)
    px_um = 12.0  # ASSUMED pixel size for this software test
    net = g.to_network(px_um)
    np.testing.assert_allclose(net.radius_m, g.radius_px * px_um * 1e-6)
    np.testing.assert_allclose(net.node_xy[:, 0], g.node_rc[:, 1] * px_um * 1e-6)  # x = column
    # one connected tree: inlet = widest tip, outlets = other tips (software integration only)
    comp_n, comp = net.connected_components()
    big = np.bincount(comp).argmax()
    tips = [int(t) for t in net.terminal_nodes() if comp[t] == big]
    seg_r = {t: net.radius_m[np.flatnonzero((net.edges == t).any(axis=1))[0]] for t in tips}
    inlet = max(tips, key=seg_r.get)
    bc = {inlet: 6000.0, **{t: 2000.0 for t in tips if t != inlet}}
    for k in range(comp_n):  # anchor any other pieces so the system is solvable
        if k != big:
            bc.setdefault(int(np.flatnonzero(comp == k)[0]), 2000.0)
    sol = solve_network(net, BoundaryConditions(pressure_pa=bc), 3.5e-3)
    assert sol.max_rel_conservation_error < 1e-10
    feats = geometry_features(net, [inlet] + [int(np.flatnonzero(comp == k)[0]) for k in range(comp_n) if k != big],
                              edge_paths=g.paths_xy_m(px_um))
    assert np.all(feats["tortuosity"] >= 1.0 - 1e-9)


@pytest.mark.parametrize("bad", [np.zeros(10), np.zeros((2, 50)), np.zeros((50, 50))])
def test_invalid_or_empty_masks_rejected(bad):
    with pytest.raises(ValueError):
        extract_vessel_graph(bad)


def test_invalid_auxiliary_inputs_rejected():
    img = straight_tube()
    with pytest.raises(ValueError):
        extract_vessel_graph(img.binary, probability=img.soft * 2)
    with pytest.raises(ValueError):
        extract_vessel_graph(img.binary, fov=np.ones((5, 5)))
    with pytest.raises(ValueError):
        extract_vessel_graph(img.binary).to_network(0.0)
