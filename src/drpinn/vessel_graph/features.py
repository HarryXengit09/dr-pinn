"""Geometry-only segment features: the "no physics" baseline for RQ1 (PDF §11.3).

These use only what the image shows (radius, length, shape, branching). They
deliberately contain no flow, pressure or resistance-sum quantities, so any
advantage of the physics model over them is attributable to the network-level
flow distribution.

Orientation: the network is oriented by hop distance from the inlet node(s);
for each segment the endpoint closer to an inlet is its upstream node.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph

from ..hemodynamics.network import VesselNetwork

FEATURE_NAMES = (
    "radius_um", "length_um", "length_to_radius", "tortuosity", "branch_order",
    "path_length_from_inlet_um", "parent_radius_ratio", "murray_deviation",
    "sibling_radius_ratio", "branch_angle_deg", "is_terminal",
)


def _hop_distance(network: VesselNetwork, inlet_nodes: Sequence[int]) -> NDArray[np.float64]:
    n = network.n_nodes
    adj = sparse.csr_matrix((np.ones(network.n_edges), (network.edges[:, 0], network.edges[:, 1])),
                            shape=(n, n))
    d = csgraph.shortest_path(adj, directed=False, unweighted=True, indices=list(inlet_nodes))
    return np.atleast_2d(d).min(axis=0)


def _weighted_distance(network: VesselNetwork, inlet_nodes: Sequence[int]) -> NDArray[np.float64]:
    n = network.n_nodes
    adj = sparse.csr_matrix((network.length_m, (network.edges[:, 0], network.edges[:, 1])),
                            shape=(n, n))
    d = csgraph.dijkstra(adj, directed=False, indices=list(inlet_nodes))
    return np.atleast_2d(d).min(axis=0)


def tortuosity(paths: Sequence[ArrayLike]) -> NDArray[np.float64]:
    """Arc length / chord length per polyline (1.0 = straight)."""
    out = np.empty(len(paths))
    for i, p in enumerate(paths):
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 2:
            raise ValueError(f"path {i} must have shape (k>=2, 2)")
        arc = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(p[-1] - p[0]))
        out[i] = arc / chord if chord > 0 else math.inf
    return out


def geometry_features(network: VesselNetwork, inlet_nodes: Iterable[int],
                      edge_paths: Sequence[ArrayLike] | None = None) -> dict[str, NDArray[np.float64]]:
    """Per-segment geometry features (one array per name in FEATURE_NAMES).

    Args:
        network: geometry in metres; node_xy (same units as edge_paths) enables angles.
        inlet_nodes: where flow enters (e.g. arterial trunk at the optic disc).
        edge_paths: optional centreline polyline per segment, for tortuosity.
            Without it tortuosity is NaN (unknown), never a silent 1.0.

    Parent-dependent features are NaN for segments that start at an inlet.
    """
    inlets = [int(i) for i in inlet_nodes]
    if not inlets:
        raise ValueError("at least one inlet node is required")
    e_count, edges, r = network.n_edges, network.edges, network.radius_m
    hop = _hop_distance(network, inlets)
    if not np.all(np.isfinite(hop)):
        raise ValueError("some nodes are not connected to any inlet")
    dist = _weighted_distance(network, inlets)

    # upstream node = endpoint nearer an inlet (ties broken by path length, then u)
    key = np.stack([hop, dist], axis=1)
    u_first = np.array([tuple(key[a]) <= tuple(key[b]) for a, b in edges])
    up = np.where(u_first, edges[:, 0], edges[:, 1])
    down = np.where(u_first, edges[:, 1], edges[:, 0])

    incident: dict[int, list[int]] = {}
    for e, (a, b) in enumerate(edges):
        incident.setdefault(int(a), []).append(e)
        incident.setdefault(int(b), []).append(e)

    parent = np.full(e_count, -1)
    for e in range(e_count):
        j = int(up[e])
        feeders = [f for f in incident[j] if f != e and down[f] == j]
        if feeders:
            parent[e] = max(feeders, key=lambda f: r[f])  # main feeder = widest

    xy = network.node_xy
    feats = {name: np.full(e_count, np.nan) for name in FEATURE_NAMES}
    feats["radius_um"] = r * 1e6
    feats["length_um"] = network.length_m * 1e6
    feats["length_to_radius"] = network.length_m / r
    if edge_paths is not None:
        if len(edge_paths) != e_count:
            raise ValueError("edge_paths needs one polyline per segment")
        feats["tortuosity"] = tortuosity(edge_paths)
    feats["branch_order"] = hop[up].astype(float)
    feats["path_length_from_inlet_um"] = dist[up] * 1e6
    deg = network.degree()
    feats["is_terminal"] = (deg[down] == 1).astype(float)

    for e in range(e_count):
        p = parent[e]
        if p < 0:
            continue
        j = int(up[e])
        children = [f for f in incident[j] if f != p and up[f] == j]
        rc = r[children]
        feats["parent_radius_ratio"][e] = r[e] / r[p]
        feats["murray_deviation"][e] = (r[p] ** 3 - np.sum(rc**3)) / r[p] ** 3
        feats["sibling_radius_ratio"][e] = rc.min() / rc.max()
        if xy is not None:
            v_in = xy[j] - xy[up[p]]
            v_out = xy[down[e]] - xy[j]
            denom = np.linalg.norm(v_in) * np.linalg.norm(v_out)
            if denom > 0:
                cos = np.clip(np.dot(v_in, v_out) / denom, -1.0, 1.0)
                feats["branch_angle_deg"][e] = math.degrees(math.acos(cos))
    return feats
