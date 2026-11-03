"""Synthetic vessel networks for software and physics verification.

SYNTHETIC DATA ONLY. These networks test code paths and known physics
(analytical solutions, conservation, invariances). They carry no clinical
meaning and must never be reported as patient results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .network import VesselNetwork


@dataclass(frozen=True, eq=False)
class SyntheticNetwork:
    network: VesselNetwork
    inlet_nodes: NDArray[np.int64]
    outlet_nodes: NDArray[np.int64]
    edge_generation: NDArray[np.int64]
    parent_edge: NDArray[np.int64]  # -1 for root segments and added loop segments
    description: str
    synthetic: bool = True

    def descendants(self, edge: int) -> NDArray[np.int64]:
        """Edges strictly downstream of `edge` in the tree (loop segments excluded)."""
        children: dict[int, list[int]] = {}
        for e, par in enumerate(self.parent_edge):
            if par >= 0:
                children.setdefault(int(par), []).append(e)
        out, stack = [], list(children.get(edge, []))
        while stack:
            e = stack.pop()
            out.append(e)
            stack.extend(children.get(e, []))
        return np.array(sorted(out), dtype=np.int64)


def single_pipe(length_m: float, radius_m: float) -> SyntheticNetwork:
    net = VesselNetwork(2, [[0, 1]], [length_m], [radius_m], [[0.0, 0.0], [length_m, 0.0]])
    return SyntheticNetwork(net, np.array([0]), np.array([1]), np.array([0]),
                            np.array([-1]), "single straight pipe")


def series_chain(lengths_m: ArrayLike, radii_m: ArrayLike) -> SyntheticNetwork:
    lengths = np.asarray(lengths_m, dtype=float)
    k = lengths.size
    edges = np.column_stack([np.arange(k), np.arange(1, k + 1)])
    xy = np.column_stack([np.concatenate([[0.0], np.cumsum(lengths)]), np.zeros(k + 1)])
    net = VesselNetwork(k + 1, edges, lengths, radii_m, xy)
    return SyntheticNetwork(net, np.array([0]), np.array([k]), np.zeros(k, dtype=np.int64),
                            np.arange(-1, k - 1), f"{k} pipes in series")


def _binary_tree(generations: int):
    if generations < 1:
        raise ValueError("generations must be >= 1")
    edges, gen, parent = [(0, 1)], [0], [-1]
    frontier, next_node = [(1, 0)], 2  # (distal node, edge index)
    for g in range(1, generations):
        new = []
        for node, e in frontier:
            for _ in range(2):
                edges.append((node, next_node))
                gen.append(g)
                parent.append(e)
                new.append((next_node, len(edges) - 1))
                next_node += 1
        frontier = new
    leaves = np.array([node for node, _ in frontier], dtype=np.int64)
    return (np.array(edges, dtype=np.int64), np.array(gen, dtype=np.int64),
            np.array(parent, dtype=np.int64), next_node, leaves)


def _layout(edges: NDArray, gen: NDArray, parent: NDArray, lengths: NDArray,
            n_nodes: int) -> NDArray:
    """Planar coordinates (metres) for plotting; children fan out +/- an angle."""
    xy = np.zeros((n_nodes, 2))
    angle = np.zeros(len(edges))
    sibling_seen: dict[int, int] = {}
    for e, (u, v) in enumerate(edges):
        if parent[e] >= 0:
            k = sibling_seen.get(int(parent[e]), 0)
            sibling_seen[int(parent[e])] = k + 1
            angle[e] = angle[parent[e]] + (0.6 if k % 2 == 0 else -0.6) / (1 + 0.3 * gen[e])
        xy[v] = xy[u] + lengths[e] * np.array([math.cos(angle[e]), math.sin(angle[e])])
    return xy


def murray_tree(generations: int, root_radius_m: float, length_to_radius: float = 20.0,
                murray_exponent: float = 3.0) -> SyntheticNetwork:
    """Symmetric binary tree obeying Murray's law r_parent^n = 2 r_child^n.

    With n = 3 and equal outlet pressures, Poiseuille flow halves at each split
    and r^3 halves too, so wall shear stress is identical in every segment.
    That closed-form property is used as a solver test.
    """
    edges, gen, parent, n_nodes, leaves = _binary_tree(generations)
    radius = root_radius_m * 2.0 ** (-gen / murray_exponent)
    length = length_to_radius * radius
    xy = _layout(edges, gen, parent, length, n_nodes)
    net = VesselNetwork(n_nodes, edges, length, radius, xy)
    return SyntheticNetwork(net, np.array([0]), leaves, gen, parent,
                            f"symmetric Murray tree, {generations} generations")


def random_tree(generations: int, seed: int, root_radius_m: float = 60e-6,
                length_to_radius: float = 20.0, radius_jitter: float = 0.15,
                length_jitter: float = 0.3, n_loops: int = 0,
                loop_radius_fraction: float = 0.5) -> SyntheticNetwork:
    """Murray tree with log-normal jitter on radii and lengths, plus optional loops.

    Loops add cycles (as false crossings do in real skeleton graphs) so the
    solver is exercised beyond trees. Reproducible through `seed`.
    """
    rng = np.random.default_rng(seed)
    edges, gen, parent, n_nodes, leaves = _binary_tree(generations)
    radius = root_radius_m * 2.0 ** (-gen / 3.0) * rng.lognormal(0.0, radius_jitter, gen.size)
    length = length_to_radius * root_radius_m * 2.0 ** (-gen / 3.0)
    length = length * rng.lognormal(0.0, length_jitter, gen.size)

    existing = {tuple(sorted(map(int, e))) for e in edges}
    extra_e, extra_r, extra_l = [], [], []
    r_min = float(radius.min())
    attempts = 0
    while len(extra_e) < n_loops and attempts < 100 * max(n_loops, 1):
        attempts += 1
        u, v = (int(x) for x in rng.choice(np.arange(1, n_nodes), size=2, replace=False))
        key = (min(u, v), max(u, v))
        if key in existing:
            continue
        existing.add(key)
        extra_e.append(key)
        extra_r.append(loop_radius_fraction * r_min * rng.uniform(0.8, 1.2))
        extra_l.append(length_to_radius * r_min * rng.uniform(1.0, 3.0))
    if len(extra_e) < n_loops:
        raise RuntimeError("could not place the requested number of loop segments")

    xy = _layout(edges, gen, parent, length, n_nodes)
    if extra_e:
        edges = np.vstack([edges, np.array(extra_e, dtype=np.int64)])
        radius = np.concatenate([radius, extra_r])
        length = np.concatenate([length, extra_l])
        gen = np.concatenate([gen, np.full(len(extra_e), -1)])
        parent = np.concatenate([parent, np.full(len(extra_e), -1)])
    net = VesselNetwork(n_nodes, edges, length, radius, xy)
    return SyntheticNetwork(net, np.array([0]), leaves, gen, parent,
                            f"random tree, {generations} generations, {n_loops} loops, seed {seed}")
