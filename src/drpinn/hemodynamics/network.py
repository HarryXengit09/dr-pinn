"""Vessel network container for the 1D reduced-order hemodynamic model.

A network is an undirected multigraph. Nodes are junctions or endpoints; each
edge is one vessel segment approximated as a straight cylindrical pipe with a
single effective radius. Geometry is stored in SI metres. Converting skeleton
pixels to metres needs a scale calibration (units.pixel_size_um_from_disc),
which is an assumption, not a measurement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph

# Loose plausibility limits, used only to catch unit mistakes such as passing
# pixel or micrometre values where metres are expected.
MAX_PLAUSIBLE_RADIUS_M = 1e-3  # retinal vessels are far below 1 mm in radius
MAX_PLAUSIBLE_LENGTH_M = 5e-2  # longer than the whole retina


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class VesselNetwork:
    """Graph of cylindrical vessel segments.

    Attributes:
        n_nodes: number of nodes, indexed 0..n_nodes-1.
        edges: (E, 2) node indices (u, v). Positive flow means u -> v.
        length_m: (E,) segment lengths in metres.
        radius_m: (E,) effective segment radii in metres.
        node_xy: optional (n_nodes, 2) coordinates, used only for plotting.
    """

    n_nodes: int
    edges: ArrayLike
    length_m: ArrayLike
    radius_m: ArrayLike
    node_xy: ArrayLike | None = None

    def __post_init__(self) -> None:
        n = int(self.n_nodes)
        edges = np.array(self.edges, dtype=np.int64, copy=True)
        length = np.array(self.length_m, dtype=np.float64, copy=True).reshape(-1)
        radius = np.array(self.radius_m, dtype=np.float64, copy=True).reshape(-1)

        if n < 2:
            raise ValueError("a network needs at least 2 nodes")
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
            raise ValueError(f"edges must have shape (E>=1, 2), got {edges.shape}")
        n_edges = edges.shape[0]
        if length.shape != (n_edges,) or radius.shape != (n_edges,):
            raise ValueError("length_m and radius_m must each have one value per edge")
        if edges.min() < 0 or edges.max() >= n:
            raise ValueError(f"edge node index outside [0, {n})")
        loops = np.flatnonzero(edges[:, 0] == edges[:, 1])
        if loops.size:
            raise ValueError(f"self-loop edges are not allowed: {loops[:5].tolist()}")
        if not np.all(np.isfinite(length)) or np.any(length <= 0):
            raise ValueError("all segment lengths must be finite and > 0 m")
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("all segment radii must be finite and > 0 m")
        isolated = np.flatnonzero(np.bincount(edges.ravel(), minlength=n) == 0)
        if isolated.size:
            raise ValueError(f"nodes without any segment: {isolated[:5].tolist()}")
        if radius.max() > MAX_PLAUSIBLE_RADIUS_M or length.max() > MAX_PLAUSIBLE_LENGTH_M:
            warnings.warn(
                "segment radius > 1 mm or length > 5 cm: geometry must be in metres; "
                "pixels or micrometres were probably passed without conversion",
                UserWarning,
                stacklevel=3,
            )

        xy = None
        if self.node_xy is not None:
            xy = np.array(self.node_xy, dtype=np.float64, copy=True)
            if xy.shape != (n, 2):
                raise ValueError(f"node_xy must have shape ({n}, 2), got {xy.shape}")
            xy = _frozen(xy)

        object.__setattr__(self, "n_nodes", n)
        object.__setattr__(self, "edges", _frozen(edges))
        object.__setattr__(self, "length_m", _frozen(length))
        object.__setattr__(self, "radius_m", _frozen(radius))
        object.__setattr__(self, "node_xy", xy)

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def degree(self) -> NDArray[np.int64]:
        return np.bincount(self.edges.ravel(), minlength=self.n_nodes)

    def terminal_nodes(self) -> NDArray[np.int64]:
        """Nodes of degree 1 (vessel endpoints, including image-border cut-offs)."""
        return np.flatnonzero(self.degree() == 1)

    def incidence_matrix(self) -> sparse.csr_matrix:
        """Signed (E, N) incidence matrix: +1 at u, -1 at v for edge (u, v)."""
        e = np.arange(self.n_edges)
        rows = np.concatenate([e, e])
        cols = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        vals = np.concatenate([np.ones(self.n_edges), -np.ones(self.n_edges)])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_edges, self.n_nodes))

    def connected_components(self) -> tuple[int, NDArray[np.int32]]:
        adj = sparse.csr_matrix(
            (np.ones(self.n_edges), (self.edges[:, 0], self.edges[:, 1])),
            shape=(self.n_nodes, self.n_nodes),
        )
        return csgraph.connected_components(adj, directed=False)

    def scaled(self, factor: float) -> "VesselNetwork":
        """Uniformly rescale all lengths and radii (e.g. a different pixel size)."""
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("scale factor must be finite and > 0")
        xy = None if self.node_xy is None else self.node_xy * factor
        return VesselNetwork(self.n_nodes, self.edges, self.length_m * factor,
                             self.radius_m * factor, xy)

    def with_radius(self, radius_m: ArrayLike) -> "VesselNetwork":
        """Same topology and lengths, new radii (e.g. a perturbed sample)."""
        return VesselNetwork(self.n_nodes, self.edges, self.length_m, radius_m, self.node_xy)
