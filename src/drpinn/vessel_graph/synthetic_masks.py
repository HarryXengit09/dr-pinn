"""Render vessel geometry of KNOWN width and length into masks.

SYNTHETIC TEST DATA ONLY. Used to measure how well graph extraction recovers
radius, length, topology and crossings, because real masks have no ground
truth for these quantities.

Coordinates are (row, col) in pixels. Each tube is the set of points within
`radius` of its centreline polyline. `soft` is the fractional pixel coverage
(supersampled), mimicking a probability map; `binary` is coverage >= 0.5,
mimicking a hand-drawn or thresholded mask.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True, eq=False)
class SyntheticVesselImage:
    soft: NDArray[np.float64]
    binary: NDArray[np.bool_]
    paths_rc: tuple[NDArray[np.float64], ...]  # true centrelines
    radii_px: NDArray[np.float64]
    description: str
    synthetic: bool = True
    tree_ids: NDArray[np.int64] | None = None  # per path: 0 = "artery" tree, 1 = "vein" tree

    def tree_soft(self, tree: int, supersample: int = 4) -> NDArray[np.float64]:
        """Fractional coverage of one tree only (e.g. a synthetic artery or vein map)."""
        if self.tree_ids is None:
            raise ValueError("this image has no tree labels")
        sel = [k for k, t in enumerate(self.tree_ids) if t == tree]
        return render_tubes(self.soft.shape, [self.paths_rc[k] for k in sel], self.radii_px[sel], supersample)

    @property
    def true_lengths_px(self) -> NDArray[np.float64]:
        return np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in self.paths_rc])


def _point_polyline_distance(points: NDArray, path: NDArray, spacing: float = 0.1,
                             upper: float = np.inf) -> NDArray:
    """Distance to a polyline via its densely resampled points (error < spacing^2 / 8r)."""
    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        dense.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return cKDTree(np.vstack(dense)).query(points, distance_upper_bound=upper)[0]


def render_tubes(shape: tuple[int, int], paths: Sequence[ArrayLike], radii: Sequence[float],
                 supersample: int = 4) -> NDArray[np.float64]:
    """Fractional coverage of the union of tubes, by supersampling each pixel."""
    h, w = shape
    ss = int(supersample)
    canvas = np.zeros((h * ss, w * ss), dtype=bool)
    for path, r in zip(paths, radii):
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2 or r <= 0:
            raise ValueError("each path must be (k>=2, 2) and each radius > 0")
        r0 = max(0, int(math.floor(path[:, 0].min() - r - 1)))
        r1 = min(h, int(math.ceil(path[:, 0].max() + r + 2)))
        c0 = max(0, int(math.floor(path[:, 1].min() - r - 1)))
        c1 = min(w, int(math.ceil(path[:, 1].max() + r + 2)))
        if r0 >= r1 or c0 >= c1:
            continue
        rows = r0 + (np.arange((r1 - r0) * ss) + 0.5) / ss - 0.5
        cols = c0 + (np.arange((c1 - c0) * ss) + 0.5) / ss - 0.5
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        d = _point_polyline_distance(np.column_stack([rr.ravel(), cc.ravel()]), path, upper=r + 1.0)
        canvas[r0 * ss:r1 * ss, c0 * ss:c1 * ss] |= (d <= r).reshape(rr.shape)
    return canvas.reshape(h, ss, w, ss).mean(axis=(1, 3))


def bezier(p0: ArrayLike, p1: ArrayLike, bend: float, spacing: float = 1.0) -> NDArray[np.float64]:
    """Quadratic Bezier from p0 to p1; control point offset `bend` x chord length sideways."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    chord = p1 - p0
    normal = np.array([-chord[1], chord[0]])
    ctrl = (p0 + p1) / 2 + bend * normal
    n = max(2, int(np.linalg.norm(chord) * (1 + 2 * abs(bend)) / spacing) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1


def make_image(shape: tuple[int, int], paths: Sequence[ArrayLike], radii: Sequence[float],
               description: str, supersample: int = 4) -> SyntheticVesselImage:
    soft = render_tubes(shape, paths, radii, supersample)
    return SyntheticVesselImage(soft, soft >= 0.5, tuple(np.asarray(p, float) for p in paths),
                                np.asarray(radii, float), description)


def straight_tube(shape=(120, 240), length=180.0, radius=4.0, angle_deg=0.0,
                  offset=(0.0, 0.0)) -> SyntheticVesselImage:
    h, w = shape
    centre = np.array([h / 2 + offset[0], w / 2 + offset[1]])
    d = np.array([-math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))])
    p = np.stack([centre - d * length / 2, centre + d * length / 2])
    return make_image(shape, [p], [radius], f"straight tube r={radius}, angle={angle_deg}")


def y_bifurcation(shape=(200, 260), r_parent=5.0, r_children=(4.0, 3.5), half_angle_deg=35.0,
                  lengths=(90.0, 110.0, 110.0)) -> SyntheticVesselImage:
    j = np.array([shape[0] / 2, 30.0 + lengths[0]])
    p0 = np.stack([j - [0, lengths[0]], j])
    paths, radii = [p0], [r_parent]
    for sign, rc, length in zip((-1, 1), r_children, lengths[1:]):
        a = math.radians(sign * half_angle_deg)
        paths.append(np.stack([j, j + length * np.array([math.sin(a), math.cos(a)])]))
        radii.append(rc)
    return make_image(shape, paths, radii, "Y bifurcation")


def x_crossing(shape=(240, 240), r=(4.0, 3.0), angle_deg=70.0, length=200.0) -> SyntheticVesselImage:
    c = np.array([shape[0] / 2, shape[1] / 2])
    paths = []
    for ang in (0.0, angle_deg):
        d = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
        paths.append(np.stack([c - d * length / 2, c + d * length / 2]))
    return make_image(shape, paths, list(r), f"X crossing at {angle_deg} deg")


def curved_tube(shape=(160, 260), radius=4.0, bend=0.25) -> SyntheticVesselImage:
    p = bezier([shape[0] * 0.65, 25.0], [shape[0] * 0.65, shape[1] - 25.0], bend, spacing=0.5)
    return make_image(shape, [p], [radius], f"curved tube bend={bend}")


def fundus_like(shape=(400, 400), seed: int = 0, generations: int = 4,
                root_radius: float = 5.5, segment_length: float = 55.0,
                speckle: int = 12, disc_rc: tuple[float, float] | None = None,
                trunk_angles_deg: tuple[float, float] = (-25.0, 30.0)) -> SyntheticVesselImage:
    """Two Murray-like trees (an 'artery' and a 'vein') leaving a disc, crossing each other.

    Gently curved segments, radii from Murray's law with jitter, plus isolated
    speckles that cleaning should remove. Ground-truth paths list every segment.
    """
    rng = np.random.default_rng(seed)
    disc = np.array(disc_rc if disc_rc is not None else (shape[0] / 2, shape[1] * 0.2), float)
    paths, radii = [], []

    def grow(start, angle, r, length, gen):
        if gen >= generations:
            return
        end = start + length * np.array([math.sin(angle), math.cos(angle)])
        end = np.clip(end, 8, np.array(shape) - 9)
        paths.append(bezier(start, end, rng.uniform(-0.12, 0.12), spacing=1.0))
        radii.append(r)
        r_child = r * 2 ** (-1 / 3) * rng.uniform(0.9, 1.1, 2)
        for k, sign in enumerate((-1, 1)):
            grow(end, angle + sign * rng.uniform(0.35, 0.6), max(r_child[k], 1.6),
                 length * rng.uniform(0.75, 0.95), gen + 1)

    grow(disc - [12, 0], math.radians(trunk_angles_deg[0]), root_radius, segment_length, 0)  # "artery"
    n_artery = len(paths)
    grow(disc + [12, 0], math.radians(trunk_angles_deg[1]), root_radius * 1.2, segment_length, 0)  # "vein"
    img = make_image(shape, paths, radii, f"fundus-like two-tree mask, seed {seed}")
    binary = img.binary.copy()
    for _ in range(speckle):  # tiny noise blobs far from vessels are removed by cleaning
        r, c = rng.integers(5, shape[0] - 5), rng.integers(5, shape[1] - 5)
        binary[r:r + 2, c:c + 2] |= img.soft[r - 4:r + 6, c - 4:c + 6].max() == 0
    tree_ids = np.array([0] * n_artery + [1] * (len(paths) - n_artery), dtype=np.int64)
    return SyntheticVesselImage(img.soft, binary, img.paths_rc, img.radii_px, img.description,
                                tree_ids=tree_ids)
