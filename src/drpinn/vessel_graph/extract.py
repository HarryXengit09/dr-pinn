"""Vessel mask -> 1D vessel graph (Milestone 3).

Pipeline
--------
1. Clean: keep the FOV, drop small specks, fill tiny holes.
2. Skeletonize to a 1-pixel centreline; build an 8-connected pixel graph with
   redundant diagonal links removed (so every centreline pixel has degree 2).
3. Nodes: vessel tips (degree 1) and junction clusters (degree >= 3).
4. Trace each centreline run between nodes into a segment.
5. Prune skeleton spurs (short tips hanging off a junction), merge degree-2 nodes.
6. Crossings: a degree-4 node, or two degree-3 nodes joined by a short bridge,
   whose branches pair up into two nearly straight paths is treated as one
   vessel passing over another and split into two unconnected vessels.
   Heuristic: a thin vessel crossing a thick one is indistinguishable in a
   binary mask from two opposite side branches. Flags are reported.
7. Width (sub-pixel): every vessel pixel is assigned to its nearest centreline
   pixel; segment width = assigned area / centreline length, using only the
   segment core (away from junctions, tips and crossings). With a soft
   probability map, area is the summed probability. Falls back to the
   distance transform (flagged unreliable) for segments with no core.
8. Flags: tips within a margin of the image/FOV border are 'truncated'
   (cut off by the photo, not real vessel ends).

All geometry stays in pixels until `VesselGraph.to_network(pixel_size_um)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from skimage.morphology import skeletonize

from ..hemodynamics.network import VesselNetwork

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True)
class ExtractionConfig:
    min_component_area_px: int = 30
    max_hole_area_px: int = 10
    spur_length_factor: float = 2.0     # prune tips shorter than factor * junction radius + spur_min_px
    spur_min_px: float = 3.0
    trim_factor: float = 1.5            # exclude width pixels within factor * radius of a node/crossing
    border_margin_px: float = 3.0
    resolve_crossings: bool = True
    crossing_bridge_factor: float = 2.5  # bridge shorter than factor * (r_a + r_b) -> crossing candidate
    crossing_min_angle_deg: float = 155.0  # both through-paths must be at least this straight
    direction_reach_factor: float = 4.0    # branch direction measured this many node radii out (min 8 px)
    smoothing_window: int = 5           # centreline smoothing for lengths (odd, >= 1)


@dataclass(frozen=True, eq=False)
class VesselGraph:
    """Extracted vessel graph in pixel units. Edge i runs edges[i,0] -> edges[i,1]."""

    image_shape: tuple[int, int]
    node_rc: NDArray[np.float64]
    node_kind: tuple[str, ...]           # 'tip', 'junction', 'split'
    node_truncated: NDArray[np.bool_]
    node_possible_crossing: NDArray[np.bool_]
    edges: NDArray[np.int64]
    paths_rc: tuple[NDArray[np.float64], ...]  # smoothed centrelines incl. end nodes
    length_px: NDArray[np.float64]
    radius_px: NDArray[np.float64]       # sub-pixel area-based estimate
    radius_edt_px: NDArray[np.float64]   # distance-transform estimate (for comparison)
    radius_reliable: NDArray[np.bool_]
    through_crossing: NDArray[np.bool_]
    stats: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_rc.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def degree(self) -> NDArray[np.int64]:
        return np.bincount(self.edges.ravel(), minlength=self.n_nodes)

    def to_network(self, pixel_size_um: float) -> VesselNetwork:
        """Convert to SI metres. pixel_size_um is an ASSUMPTION unless calibrated
        (e.g. optic-disc reference); WSS rankings are scale-free only for the
        Newtonian model (M1), not with diameter-dependent viscosity (M2)."""
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be finite and > 0")
        s = pixel_size_um * 1e-6
        xy = self.node_rc[:, ::-1] * s  # x = column, y = row
        return VesselNetwork(self.n_nodes, self.edges, self.length_px * s, self.radius_px * s, xy)

    def paths_xy_m(self, pixel_size_um: float) -> list[NDArray[np.float64]]:
        s = pixel_size_um * 1e-6
        return [p[:, ::-1] * s for p in self.paths_rc]


# ----------------------------------------------------------------------------- helpers

def _smooth(pts: NDArray, window: int) -> NDArray:
    """Moving-average smoothing that keeps both end points fixed."""
    if window <= 1 or len(pts) <= 2:
        return pts.copy()
    k = window // 2
    out = pts.copy()
    n = len(pts)
    for i in range(1, n - 1):
        h = min(k, i, n - 1 - i)
        out[i] = pts[i - h:i + h + 1].mean(axis=0)
    return out


def _arc_length(pts: NDArray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def _clean(mask: NDArray[np.bool_], cfg: ExtractionConfig) -> tuple[NDArray[np.bool_], int, int]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= cfg.min_component_area_px
    keep[0] = False
    cleaned = keep[lab]
    removed = int(n - keep.sum())
    holes, nh = ndimage.label(~cleaned)
    hsizes = np.bincount(holes.ravel())
    border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    fill = (hsizes <= cfg.max_hole_area_px)
    fill[border] = False
    fill[0] = False
    cleaned |= fill[holes]
    return cleaned, removed, int(fill.sum())


def _pixel_graph(skel: NDArray[np.bool_]):
    h, w = skel.shape
    flat = np.flatnonzero(skel)
    ids = np.full(h * w, -1, dtype=np.int64)
    ids[flat] = np.arange(flat.size)
    rr, cc = np.divmod(flat, w)
    padded = np.pad(skel, 1)
    rows, cols = [], []
    for dr, dc in _OFFSETS:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        j = np.full(flat.size, -1)
        j[ok] = ids[r2[ok] * w + c2[ok]]
        ok &= j >= 0
        if dr != 0 and dc != 0:  # drop a diagonal link when a 4-connected detour exists
            ok &= ~(padded[rr + dr + 1, cc + 1] | padded[rr + 1, cc + dc + 1])
        rows.append(np.flatnonzero(ok))
        cols.append(j[ok])
    i, j = np.concatenate(rows), np.concatenate(cols)
    adj = sparse.csr_matrix((np.ones(i.size, dtype=np.int8), (i, j)), shape=(flat.size, flat.size))
    return flat, rr, cc, adj


# ----------------------------------------------------------------------------- main

def extract_vessel_graph(mask: ArrayLike, fov: ArrayLike | None = None,
                         probability: ArrayLike | None = None,
                         config: ExtractionConfig | None = None) -> VesselGraph:
    """Extract a vessel graph from a binary vessel mask.

    Args:
        mask: 2D array; nonzero (or > 0.5 for floats) = vessel.
        fov: optional 2D field-of-view mask (nonzero = inside the camera aperture).
        probability: optional soft vessel probability in [0, 1] (same shape) used
            for sub-pixel width; the topology always comes from `mask`.
        config: extraction parameters.
    """
    cfg = config or ExtractionConfig()
    m = np.asarray(mask)
    if m.ndim != 2 or min(m.shape) < 3:
        raise ValueError(f"mask must be a 2D image, got shape {m.shape}")
    binary = (m > 0.5) if np.issubdtype(m.dtype, np.floating) else (m != 0)
    h, w = binary.shape
    fov_b = None
    if fov is not None:
        fov_b = np.asarray(fov) != 0
        if fov_b.shape != binary.shape:
            raise ValueError("fov must have the same shape as mask")
        binary &= fov_b
    prob = None
    if probability is not None:
        prob = np.asarray(probability, dtype=np.float64)
        if prob.shape != binary.shape or not np.all(np.isfinite(prob)) or prob.min() < 0 or prob.max() > 1:
            raise ValueError("probability must match mask shape with values in [0, 1]")

    binary, n_removed, n_filled = _clean(binary, cfg)
    if not binary.any():
        raise ValueError("no vessels left after cleaning; check the mask")
    skel = skeletonize(binary)
    edt = ndimage.distance_transform_edt(binary)

    flat, rr, cc, adj = _pixel_graph(skel)
    deg = np.diff(adj.indptr)
    n_pix = flat.size
    nbrs = lambda i: adj.indices[adj.indptr[i]:adj.indptr[i + 1]]  # noqa: E731

    # junction pixels: degree >= 3, or part of a 2x2 block
    block = skel[:-1, :-1] & skel[1:, :-1] & skel[:-1, 1:] & skel[1:, 1:]
    in_block = np.zeros_like(skel)
    for dr in (0, 1):
        for dc in (0, 1):
            in_block[dr:h - 1 + dr, dc:w - 1 + dc] |= block
    is_junc = (deg >= 3) | in_block.ravel()[flat]
    is_tip = (deg == 1) & ~is_junc

    # ---- nodes
    node_of = np.full(n_pix, -1)
    nodes: list[dict] = []
    jidx = np.flatnonzero(is_junc)
    if jidx.size:
        sub = adj[jidx][:, jidx]
        nc, lab = csgraph.connected_components(sub, directed=False)
        for k in range(nc):
            pix = jidx[lab == k]
            node_of[pix] = len(nodes)
            nodes.append({"kind": "junction", "rc": np.array([rr[pix].mean(), cc[pix].mean()]),
                          "r": float(edt.ravel()[flat[pix]].max()), "alive": True})
    for p in np.flatnonzero(is_tip):
        node_of[p] = len(nodes)
        nodes.append({"kind": "tip", "rc": np.array([rr[p], cc[p]], float),
                      "r": float(edt.ravel()[flat[p]]), "alive": True})

    def pix_rc(p):
        return np.array([rr[p], cc[p]], float)

    # ---- trace segments; a segment = node u, node v, polyline pts, width runs
    segs: list[dict] = []
    used_link: set[tuple[int, int]] = set()
    visited = np.zeros(n_pix, dtype=bool)

    def new_seg(u, v, interior):
        pts = np.vstack([nodes[u]["rc"], *(pix_rc(p) for p in interior), nodes[v]["rc"]]) \
            if interior else np.vstack([nodes[u]["rc"], nodes[v]["rc"]])
        runs = [{"pix": list(interior), "a": nodes[u]["rc"], "ra": nodes[u]["r"],
                 "b": nodes[v]["rc"], "rb": nodes[v]["r"]}] if interior else []
        segs.append({"u": u, "v": v, "pts": pts, "runs": runs, "alive": True, "crossing": False})

    for p in np.flatnonzero(node_of >= 0):
        k = node_of[p]
        for q in nbrs(p):
            if node_of[q] == k:
                continue
            key = (min(p, q), max(p, q))
            if key in used_link:
                continue
            used_link.add(key)
            interior, prev, cur = [], p, q
            while node_of[cur] < 0:
                interior.append(cur)
                visited[cur] = True
                nxt = [x for x in nbrs(cur) if x != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
            if node_of[cur] < 0:  # defensive: malformed run, close it with a tip
                node_of[cur] = len(nodes)
                nodes.append({"kind": "tip", "rc": pix_rc(cur), "r": float(edt.ravel()[flat[cur]]),
                              "alive": True})
                interior = interior[:-1] if interior and interior[-1] == cur else interior
            used_link.add((min(prev, cur), max(prev, cur)))
            new_seg(k, int(node_of[cur]), interior)

    # pure loops (no junction or tip): cut with two nodes
    loop_left = np.flatnonzero((node_of < 0) & ~visited)
    while loop_left.size:
        start = loop_left[0]
        order, prev, cur = [start], -1, start
        while True:
            nxt = [x for x in nbrs(cur) if x != prev and x != start]
            if not nxt or visited[nxt[0]]:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            order.append(cur)
        visited[start] = True
        if len(order) >= 4:
            half = len(order) // 2
            a, b = order[0], order[half]
            for p in (a, b):
                node_of[p] = len(nodes)
                nodes.append({"kind": "split", "rc": pix_rc(p), "r": float(edt.ravel()[flat[p]]),
                              "alive": True})
            new_seg(int(node_of[a]), int(node_of[b]), order[1:half])
            new_seg(int(node_of[b]), int(node_of[a]), order[half + 1:])
        loop_left = np.flatnonzero((node_of < 0) & ~visited)

    stats = {"components_removed": n_removed, "holes_filled": n_filled,
             "skeleton_px": int(n_pix), "spurs_pruned": 0, "degree2_merged": 0,
             "crossings_resolved": 0, "loops_split": 0}

    def degree():
        d = np.zeros(len(nodes), dtype=int)
        for s in segs:
            if s["alive"]:
                d[s["u"]] += 1
                d[s["v"]] += 1
        return d

    def length(s):
        return _arc_length(_smooth(s["pts"], cfg.smoothing_window))

    def oriented(s, from_node):
        """(pts, runs) of segment s oriented to start at from_node."""
        if s["u"] == from_node:
            return s["pts"], s["runs"]
        return s["pts"][::-1], [{"pix": r["pix"][::-1], "a": r["b"], "ra": r["rb"],
                                 "b": r["a"], "rb": r["ra"]} for r in s["runs"][::-1]]

    def join(s1, s2, node, middle_pts=None):
        """New segment: far end of s1 -> node -> far end of s2."""
        p1, r1 = oriented(s1, node)
        p2, r2 = oriented(s2, node)
        far1 = s1["v"] if s1["u"] == node else s1["u"]
        far2 = s2["v"] if s2["u"] == node else s2["u"]
        parts = [p1[::-1]] + ([middle_pts] if middle_pts is not None else []) + [p2]
        runs1 = [{"pix": r["pix"][::-1], "a": r["b"], "ra": r["rb"], "b": r["a"], "rb": r["ra"]}
                 for r in r1[::-1]]
        s1["alive"] = s2["alive"] = False
        segs.append({"u": far1, "v": far2, "pts": np.vstack(parts), "runs": runs1 + r2,
                     "alive": True, "crossing": s1["crossing"] or s2["crossing"]})
        return segs[-1]

    def incident(node):
        return [s for s in segs if s["alive"] and node in (s["u"], s["v"])]

    # ---- prune spurs, then merge degree-2 junctions; repeat until stable
    changed = True
    while changed:
        changed = False
        d = degree()
        for s in segs:
            if not s["alive"]:
                continue
            for tip, other in ((s["u"], s["v"]), (s["v"], s["u"])):
                if (nodes[tip]["kind"] == "tip" and d[tip] == 1 and d[other] >= 3
                        and length(s) < cfg.spur_length_factor * nodes[other]["r"] + cfg.spur_min_px):
                    s["alive"] = False
                    nodes[tip]["alive"] = False
                    d[tip] -= 1
                    d[other] -= 1
                    stats["spurs_pruned"] += 1
                    changed = True
                    break
        d = degree()
        for k, nd in enumerate(nodes):
            if nd["alive"] and nd["kind"] == "junction" and d[k] == 2:
                inc = incident(k)
                if len(inc) == 2 and inc[0] is not inc[1]:
                    join(inc[0], inc[1], k)
                    nd["alive"] = False
                    stats["degree2_merged"] += 1
                    changed = True
                    d = degree()

    # ---- crossings
    def direction(s, node):
        """Unit direction of the arm itself, measured beyond the junction blob."""
        pts, _ = oriented(s, node)
        r = nodes[node]["r"]
        dist = np.linalg.norm(pts - pts[0], axis=1)
        near_d = 1.5 * r
        far_d = near_d + max(8.0, cfg.direction_reach_factor * r)
        i_near = int(np.argmax(dist >= near_d)) if np.any(dist >= near_d) else 0
        i_far = int(np.argmax(dist >= far_d)) if np.any(dist >= far_d) else len(pts) - 1
        if i_far <= i_near:
            i_near, i_far = 0, len(pts) - 1
        v = pts[i_far] - pts[i_near]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def angle(v1, v2):
        return math.degrees(math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))))

    possible = set()
    if cfg.resolve_crossings:
        d = degree()
        # (a) two degree-3 junctions joined by a short bridge
        for s in list(segs):
            if not s["alive"] or s["u"] == s["v"]:
                continue
            a, b = s["u"], s["v"]
            if not (nodes[a]["kind"] == nodes[b]["kind"] == "junction" and d[a] == 3 and d[b] == 3):
                continue
            if length(s) >= cfg.crossing_bridge_factor * (nodes[a]["r"] + nodes[b]["r"]):
                continue
            ia = [t for t in incident(a) if t is not s]
            ib = [t for t in incident(b) if t is not s]
            if len(ia) != 2 or len(ib) != 2 or any(t in ib for t in ia):
                continue
            da = [direction(t, a) for t in ia]
            db = [direction(t, b) for t in ib]
            best, best_score = None, -1.0
            for pairing in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
                score = min(angle(da[i], db[j]) for i, j in pairing)
                if score > best_score:
                    best, best_score = pairing, score
            if best_score < cfg.crossing_min_angle_deg:
                possible.update((a, b))
                continue
            bridge_pts, _ = oriented(s, a)
            s["alive"] = False
            for i, j in best:
                # far(ia[i]) -> a -> bridge -> b -> far(ib[j])
                p1, r1 = oriented(ia[i], a)
                p2, r2 = oriented(ib[j], b)
                far1 = ia[i]["v"] if ia[i]["u"] == a else ia[i]["u"]
                far2 = ib[j]["v"] if ib[j]["u"] == b else ib[j]["u"]
                runs1 = [{"pix": r["pix"][::-1], "a": r["b"], "ra": r["rb"], "b": r["a"], "rb": r["ra"]}
                         for r in r1[::-1]]
                ia[i]["alive"] = ib[j]["alive"] = False
                segs.append({"u": far1, "v": far2, "pts": np.vstack([p1[::-1], bridge_pts[1:-1], p2]),
                             "runs": runs1 + r2, "alive": True, "crossing": True})
            nodes[a]["alive"] = nodes[b]["alive"] = False
            stats["crossings_resolved"] += 1
            d = degree()
        # (b) single degree-4 node
        for k, nd in enumerate(nodes):
            if not (nd["alive"] and nd["kind"] == "junction" and d[k] == 4):
                continue
            inc = incident(k)
            if len(inc) != 4 or len({id(t) for t in inc}) != 4:
                possible.add(k)
                continue
            dirs = [direction(t, k) for t in inc]
            best, best_score = None, -1.0
            for pairing in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
                score = min(angle(dirs[i], dirs[j]) for i, j in pairing)
                if score > best_score:
                    best, best_score = pairing, score
            if best_score < cfg.crossing_min_angle_deg:
                possible.add(k)
                continue
            for i, j in best:
                join(inc[i], inc[j], k)["crossing"] = True
            nd["alive"] = False
            stats["crossings_resolved"] += 1
            d = degree()
    d = degree()
    possible.update(k for k, nd in enumerate(nodes) if nd["alive"] and nd["kind"] == "junction" and d[k] >= 4)

    # ---- sub-pixel width: assign vessel pixels to nearest centreline pixel
    seg_list = [s for s in segs if s["alive"]]
    owner = np.full(n_pix, -1)
    for si, s in enumerate(seg_list):
        for run in s["runs"]:
            owner[run["pix"]] = si
    skel_owned = np.zeros(h * w, dtype=bool)
    skel_owned[flat] = True
    _, (ir, ic) = ndimage.distance_transform_edt(~skel_owned.reshape(h, w), return_indices=True)
    pix_index = np.full(h * w, -1)
    pix_index[flat] = np.arange(n_pix)
    nearest = pix_index[(ir * w + ic).ravel()]
    if prob is not None:
        near_vessel = ndimage.binary_dilation(binary, iterations=2)
        weight = np.where(near_vessel, prob, 0.0).ravel()
    else:
        weight = binary.ravel().astype(float)
    cell = np.bincount(nearest[weight > 0], weights=weight[weight > 0], minlength=n_pix)

    radius, radius_edt, reliable = [], [], []
    edt_flat = edt.ravel()
    for s in seg_list:
        area, arclen, edt_vals = 0.0, 0.0, []
        for run in s["runs"]:
            pix = np.asarray(run["pix"], dtype=np.int64)
            if pix.size == 0:
                continue
            pts = np.column_stack([rr[pix], cc[pix]]).astype(float)
            edt_vals.extend(edt_flat[flat[pix]].tolist())
            sm = _smooth(pts, cfg.smoothing_window)
            inc = np.linalg.norm(np.diff(sm, axis=0), axis=1)
            step = np.zeros(pix.size)
            if pix.size > 1:
                step[:-1] += inc / 2
                step[1:] += inc / 2
                step[0] += inc[0] / 2
                step[-1] += inc[-1] / 2
            else:
                step[:] = 1.0
            core = ((np.linalg.norm(pts - run["a"], axis=1) >= cfg.trim_factor * run["ra"] + 1.0)
                    & (np.linalg.norm(pts - run["b"], axis=1) >= cfg.trim_factor * run["rb"] + 1.0))
            area += float(cell[pix[core]].sum())
            arclen += float(step[core].sum())
        r_edt = max(float(np.median(edt_vals)) - 0.5, 0.5) if edt_vals else math.nan
        if arclen >= 2.0:
            radius.append(area / (2.0 * arclen))
            reliable.append(True)
        else:
            radius.append(r_edt if math.isfinite(r_edt) else
                          max(0.5, min(nodes[s["u"]]["r"], nodes[s["v"]]["r"]) - 0.5))
            reliable.append(False)
        radius_edt.append(r_edt if math.isfinite(r_edt) else radius[-1])

    # ---- split self-loops (VesselNetwork forbids them), compact node indices
    final = []
    for s, r, re_, ok in zip(seg_list, radius, radius_edt, reliable):
        if s["u"] != s["v"]:
            final.append((s["u"], s["v"], s["pts"], r, re_, ok, s["crossing"]))
            continue
        mid = len(s["pts"]) // 2
        nodes.append({"kind": "split", "rc": s["pts"][mid].copy(), "r": r, "alive": True})
        m_id = len(nodes) - 1
        final.append((s["u"], m_id, s["pts"][:mid + 1], r, re_, ok, s["crossing"]))
        final.append((m_id, s["v"], s["pts"][mid:], r, re_, ok, s["crossing"]))
        stats["loops_split"] += 1
    used = sorted({x for f in final for x in f[:2]})
    remap = {old: new for new, old in enumerate(used)}

    node_rc = np.array([nodes[k]["rc"] for k in used]).reshape(-1, 2)
    kinds = tuple(nodes[k]["kind"] for k in used)
    edges = np.array([[remap[f[0]], remap[f[1]]] for f in final], dtype=np.int64).reshape(-1, 2)
    paths = tuple(_smooth(np.asarray(f[2], float), cfg.smoothing_window) for f in final)
    deg_final = np.bincount(edges.ravel(), minlength=len(used))

    border = np.minimum.reduce([node_rc[:, 0], node_rc[:, 1], h - 1 - node_rc[:, 0], w - 1 - node_rc[:, 1]])
    if fov_b is not None:
        fov_dist = ndimage.distance_transform_edt(fov_b)
        ri = np.clip(np.round(node_rc).astype(int), 0, [h - 1, w - 1])
        border = np.minimum(border, fov_dist[ri[:, 0], ri[:, 1]])
    node_r = np.array([nodes[k]["r"] for k in used])
    truncated = (deg_final == 1) & (border <= cfg.border_margin_px + node_r)
    poss = np.array([k in possible for k in used], dtype=bool)

    stats.update({"nodes": len(used), "segments": len(final), "tips": int((deg_final == 1).sum()),
                  "junctions": int((deg_final >= 3).sum()), "truncated_tips": int(truncated.sum()),
                  "possible_crossings": int(poss.sum()),
                  "unreliable_width": int((~np.array([f[5] for f in final], dtype=bool)).sum())})
    return VesselGraph(
        image_shape=(h, w), node_rc=node_rc, node_kind=kinds, node_truncated=truncated,
        node_possible_crossing=poss, edges=edges, paths_rc=paths,
        length_px=np.array([_arc_length(p) for p in paths]),
        radius_px=np.array([f[3] for f in final]), radius_edt_px=np.array([f[4] for f in final]),
        radius_reliable=np.array([f[5] for f in final], dtype=bool),
        through_crossing=np.array([f[6] for f in final], dtype=bool), stats=stats)
