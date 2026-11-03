"""Boundary-condition builders for visible vessel endpoints.

A fundus photo shows only the larger vessels; each visible endpoint feeds an
unseen capillary bed (or is just cut off by the image border or by
segmentation). Fixing every endpoint at one pressure ignores that downstream
resistance. The Murray-scaled terminal resistance below is a transparent,
single-parameter alternative.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .network import VesselNetwork


def murray_level_resistance(radius_m: float, viscosity_pa_s: float,
                            length_to_radius: float) -> float:
    """Resistance of ONE unseen Murray generation below a segment of radius r.

    Generation k below has 2^k segments of radius r 2^(-k/3) and length
    lambda * r 2^(-k/3); each has Poiseuille resistance 8 mu lambda 2^k / (pi r^3),
    so the 2^k in parallel give 8 mu lambda / (pi r^3), the same for every k.
    """
    for name, v in (("radius_m", radius_m), ("viscosity_pa_s", viscosity_pa_s),
                    ("length_to_radius", length_to_radius)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    return 8.0 * viscosity_pa_s * length_to_radius / (math.pi * radius_m**3)


def murray_terminal_resistance(network: VesselNetwork, outlet_nodes: Iterable[int],
                               viscosity_pa_s: float, downstream_levels: float,
                               length_to_radius: float, reference_pressure_pa: float
                               ) -> dict[int, tuple[float, float]]:
    """Robin BCs: each outlet drains through `downstream_levels` unseen Murray generations.

    ASSUMPTIONS: the unseen bed is a symmetric Murray tree with the given
    length/radius ratio and viscosity, ending at `reference_pressure_pa`.
    `downstream_levels` is an uncertain input to be sampled, not a measurement.
    Returns {node: (R in Pa s/m^3, reference pressure in Pa)} for BoundaryConditions.
    """
    if not math.isfinite(downstream_levels) or downstream_levels <= 0:
        raise ValueError("downstream_levels must be finite and > 0")
    deg = network.degree()
    out: dict[int, tuple[float, float]] = {}
    for node in outlet_nodes:
        node = int(node)
        if deg[node] != 1:
            raise ValueError(f"outlet node {node} has degree {deg[node]}; expected a vessel endpoint")
        e = int(np.flatnonzero((network.edges == node).any(axis=1))[0])
        r_level = murray_level_resistance(float(network.radius_m[e]), viscosity_pa_s, length_to_radius)
        out[node] = (downstream_levels * r_level, float(reference_pressure_pa))
    return out
