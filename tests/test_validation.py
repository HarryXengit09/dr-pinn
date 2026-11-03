"""Invalid inputs must fail loudly, never produce silent numbers."""

import math

import pytest

from drpinn.hemodynamics import BoundaryConditions, VesselNetwork, single_pipe, solve_network

MU = 3.5e-3


@pytest.mark.parametrize(
    "edges,lengths,radii",
    [
        ([[0, 1]], [1e-3], [-5e-5]),          # negative radius
        ([[0, 1]], [0.0], [5e-5]),            # zero length
        ([[0, 1]], [math.nan], [5e-5]),       # NaN
        ([[0, 0]], [1e-3], [5e-5]),           # self-loop
        ([[0, 2]], [1e-3], [5e-5]),           # node index out of range
        ([[0, 1]], [1e-3, 1e-3], [5e-5]),     # length/edge count mismatch
    ],
)
def test_invalid_geometry_rejected(edges, lengths, radii):
    with pytest.raises(ValueError):
        VesselNetwork(2, edges, lengths, radii)


def test_isolated_node_rejected():
    with pytest.raises(ValueError, match="without any segment"):
        VesselNetwork(3, [[0, 1]], [1e-3], [5e-5])


def test_pixels_passed_as_metres_warns():
    with pytest.warns(UserWarning, match="metres"):
        VesselNetwork(2, [[0, 1]], [40.0], [3.0])  # pixel values, not metres


def test_unanchored_component_rejected():
    net = VesselNetwork(4, [[0, 1], [2, 3]], [1e-3, 1e-3], [5e-5, 5e-5])
    with pytest.raises(ValueError, match="component"):
        solve_network(net, BoundaryConditions(pressure_pa={0: 100.0, 1: 0.0}), MU)


def test_neumann_only_rejected():
    net = single_pipe(1e-3, 5e-5).network
    with pytest.raises(ValueError):
        solve_network(net, BoundaryConditions(inflow_m3_s={0: 1e-11, 1: -1e-11}), MU)


def test_conflicting_boundary_conditions_rejected():
    net = single_pipe(1e-3, 5e-5).network
    with pytest.raises(ValueError, match="more than one"):
        solve_network(net, BoundaryConditions(pressure_pa={0: 1.0, 1: 0.0},
                                              inflow_m3_s={0: 1e-11}), MU)
    with pytest.raises(ValueError, match="both an inlet and an outlet"):
        BoundaryConditions.inlet_outlet_pressure([0], 1.0, [0], 0.0)


@pytest.mark.parametrize("mu", [0.0, -1e-3, math.nan, [1e-3, 2e-3]])
def test_invalid_viscosity_rejected(mu):
    net = single_pipe(1e-3, 5e-5).network
    with pytest.raises(ValueError):
        solve_network(net, BoundaryConditions(pressure_pa={0: 1.0, 1: 0.0}), mu)


def test_nonpositive_terminal_resistance_rejected():
    net = single_pipe(1e-3, 5e-5).network
    with pytest.raises(ValueError):
        solve_network(net, BoundaryConditions(pressure_pa={0: 1.0},
                                              terminal_resistance={1: (0.0, 0.0)}), MU)


def test_network_arrays_are_read_only():
    net = single_pipe(1e-3, 5e-5).network
    with pytest.raises(ValueError):
        net.radius_m[0] = 1.0
