"""Ways to obtain nodal pressures for one network, all scored the same way.

Per-network PINNs (trained from scratch for every network, physics loss only):
  * nodal PINN       - pressures are free parameters; minimise the residual
                       (Adam, then L-BFGS). Equivalent to solving the linear
                       system by optimisation.
  * coordinate PINN  - the classic PINN: an MLP maps a node's (x, y) position
                       to its pressure; same loss. A position-based network
                       cannot tell apart vessels that pass close to each other.
Classical iterative baselines with a fixed step budget:
  * Jacobi(k), conjugate gradient CG(k).
The direct sparse solve (hemodynamics.solver) is the reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import cg

from .graph_physics import GraphProblem


@dataclass
class FitResult:
    method: str
    p_scaled: np.ndarray
    seconds: float
    steps: int
    final_loss: float
    history: list = field(default_factory=list)


def _optimise(params, loss_fn, adam_steps: int, lbfgs_steps: int, lr: float, log_every: int = 100):
    history = []
    opt = torch.optim.Adam(params, lr=lr)
    for k in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if k % log_every == 0:
            history.append((k, loss.item()))
    if lbfgs_steps:
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_steps, history_size=50,
                               tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        lb.step(closure)
    final = loss_fn().item()
    history.append((adam_steps + lbfgs_steps, final))
    return final, history


def nodal_pinn(problem: GraphProblem, adam_steps: int = 2000, lbfgs_steps: int = 500, lr: float = 0.02,
               seed: int = 0) -> FitResult:
    torch.manual_seed(seed)
    theta = problem.initial_guess().clone().requires_grad_(True)
    t0 = time.perf_counter()
    final, hist = _optimise([theta], lambda: problem.loss(problem.apply_bc(theta)), adam_steps, lbfgs_steps, lr)
    p = problem.apply_bc(theta).detach().numpy()
    return FitResult("nodal PINN", p, time.perf_counter() - t0, adam_steps + lbfgs_steps, final, hist)


class CoordinateMLP(torch.nn.Module):
    """(x, y) -> scaled pressure. Random Fourier features let it represent sharper changes."""

    def __init__(self, width: int = 64, depth: int = 4, fourier: int = 32, sigma: float = 3.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, fourier, generator=g, dtype=torch.float64) * sigma)
        layers, d = [], 2 * fourier + 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width, dtype=torch.float64), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, 1, dtype=torch.float64))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        z = xy @ self.B
        return self.net(torch.cat([xy, torch.sin(z), torch.cos(z)], dim=1)).squeeze(1)


def coordinate_pinn(problem: GraphProblem, adam_steps: int = 2000, lbfgs_steps: int = 500, lr: float = 2e-3,
                    seed: int = 0, **mlp_kw) -> FitResult:
    torch.manual_seed(seed)
    model = CoordinateMLP(seed=seed, **mlp_kw)
    t0 = time.perf_counter()
    final, hist = _optimise(list(model.parameters()),
                            lambda: problem.loss(problem.apply_bc(model(problem.coords))), adam_steps, lbfgs_steps, lr)
    p = problem.apply_bc(model(problem.coords)).detach().numpy()
    return FitResult("coordinate PINN", p, time.perf_counter() - t0, adam_steps + lbfgs_steps, final, hist)


def jacobi(problem: GraphProblem, steps: int) -> FitResult:
    t0 = time.perf_counter()
    p = problem.initial_guess()
    with torch.no_grad():
        for _ in range(steps):
            p = problem.apply_bc(problem.neighbour_mean(p) + problem.bias)
    return FitResult(f"Jacobi({steps})", p.numpy(), time.perf_counter() - t0, steps, float(problem.loss(p)))


def conjugate_gradient(problem: GraphProblem, steps: int) -> FitResult:
    """CG on the symmetric system K_ff p_f = b_f - K_fF p_F, at most `steps` iterations."""
    t0 = time.perf_counter()
    n = problem.n
    src, dst, g = problem.src.numpy(), problem.dst.numpy(), problem.g.numpy()
    k = (sparse.diags(problem.diag.numpy()) - sparse.csr_matrix((g, (dst, src)), shape=(n, n))).tocsr()
    fixed = problem.fixed.numpy()
    free = ~fixed
    b = problem.diag.numpy() * problem.bias.numpy()
    p = problem.initial_guess().numpy().copy()
    if free.any():
        kff = k[free][:, free]
        rhs = b[free] - k[free][:, fixed] @ p[fixed]
        try:
            x, _ = cg(kff, rhs, x0=p[free], rtol=1e-14, atol=0.0, maxiter=steps)
        except TypeError:  # SciPy < 1.12
            x, _ = cg(kff, rhs, x0=p[free], tol=1e-14, maxiter=steps)
        p[free] = x
    loss = float(problem.loss(torch.as_tensor(p, dtype=problem.w.dtype)))
    return FitResult(f"CG({steps})", p, time.perf_counter() - t0, steps, loss)
