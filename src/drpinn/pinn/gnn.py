"""Physics-informed graph neural network: a learned iterative solver.

Trained ONCE across many random networks with the physics residual as the
only loss (no solver labels), then applied to unseen networks in one
forward pass. Each of its T steps computes the current Kirchhoff residual r,
passes messages between neighbouring nodes, and updates pressures:

    p <- BC(p - (r * (1 + a) + b)),   (a, b) = MLP(node state, messages)

With a = b = 0 a step is exactly one Jacobi sweep; the last layer starts at
zero, so the untrained network IS Jacobi(T) and training can only aim to do
better. That makes Jacobi(T) and CG(T) the fair baselines for the same budget.
All inputs are dimensionless, so a model trained on synthetic trees can be
applied to networks extracted from real images.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .graph_physics import GraphProblem


def batch_problems(problems: list[GraphProblem]) -> GraphProblem:
    """Disjoint union of several networks (node indices offset)."""
    off, parts = 0, {k: [] for k in ("src", "dst", "w", "fixed", "fixed_value", "bias", "robin_frac", "coords", "g", "diag")}
    for p in problems:
        for k in parts:
            v = getattr(p, k)
            parts[k].append(v + off if k in ("src", "dst") else v)
        off += p.n
    cat = {k: torch.cat(v) for k, v in parts.items()}
    first = problems[0]
    return GraphProblem(off, **cat, p_low=first.p_low, p_span=first.p_span, network=first.network,
                        bc=first.bc, viscosity=first.viscosity)


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class PhysicsGNN(nn.Module):
    def __init__(self, hidden: int = 32, steps: int = 30, residual_scale: float = 20.0):
        super().__init__()
        self.steps, self.scale = steps, residual_scale
        self.encode = _mlp(5, hidden, hidden)
        self.message = _mlp(2 * hidden + 4, hidden, hidden)
        self.update = _mlp(2 * hidden + 1, hidden, hidden)
        self.correct = nn.Linear(hidden + 1, 2)
        nn.init.zeros_(self.correct.weight)
        nn.init.zeros_(self.correct.bias)

    def node_features(self, pr: GraphProblem) -> torch.Tensor:
        deg = torch.zeros(pr.n, dtype=torch.float32).index_add_(0, pr.dst, torch.ones_like(pr.dst, dtype=torch.float32))
        return torch.stack([pr.fixed.float(), pr.fixed_value.float(), pr.robin_frac.float(),
                            pr.bias.float().clamp(-1, 1), deg / 4], dim=1)

    def forward(self, pr: GraphProblem, steps: int | None = None, return_all: bool = False):
        pr32 = pr.to(torch.float32)
        h = self.encode(self.node_features(pr32))
        p = pr32.initial_guess()
        logw = torch.log(pr32.w.clamp_min(1e-12)).unsqueeze(1) / 10
        out = []
        for _ in range(steps or self.steps):
            r = p - pr32.neighbour_mean(p) - pr32.bias
            r = torch.where(pr32.fixed, torch.zeros_like(r), r)
            rs = (self.scale * r).unsqueeze(1)
            e = torch.cat([h[pr32.src], h[pr32.dst], rs[pr32.src], rs[pr32.dst], pr32.w.unsqueeze(1), logw], dim=1)
            m = torch.zeros_like(h).index_add_(0, pr32.dst, pr32.w.unsqueeze(1) * self.message(e))
            h = h + self.update(torch.cat([h, m, rs], dim=1))
            ab = self.correct(torch.cat([h, rs], dim=1))
            delta = r * (1 + ab[:, 0]) + ab[:, 1] / self.scale
            p = pr32.apply_bc(p - delta)
            out.append(p)
        return out if return_all else p


def physics_loss(model: PhysicsGNN, pr: GraphProblem) -> torch.Tensor:
    """Mean squared residual after the last step, plus a small weight on earlier steps."""
    ps = model(pr, return_all=True)
    pr32 = pr.to(torch.float32)
    w = torch.linspace(0.1, 1.0, len(ps))
    return sum(wi * pr32.loss(p) for wi, p in zip(w, ps)) / w.sum()


def train_gnn(make_problem, n_iter: int = 400, batch: int = 8, hidden: int = 32, steps: int = 30,
              lr: float = 2e-3, seed: int = 0, log_every: int = 50) -> tuple[PhysicsGNN, dict]:
    """`make_problem(rng)` returns a fresh random GraphProblem for every draw."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PhysicsGNN(hidden, steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
    history, t0 = [], time.perf_counter()
    with torch.no_grad():
        start = float(np.mean([physics_loss(model, batch_problems([make_problem(rng) for _ in range(batch)]))
                               for _ in range(3)]))
    for it in range(n_iter):
        pr = batch_problems([make_problem(rng) for _ in range(batch)])
        loss = physics_loss(model, pr)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == n_iter - 1:
            history.append((it, loss.item()))
            print(f"  GNN step {it}/{n_iter}  physics loss {loss.item():.3e}", flush=True)
    return model, {"seconds": time.perf_counter() - t0, "history": history, "untrained_loss": start,
                   "hidden": hidden, "steps": steps, "iterations": n_iter, "batch": batch}


def gnn_solve(model: PhysicsGNN, problem: GraphProblem):
    from .methods import FitResult

    t0 = time.perf_counter()
    with torch.no_grad():
        p = model(problem).double()
    p = problem.apply_bc(p)
    return FitResult(f"PI-GNN({model.steps})", p.numpy(), time.perf_counter() - t0, model.steps,
                     float(problem.loss(p)))
