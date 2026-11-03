# Physics-informed learning vs the classical solver (Milestone 8)

**Question** (PDF §9.4, plan assumption A11): for the 1D vessel-network model, does a physics-informed neural network add value over the classical sparse solver?

## The physics loss
For every node without a fixed pressure, mass conservation with Poiseuille flows reads
Σ_j g_ij (p_i − p_j) + gt_i (p_i − pref_i) − q_i = 0,
where:
- g_ij is the conductance of the segment between nodes i and j;
- gt_i and pref_i are the terminal resistance and reference pressure of an outlet;
- q_i is a prescribed inflow.

Dividing by the node's total conductance D_i gives the Jacobi-normalised residual
**r_i = p_i − (Σ_j g_ij p_j + gt_i pref_i + q_i) / D_i**.
It is dimensionless once pressures are scaled to [0, 1] between the lowest and highest boundary pressure, and it is exactly 0 at the classical solution (tested for fixed-pressure, terminal-resistance and inflow boundary conditions).

Fixed-pressure nodes are imposed exactly ("hard" boundary conditions) by every method, so the loss contains physics only.

## Methods compared (identical networks, boundary conditions and viscosity)
| Method | Training | Per-network cost |
|---|---|---|
| Direct sparse solve (reference) | none | one sparse factorisation |
| Jacobi(T), conjugate gradient CG(T) | none | T iterations |
| **PI-GNN(T)** — physics-informed graph neural network | once, on random synthetic trees, physics loss only (no solver answers) | T learned steps |
| Nodal PINN | per network | L-BFGS on the physics loss, nodal pressures as parameters |
| Coordinate PINN (classic PINN) | per network | MLP of node (x, y) with Fourier features, Adam then L-BFGS |

**How the PI-GNN steps work.** Each step:
1. computes the residual r at every node;
2. passes messages between neighbouring nodes, weighted by conductance;
3. updates p ← p − (r·(1 + a) + b), where a and b come from the network.

Its last layer starts at zero, so the untrained network is *exactly* Jacobi(T) (tested), and Jacobi(T) and CG(T) are the fair same-budget baselines. All inputs are dimensionless, so a model trained on synthetic trees can be applied to graphs from real images.

**Fairness.** The coordinate PINN's Fourier scale is chosen on two separate tuning networks; test networks are never used for tuning. Test networks use seeds the training generator cannot produce.

## Results (SYNTHETIC test networks, my CPU; DRIVE-geometry graphs are added in the Colab run)
20 random trees with 6–9 generations, loops, and two outlet models; 64–512 nodes. Errors are fractions of the inlet–outlet pressure drop.

| Method | Median error | Worst error | WSS top-10 % kept | Median time |
|---|---|---|---|---|
| Direct solve | 0 | 0 | 1.00 | 2.2 ms |
| Jacobi(30) | 1.8e-1 | 4.8e-1 | 0.71 | 0.8 ms |
| CG(30) | 4.2e-2 | 2.6e-1 | 0.96 | 1.7 ms |
| PI-GNN(30) | 5.4e-3 | 3.5e-1 | 1.00 | 14.5 ms (+ 90 s training once) |
| Nodal PINN | 1.2e-5 | 6.8e-3 | 1.00 | 2.2 s |
| Coordinate PINN (σ = 30 chosen in tuning) | 8.3e-3 | 1.3e-1 | 1.00 | 10.1 s |

**Findings**
1. **For the forward problem, no PINN variant beats the direct solve.** The direct solve is exact and the fastest. Per-network PINNs cost 1,000–5,000× more time and are still approximate.
2. **The classic coordinate PINN scales badly.** A first, untuned run on one 256-node network had 34 % error. After tuning, on 512-node networks it still reached 27–34 % error in the quick run. A position-based network cannot easily represent pressures on a branching graph.
3. **The PI-GNN is a genuine learned acceleration of iterative solving.** At the same 30 steps, its median error is 8× lower than CG and 33× lower than Jacobi. But its worst cases are large, and it is still slower and less exact than the direct solve.
4. **A small physics loss is not proof of accuracy.** The final loss tracks the pressure error, but at the same loss the error varies about 100-fold (`m8_accuracy_vs_time.png`, right panel). Long trees have smooth error patterns that leave only a small residual.

**Implication for the thesis.** Report the PINN as evaluated and found not to add value for the forward problem with known geometry, as the PDF anticipates (§9.4: "a null result … is a scientifically useful finding"). Physics-informed learning could add value in *inverse* problems, inferring unknown parameters from measurements, but ordinary fundus photographs do not provide such measurements. Discuss the title's framing with your supervisor.

Reproduce: `python scripts/m8_pinn_vs_solver.py --drive-dir data/DRIVE` (about 6 min on my CPU for the synthetic networks).
