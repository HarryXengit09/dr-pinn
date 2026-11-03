# Results summary

Generated 2026-11-01T08:47 by `scripts/m11_collect_results.py` (code 0.8.2). Every number is copied from the file named in its section; nothing is recomputed.

> Research prototype, not a medical device. Model-derived risk estimates are experimental, depend on stated modelling assumptions, and require independent clinical validation.

## Warnings
- M4: the split manifest was built before the IDRiD fix (353 IDRiD grading rows with old ids). Re-run step 11d, then step 15; until then every IDRiD triage number is invalid
- M6: IDRiD results are left out (made before the IDRiD fix, or from an old or different manifest); re-run step 11d, then step 15, to get valid external results
- M7 · Shear risk vs lesions (core experiment): made by code version 0.11.0, which is not a version of this project (installed: 0.8.2). Re-run that step with the installed code.
- M8: 'quick' run (fewer networks); the 'full' run is the one to report

## M4 · Datasets and splits
Source `reports/m4_data/integrity_report.json`. Manifest fingerprint **196f47a2f7c67930** (quote it: it identifies the exact split).

| dataset | split | photos | grades 0/1/2/3/4 |
|---|---|---|---|
| APTOS2019/grading | test | 541 | 269 / 54 / 148 / 27 / 43 |
| APTOS2019/grading | train | 2503 | 1262 / 249 / 680 / 120 / 192 |
| APTOS2019/grading | val | 539 | 270 / 57 / 147 / 25 / 40 |
| DRIVE/vessels | cv | 20 | 0 / 0 / 0 / 0 / 0 |
| IDRiD/grading | test | 53 | 0 / 0 / 26 / 18 / 9 |
| IDRiD/grading | train | 261 | 113 / 15 / 77 / 38 / 18 |
| IDRiD/grading | val | 39 | 20 / 3 / 9 / 5 / 2 |
| IDRiD/segmentation | test | 28 | 0 / 0 / 0 / 0 / 0 |
| IDRiD/segmentation | train | 51 | 0 / 0 / 0 / 0 / 0 |
| RITE/av | cv | 40 | 0 / 0 / 0 / 0 / 0 |

## M5 · Vessel and artery/vein segmentation
Source `reports/m5_full/m5_results.json` (preset full, artery/vein trained: True). Mean ± SD over photos.

| metric | 5-fold CV (DRIVE, 1st observer) | test (DRIVE test photos, 2nd observer) |
|---|---|---|
| Dice | 0.799 ± 0.058 | 0.841 ± 0.018 |
| AUC | 0.973 ± 0.021 | 0.983 ± 0.005 |
| Sensitivity | 0.799 ± 0.086 | 0.862 ± 0.039 |
| Specificity | 0.973 ± 0.012 | 0.974 ± 0.007 |
| Centreline recall | 0.780 ± 0.082 | 0.800 ± 0.058 |
| Centreline precision | 0.963 ± 0.074 | 0.981 ± 0.012 |
| Radius error (median, relative) | 0.174 ± 0.043 | 0.184 ± 0.075 |
| Artery/vein accuracy | 0.823 ± 0.057 | 0.803 ± 0.032 |

## M6 · DR grade triage
Source `reports/m6_full/m6_results.json` (code 0.7.2, preset full). Fixed on validation: epoch 9, temperature 0.85, referral threshold 0.723 (target sensitivity 0.9). 95 % bootstrap CIs.

| set | photos | QWK | accuracy | macro-F1 | referable AUC | sensitivity | specificity | referable ECE (calibrated) |
|---|---|---|---|---|---|---|---|---|
| APTOS validation (used to calibrate) | 539 | 0.921 [0.897, 0.941] | 0.863 [0.835, 0.889] | 0.713 [0.653, 0.767] | 0.986 [0.977, 0.992] | 0.901 [0.859, 0.939] | 0.963 [0.943, 0.982] | 0.025 |
| APTOS internal test | 541 | 0.911 [0.882, 0.935] | 0.850 [0.823, 0.880] | 0.689 [0.632, 0.739] | 0.988 [0.980, 0.995] | 0.885 [0.842, 0.926] | 0.975 [0.957, 0.991] | 0.033 |

Primary endpoints (pre-registered): referable AUC on the IDRiD official test; QWK on the APTOS internal test.

## M7 · Shear risk vs lesions (core experiment)
Source `reports/m7_full/m7_results.json` (code 0.11.0): 81 photos (27 official test), 13185 vessel patches, radius σ 0.258.

**Primary (pre-registered):** within-image AUROC of the Monte Carlo high-shear score for red lesions = **0.514 [0.477, 0.551]**, p = 0.435 vs 0.5, 27 test photos → **no evidence of association (CI includes 0.5)**.

| baseline | high-shear minus baseline (AUROC) | 95 % CI | p | photos |
|---|---|---|---|---|
| vessel_density | +0.113 | [0.068, 0.159] | 0.000 | 27 |
| near_fovea | -0.092 | [-0.151, -0.033] | 0.003 | 27 |
| high_shear_deterministic | -0.025 | [-0.054, 0.004] | 0.082 | 27 |
| thinness | -0.058 | [-0.115, -0.000] | 0.049 | 27 |

Exploratory: adding physics to position + geometry (logistic, trained on the 54 train photos) changes the AUROC by +0.001 [-0.009, 0.010].

Median rank stability: {'artery': 0.7808270628336228, 'vein': 0.7110171277997364}. Median share of red-lesion pixels drawn as vessel: 0.143.

## M8 · PINN vs classical solver
Source `reports/m8_quick/m8_results.json`. Pressure errors are fractions of the inlet–outlet pressure drop.

**DRIVE networks (n = 4)**

| method | pressure error (median) | pressure error (worst) | WSS top-10 % kept | time per network |
|---|---|---|---|---|
| direct solve (reference) | 0.0e+00 | 0.0e+00 | 1.00 | 4.5 ms |
| Jacobi(30) | 2.7e-01 | 3.8e-01 | 0.71 | 1.8 ms |
| CG(30) | 2.0e-01 | 3.1e-01 | 0.71 | 3.2 ms |
| PI-GNN(30) | 2.9e-01 | 3.3e-01 | 0.58 | 25.5 ms |
| nodal PINN | 2.6e-01 | 3.5e-01 | 0.68 | 2259.8 ms |
| coordinate PINN | 2.4e-01 | 3.7e-01 | 0.76 | 8656.8 ms |

**synthetic networks (n = 8)**

| method | pressure error (median) | pressure error (worst) | WSS top-10 % kept | time per network |
|---|---|---|---|---|
| direct solve (reference) | 0.0e+00 | 0.0e+00 | 1.00 | 4.1 ms |
| Jacobi(30) | 1.4e-01 | 2.8e-01 | 0.76 | 1.8 ms |
| CG(30) | 2.8e-02 | 2.6e-01 | 1.00 | 3.0 ms |
| PI-GNN(30) | 3.0e-02 | 4.2e-01 | 0.98 | 28.2 ms |
| nodal PINN | 9.4e-06 | 5.0e-03 | 1.00 | 1849.1 ms |
| coordinate PINN | 2.3e-02 | 3.4e-01 | 0.94 | 9450.0 ms |

## M9 · Optic-disc estimate used by the API
Source `reports/m9_disc/disc_check.json` (81 IDRiD photos). Expert disc diameter / field of view: median 0.161 (assumed 0.16).

| estimate | hit rate (inside expert disc) | median error (disc diameters) |
|---|---|---|
| brightness | 99% | 0.13 |
| brightness + vessels | 98% | 0.06 |
