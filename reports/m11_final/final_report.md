# DR-PINN: final results summary

Generated 2026-11-01 by drpinn 0.11.0. Research prototype, not a medical device. Model-derived risk estimates are experimental, depend on stated modelling assumptions, and require independent clinical validation.

## 1. What was run

| Milestone | Status | File |
|---|---|---|
| M5 | OK | /content/drive/MyDrive/dr-pinn/reports/m5_full/m5_results.json |
| M6 | OK | /content/drive/MyDrive/dr-pinn/reports/m6_full/m6_results.json |
| M7 | OK | /content/drive/MyDrive/dr-pinn/reports/m7_full/m7_results.json |
| M8 | OK | /content/drive/MyDrive/dr-pinn/reports/m8_quick/m8_results.json |

## 2. Vessel and artery/vein segmentation (M5)

**Cross-validation on DRIVE training photos**

| metric | value |
|---|---|
| fold · mean | 2.000 |
| fold · sd | 1.451 |
| fold · n | 20.000 |
| dice · mean | 0.799 |
| dice · sd | 0.058 |
| dice · n | 20.000 |
| iou · mean | 0.669 |
| iou · sd | 0.073 |
| iou · n | 20.000 |
| sensitivity · mean | 0.799 |
| sensitivity · sd | 0.086 |
| sensitivity · n | 20.000 |
| specificity · mean | 0.973 |
| specificity · sd | 0.012 |
| specificity · n | 20.000 |
| accuracy · mean | 0.951 |
| accuracy · sd | 0.013 |
| accuracy · n | 20.000 |
| auc · mean | 0.973 |
| auc · sd | 0.021 |
| auc · n | 20.000 |
| centreline_recall · mean | 0.780 |
| centreline_recall · sd | 0.082 |
| centreline_recall · n | 20.000 |
| centreline_precision · mean | 0.963 |
| centreline_precision · sd | 0.074 |
| centreline_precision · n | 20.000 |
| radius_median_abs_error · mean | 0.174 |
| radius_median_abs_error · sd | 0.043 |
| radius_median_abs_error · n | 20.000 |
| segments_pred · mean | 158.700 |
| segments_pred · sd | 32.164 |
| segments_pred · n | 20.000 |
| segments_ref · mean | 248.350 |
| segments_ref · sd | 78.051 |
| segments_ref · n | 20.000 |
| av_accuracy · mean | 0.823 |
| av_accuracy · sd | 0.057 |
| av_accuracy · n | 20.000 |
| av_sensitivity · mean | 0.837 |

**Held-out DRIVE test photos**

| metric | value |
|---|---|
| dice · mean | 0.841 |
| dice · sd | 0.018 |
| dice · n | 20.000 |
| iou · mean | 0.726 |
| iou · sd | 0.026 |
| iou · n | 20.000 |
| sensitivity · mean | 0.862 |
| sensitivity · sd | 0.039 |
| sensitivity · n | 20.000 |
| specificity · mean | 0.974 |
| specificity · sd | 0.007 |
| specificity · n | 20.000 |
| accuracy · mean | 0.960 |
| accuracy · sd | 0.004 |
| accuracy · n | 20.000 |
| auc · mean | 0.983 |
| auc · sd | 0.005 |
| auc · n | 20.000 |
| centreline_recall · mean | 0.800 |
| centreline_recall · sd | 0.058 |
| centreline_recall · n | 20.000 |
| centreline_precision · mean | 0.981 |
| centreline_precision · sd | 0.012 |
| centreline_precision · n | 20.000 |
| radius_median_abs_error · mean | 0.184 |
| radius_median_abs_error · sd | 0.075 |
| radius_median_abs_error · n | 20.000 |
| segments_pred · mean | 167.750 |
| segments_pred · sd | 31.706 |
| segments_pred · n | 20.000 |
| segments_ref · mean | 269.650 |
| segments_ref · sd | 65.855 |
| segments_ref · n | 20.000 |
| av_accuracy · mean | 0.803 |
| av_accuracy · sd | 0.032 |
| av_accuracy · n | 20.000 |
| av_sensitivity · mean | 0.820 |
| av_sensitivity · sd | 0.054 |
| av_sensitivity · n | 20.000 |
| av_specificity · mean | 0.790 |


## 3. Core experiment: shear and lesions (M7)

**Primary (pre-registered):** within-image AUROC of high_shear_mc for red_lesion = **0.514** [0.477, 0.551], p = 0.435 against 0.5, over 27 test photos → **no evidence of association (CI includes 0.5)**.

Every score, red lesions, test photos:

| score | within-image AUROC | 95 % CI | p vs 0.5 |
|---|---|---|---|
| high_shear_mc | 0.514 | [0.477, 0.551] | 0.435 |
| low_shear_mc | 0.469 | [0.427, 0.510] | 0.132 |
| high_shear_deterministic | 0.540 | [0.496, 0.584] | 0.075 |
| low_shear_deterministic | 0.463 | [0.419, 0.506] | 0.091 |
| tortuosity | 0.512 | [0.469, 0.555] | 0.584 |
| thinness | 0.572 | [0.542, 0.602] | 0.000 |
| murray_deviation | 0.473 | [0.439, 0.507] | 0.109 |
| vessel_density | 0.401 | [0.363, 0.438] | 0.000 |
| near_fovea | 0.607 | [0.554, 0.659] | 0.000 |
| near_disc | 0.369 | [0.321, 0.417] | 0.000 |

High-shear minus each baseline (negative = baseline is better):

| baseline | difference | 95 % CI | p | photos |
|---|---|---|---|---|
| vessel_density | +0.113 | [0.068, 0.159] | 0.000 | 27 |
| near_fovea | -0.092 | [-0.151, -0.033] | 0.003 | 27 |
| high_shear_deterministic | -0.025 | [-0.054, 0.004] | 0.082 | 27 |
| thinness | -0.058 | [-0.115, -0.000] | 0.049 | 27 |

Exploratory: adding physics to position + geometry changes AUROC by +0.001 [-0.009, 0.010].

Median rank stability under radius noise: artery 0.78, vein 0.71. Median share of red-lesion pixels that fall on drawn vessels: 0.14. 81 photos analysed, 27 in the official test split.

## 4. PINN versus classical solver (M8)

**DRIVE networks**

| method | n | pressure error, median | worst | WSS top-10 % kept | WSS error, median | time |
|---|---|---|---|---|---|---|
| direct solve (reference) | 4 | 0.0e+00 | 0.0e+00 | 1.00 | 0.00 | 4.5 ms |
| Jacobi(30) | 4 | 2.7e-01 | 3.8e-01 | 0.71 | 0.86 | 1.8 ms |
| CG(30) | 4 | 2.0e-01 | 3.1e-01 | 0.71 | 0.72 | 3.2 ms |
| PI-GNN(30) | 4 | 2.9e-01 | 3.3e-01 | 0.58 | 0.91 | 25.5 ms |
| nodal PINN | 4 | 2.6e-01 | 3.5e-01 | 0.68 | 0.65 | 2.3 s |
| coordinate PINN | 4 | 2.4e-01 | 3.7e-01 | 0.76 | 0.50 | 8.7 s |

**synthetic networks**

| method | n | pressure error, median | worst | WSS top-10 % kept | WSS error, median | time |
|---|---|---|---|---|---|---|
| direct solve (reference) | 8 | 0.0e+00 | 0.0e+00 | 1.00 | 0.00 | 4.1 ms |
| Jacobi(30) | 8 | 1.4e-01 | 2.8e-01 | 0.76 | 0.27 | 1.8 ms |
| CG(30) | 8 | 2.8e-02 | 2.6e-01 | 1.00 | 0.01 | 3.0 ms |
| PI-GNN(30) | 8 | 3.0e-02 | 4.2e-01 | 0.98 | 0.03 | 28.2 ms |
| nodal PINN | 8 | 9.4e-06 | 5.0e-03 | 1.00 | 0.00 | 1.8 s |
| coordinate PINN | 8 | 2.3e-02 | 3.4e-01 | 0.94 | 0.01 | 9.4 s |


## 5. DR grade triage (M6)

- PRIMARY  referable AUC, IDRiD external test (n=53): nan [nan, nan]
-          at the validation threshold: sensitivity 0.755 [0.642, 0.868], specificity nan [nan, nan]
- PRIMARY  QWK, APTOS internal test (n=541): 0.911 [0.882, 0.935]; referable AUC 0.988 [0.980, 0.995]
- Fixed on validation: temperature 0.85, referral threshold P(referable) ≥ 0.723, uncertainty limit 0.52.

## 6. What the results support

- Model-derived high-shear regions did not locate red lesions better than chance (AUROC 0.514 [0.477, 0.551], 27 test photos).
- The data rule out a strong association: AUROCs above 0.55 are outside the interval.
- Simple baselines performed better than high-shear: near_fovea, thinness (paired CI below 0).
- Adding the physics scores to position and geometry did not change AUROC measurably (+0.001 [-0.009, 0.010]).
- DRIVE: the direct solve is exact by definition and took 4.5 ms; the most accurate alternative (CG(30)) had a median pressure error of 2.0e-01 in 3.2 ms; the fastest (Jacobi(30)) took 1.8 ms.
- synthetic: the direct solve is exact by definition and took 4.1 ms; the most accurate alternative (nodal PINN) had a median pressure error of 9.4e-06 in 1.8 s; the fastest (Jacobi(30)) took 1.8 ms.
- synthetic: the most accurate alternative was not faster than the direct solve.

### Wording to avoid

- Do not write that shear and lesions are unrelated: an interval that includes 0.5 is absence of evidence, not evidence of absence.
- Do not present the map as showing where disease is or will develop.
- Only 27 test photos: state the small sample wherever the interval is quoted.
- Do not generalise the M8 comparison beyond these networks, boundary conditions and iteration budgets.

## 7. Limitations

- Shear stress comes from a one-dimensional network model with assumed pressures, hematocrit and optic-disc size and from vessel radii measured on a 2-D photo; absolute values are illustrative and only within-eye rankings are used.
- Branches too small or too faint to segment are missing, and the artery and vein trees are connected only as far as the segmentation allows.
- The M7 lesion analysis uses one dataset (IDRiD) and one camera; its optic-disc masks are expert-drawn, whereas the deployed API locates the disc with an unvalidated heuristic.
- Lesions are labelled on the photo, but only vessel patches are scored; many lesions lie between vessels.
- The DR grade model is trained on public data (APTOS) and tested externally on IDRiD only; it is not validated for clinical use, other cameras, or other populations.
- Nothing in this project is a medical device, a diagnosis or a referral decision.
- Measured: shear ranks are only moderately stable under the measured radius error (median rank stability artery 0.78, vein 0.71).
- Measured: a median 14% of red-lesion pixels lie on drawn vessels, so most lesions are outside what a vessel-patch method sees.

## 8. Thesis outline built on these results

1. **Introduction.** Retinal haemodynamics and diabetic retinopathy; the hypothesis that high wall shear marks where lesions appear; why this was pre-registered.
2. **Data and ethics.** DRIVE/RITE, IDRiD, APTOS: licences, splits fixed before testing, what was never used for training.
3. **Methods: vessel graphs.** Segmentation, artery/vein separation, graph extraction, radius uncertainty (M5; report the radius error from cross-validation).
4. **Methods: haemodynamics.** 1-D network flow model, assumptions, Monte Carlo propagation of radius error, within-eye ranks.
5. **Methods: lesion association.** Patch scoring, within-image AUROC, baselines, paired comparisons, the pre-registered primary test.
6. **Results: segmentation.** M5 table and overlays.
7. **Results: shear versus lesions.** Primary: within-image AUROC 0.514 [0.477, 0.551]; baselines; physics-added test; rank stability; lesion coverage. Report it as a negative result if the interval includes 0.5.
8. **Results: PINN versus solver.** M8 tables: accuracy, time and WSS top-10 % overlap on synthetic and DRIVE networks.
9. **Results: DR triage.** M6 headline, calibration, external IDRiD test, Grad-CAM.
10. **Software.** API and dashboard (M9, M10): what it does, the disclaimers it shows, the heuristic optic-disc locator.
11. **Discussion.** Why the signal may be absent (radius error, sample size, lesions between vessels, model simplifications); what would test it better.
12. **Limitations and conclusion.** Section 7 above; claims limited to Section 6.
