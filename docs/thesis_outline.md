# Thesis outline (Milestone 11)

**Working title (to agree with Dr. Sangeeta Yadav):** *Physics-Informed Neural Networks for Retinal Hemodynamic Stress Estimation and Early Risk Mapping in Diabetic Retinopathy*.

M8 showed that a PINN does not beat the classical solver for the forward problem (known geometry and boundary conditions). Two honest options:
- keep the title and present the PINN comparison as a result; or
- reword it, for example to *Physics-based retinal hemodynamic risk mapping, with an evaluation of physics-informed neural networks*.

Decide this before writing Chapter 1.

All numbers come from `reports/summary/results_summary.md`, written by notebook step 17 (`scripts/m11_collect_results.py`). Never type a number from memory; copy it from that file, which also names the source file of each number.

> Research prototype, not a medical device. Say so in the abstract, in the methods and in every figure caption that shows a risk map.

---

## 1 · Introduction
- **The problem:** diabetic retinopathy (DR) screening, and why vessel-level risk is interesting. Add your own literature citations here; none are supplied by the code.
- **The idea:** combine a DR-grade triage model with a physics model of blood flow in the visible retinal vessels, and ask whether the physics adds anything.
- **Research questions.** Align the numbering with the PDF §11.1.
  - **Physics vs geometry (RQ1):** does a wall-shear-stress ranking relate to lesions beyond what vessel geometry and position explain? (M7 baselines and combined models.)
  - **Robustness (RQ3):** are the rankings stable under the unknowns a photo cannot measure? (M1 theorem, M2 study, M7 rank stability.)
  - **Triage:** how well does a lightweight CNN grade DR, and does it transfer to another camera? (M6)
  - **PINN value:** does a PINN or physics-informed GNN add value over the classical solver? (M8)
- **Contributions:**
  - an open, tested pipeline from photo to vessel graph to network hemodynamics;
  - a proof that blood pressure, viscosity, image scale and uniform radius bias cannot change the ranking in the Newtonian model;
  - a pre-registered association test with null and positive controls;
  - an honest PINN comparison;
  - a research dashboard.

## 2 · Background
- DR grading (ICDR scale; the "referable" definition) and screening practice.
- Retinal vascular anatomy: arteries and veins, crossings, the optic disc.
- Network hemodynamics: Hagen–Poiseuille, Kirchhoff's laws, wall shear stress, Murray's law, the Fåhræus–Lindqvist effect (Pries 1992), Carreau–Yasuda.
- Shear stress and endothelial damage: both high- and low-shear hypotheses (plan A6).
- PINNs and graph neural networks.
- Related work on retinal vessel segmentation and DR deep learning. This is your literature review; the `related-work` and `citation-check` skills can help draft and verify it.

## 3 · Data (M4)
- The datasets, their licences and citations (`docs/datasets.md`):
  - DRIVE: 40 photos;
  - RITE: artery/vein labels on the DRIVE photos;
  - IDRiD: 516 graded photos, plus 81 with lesion and disc masks;
  - APTOS 2019: 3,662 graded photos.
- **Integrity checks:**
  - corrupt files;
  - near-duplicate grouping across datasets;
  - leakage-free grouped splits;
  - the manifest fingerprint (quote it).
- **Table:** photos per dataset, split and grade (`results_summary.md`, M4 section).
- **Note the IDRiD file-name reuse** between the grading train and test sets and how it is handled (0.7.2).

## 4 · Methods
1. **Vessel and artery/vein segmentation (M5):**
   - U-Net with three outputs;
   - tiled prediction;
   - 5-fold cross-validation with folds fixed per photo, then the final model;
   - test on the DRIVE test photos against the second observer (state this protocol difference).
2. **From mask to vessel graph (M3):**
   - skeletonisation, junctions and crossings;
   - area-based sub-pixel width;
   - separate artery and vein graphs, which stop false shortcuts at crossings.
3. **Network hemodynamics (M1–M2):**
   - solver and boundary conditions (disc end; Murray-scaled terminal resistance at the other ends);
   - rheology options;
   - the ranking-invariance theorem and what it means for uncertainty analysis.
4. **Uncertainty (M2, M7):** Monte Carlo over the unknowns that actually reorder rankings:
   - outlet resistance and pressure;
   - per-segment radius error, with σ from the M5 cross-validation;
   - viscosity model;
   - pixel size.
5. **Regional risk and association analysis (M7):**
   - patch scores;
   - within-image AUROC with t-intervals over images, and why not the bootstrap (simulation table in `docs/m7_analysis_plan.md`);
   - paired baselines;
   - the pre-registration date.
6. **PINN and physics-informed GNN comparison (M8):** same networks and boundary conditions, same step budgets.
7. **Triage model (M6):**
   - preprocessing and quality checks;
   - MobileNetV3;
   - choices fixed on validation (epoch, temperature, threshold, uncertainty limit);
   - pre-registration.
8. **Prototype system (M9–M10):** API and dashboard; the disc estimate.

## 5 · Results
Each item names its source. Figures are in `reports/`.

| § | Result | Source |
|---|---|---|
| 5.1 | Solver correctness (analytical cases, conservation) and speed | `docs/hemodynamics_solver.md`, `reports/benchmarks/` |
| 5.2 | Which unknowns reorder the WSS ranking | `reports/experiments/m2_rank_sensitivity.json` |
| 5.3 | Width accuracy | `reports/experiments/m3_width_accuracy.json` |
| 5.4 | Segmentation and A/V (CV and test tables, overlays, crossings removed) | `reports/m5_full/` |
| 5.5 | Triage: internal and external results, calibration, selective prediction, Grad-CAM | `reports/m6_full/` |
| 5.6 | **Core experiment:** the primary endpoint, baselines, combined models, rank stability, contamination, controls | `reports/m7_full/`, `docs/m7_analysis_plan.md` |
| 5.7 | PINN vs solver | `reports/m8_full/` |
| 5.8 | Disc estimate accuracy | `reports/m9_disc/` |

## 6 · Discussion
- **Core experiment:** write the paragraph that matches the outcome. The verdict rule was fixed before the run.
  - *Positive and beating the baselines:* physics carries information beyond geometry and position, cross-sectionally; it cannot claim prediction.
  - *Positive but not beating the baselines:* the association is explained by where vessels are, not by the physics.
  - *No evidence:* report it as a null result, with power considerations: 27 test images and the measured rank stability.
  - *Negative:* lesions sit where shear is not high. Discuss the low-shear hypothesis (A6) and the secondary low-shear score.
- **Triage:** internal vs external performance; why a threshold set on one camera need not carry over; how useful the uncertainty and quality flags are.
- **PINN:** no gain for the forward problem; where physics-informed learning could matter (inverse problems with measurements fundus photos do not provide).
- **The seven uses:** what was and was not delivered (table in `docs/PLAN.md` §2).

## 7 · Limitations (consolidated; keep all of them)
1. **Cross-sectional data:** lesions already exist, so nothing here shows prediction of *future* lesions (PDF §11.5).
2. **Assumed boundary conditions:** no patient pressures, flows or vitals (A2, A16). Absolute stresses are illustrative; only rankings are reported.
3. **1D steady model:**
   - no curvature or secondary flow, so bends enter only as tortuosity (A4);
   - no autoregulation, no pulsatility (A10).
4. **Invisible microcirculation:** microaneurysms sit on capillaries the photo cannot resolve, so the association is at patch level only (A7).
5. **Segmentation domain shift:** the vessel model was trained on 20 DRIVE photos (one camera, Netherlands) and applied to IDRiD (another camera, India). The width error enters the physics as σ.
6. **Downscaling:** the working scale (540 px field of view) loses the smallest vessels.
7. **Estimated positions:**
   - the fovea is estimated (2.5 disc diameters from the disc);
   - in the API, the disc is estimated unless the user clicks it.
8. **Small samples:**
   - 27 IDRiD test images for the primary endpoint;
   - 103 IDRiD test photos for the external triage test;
   - wide intervals.
9. **No patient ids:** both eyes of one person may sit in different splits, or both in a test set, so the intervals are somewhat too narrow.
10. **Labels:**
    - APTOS grades come from one grader each, with known noise;
    - "referable" excludes macular oedema (APTOS has no DME label).
11. **Quality flags** are a heuristic relative to the training photos, not validated against gradability labels.
12. **Grad-CAM** is coarse and can look convincing for wrong predictions.
13. **Multiple comparisons:** only the pre-registered primary endpoints are confirmatory; everything else is exploratory.
14. **Deviations from the pre-registrations:** list them, with reasons. So far:
    - the IDRiD loader fix and the FOV-fraction flag (M6, `docs/m6_triage.md`);
    - the rule for σ fixed before the run (M7).
15. **No clinical validation:** the prototype must not be used for any clinical decision.

## 8 · Conclusion and future work
- Answer each research question in one or two sentences, from the results.
- **Phase 3 (needs a clinical partner):**
  - longitudinal photos, for prediction;
  - paired blood pressure, HbA1c and vitals;
  - fluorescein angiography or OCT-A, to check whether the stress maps relate to leakage or perfusion;
  - prospective external validation.
- **Methods:**
  - inverse problems for physics-informed learning;
  - a curvature-aware model;
  - artery/vein models trained on more datasets.

## Appendices
- **A · Reproducibility:**
  - code version;
  - the manifest fingerprint;
  - seeds;
  - the notebook steps in order;
  - hardware (Colab T4 / CPU);
  - `results_summary.json`.
- **B · Software testing:** the number of tests, mutation checks, and synthetic null and positive controls for M6 and M7.
- **C · Pre-registered plans:** `docs/m7_analysis_plan.md` and `docs/m6_triage.md`, verbatim, with the deviation list.
- **D · Prototype screenshots** (the dashboard, with its disclaimers).

## Claims to avoid (and what to say instead)
| Avoid | Say instead |
|---|---|
| "predicts where lesions will form" | "is associated (or not) with existing lesions, cross-sectionally" |
| "virtual angiogram" / "shows leakage" | "model-derived vascular stress ranking under assumed conditions" |
| "replaces CFD in milliseconds" | "a reduced-order 1D network model solves in milliseconds; it is not 3D CFD" |
| "the PINN is faster or more accurate" | the M8 numbers as they are |
| "clinically validated" / "referral decision" | "research triage ranking; no clinical validation" |
| "blood-pressure stress test" | "in the Newtonian model blood pressure scales all stresses equally; rankings cannot change (proved)" |
