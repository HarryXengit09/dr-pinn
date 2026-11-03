# Milestone 7 — pre-registered analysis plan

**Fixed on 31 Oct 2026, before any real IDRiD result was computed.** If anything is changed after seeing real results, the change and the reason must be stated in the thesis.

## Question
Within one eye, do retinal regions whose visible vessels the model ranks as **high wall-shear stress** contain expert-marked **red lesions** more often than other vessel-bearing regions?

This is cross-sectional proxy validation against lesions that already exist. It cannot show that the model predicts *future* lesions (PDF §11.5).

## Data
- **Images:** IDRiD segmentation set, 81 photos (54 official train, 27 official test), with expert pixel masks: microaneurysms (MA), haemorrhages (HE), hard exudates (EX), soft exudates (SE) and optic disc (OD).
- **Vessels:** from the M5 model (`checkpoints/m5_full/final.pt`), which was trained on DRIVE only. No IDRiD image or label was used to train it.

## Processing (identical for every eye)
1. **Rescale** each photo so its field of view is 540 px across, the scale of the DRIVE photos the model learned from. Lesion masks are rescaled with "any pixel" pooling.
2. **Segment:** the model gives vessel, artery and vein probabilities, thresholded at 0.5 (fixed in M5).
3. **Build graphs:** separate artery and vein graphs (M5). **No lesion information is used to build the geometry.**
4. **Run the physics** on each tree with **Monte Carlo** sampling (M2 showed these are the inputs that matter). 50 samples per tree:
   - outlet model: terminal resistance of k ~ U(2, 10) unseen Murray levels;
   - outlet pressure jitter ±3 mmHg per outlet;
   - per-segment radius error, lognormal with σ taken from the M5 cross-validation: σ = (median absolute relative radius error) / 0.674, the relation for a normal error. This treats all of the error as random, which is conservative, because a uniform bias does not change rankings (M1). If no M5 cross-validation result exists, σ = 0.10. The value used is stated with the results;
   - viscosity: Newtonian, or Fåhræus–Lindqvist with Hd ~ U(0.30, 0.50), each with probability ½;
   - assumed optic-disc diameter ~ U(1500, 2100) µm, which sets the pixel size.

   Boundary condition: in each connected piece, the end nearest the optic-disc centre is the disc end. Pieces with no other end (closed loops, single-ended fragments) cannot carry flow under these conditions; they are left out of the solve, and the share of segments solved is reported per eye (`m7_eyes.csv`).
5. **Score segments:** P(segment in its own tree's top 10 % of WSS) = **high-shear score**; P(bottom 10 %) = **low-shear score**.
6. **Score patches:** 32 × 32 px patches. A patch is analysed if its centre is in the field of view and at least one solved segment passes through it. Patch score = maximum over its segments.
7. **Label patches:** a patch is lesion-positive if any lesion pixel falls inside it. **Red lesion = MA ∪ HE.**

## Primary endpoint (the one confirmatory test)
**Mean within-image AUROC of the high-shear score for red lesions, on the 27 official TEST images**, among vessel-bearing patches.
- Only images with ≥ 1 positive and ≥ 1 negative patch count.
- 95 % t-interval over images; two-sided one-sample t-test against 0.5.
- **Interpretation rule:** CI entirely above 0.5 → positive association; CI includes 0.5 → no evidence of association; CI entirely below 0.5 → lesions sit where shear is *not* high.

**Why a t-interval rather than a bootstrap.** A simulation with lesions placed at random checked how often each interval wrongly excludes 0.5. The target is 5 %.

| Images | Bootstrap-percentile | t-interval |
|---|---|---|
| 4 | 23 % | 6.3 % |
| 10 | 10 % | 4.7 % |
| 27 | 5.3 % | 3.7 % |

## Key secondary comparisons (paired, same images; reported, not confirmatory)
High-shear score minus each of these, as a paired t-test of per-image AUROC differences:
- vessel density;
- distance to fovea (estimated);
- the deterministic high-shear score (no uncertainty);
- vessel thinness.

Beating 0.5 is not enough. Lesions also cluster by position and vessel density, so **physics only adds evidence if it beats these simple baselines.**

## Other secondary and exploratory analyses
- **Other scores:** low-shear (Monte Carlo and deterministic); tortuosity, thinness and Murray deviation.
- **Other labels:** MA, HE, exudates, any lesion.
- **Other sets:** all 81 images.
- **Pooled patch AUROC and average precision,** with image-bootstrap CIs.
- **Combined models:** logistic regression fitted on the 54 TRAIN images, scored on TEST. The models are (i) position + density, (ii) + geometry, (iii) + geometry + physics. The paired difference (iii) − (ii) asks whether physics adds information beyond geometry (RQ1).
- **Rank stability:** Spearman of each Monte Carlo sample vs the deterministic ranking, per tree.
- **Contamination:** the share of red-lesion pixels the model labelled as vessel.

No multiple-comparison correction is applied to secondary results. They are labelled exploratory, and only the primary endpoint is confirmatory.

## Known limitations, to state with the results
- **Domain shift:** the M5 model was trained on 20 DRIVE photos (Netherlands, different camera) and is applied to IDRiD (India, Kowa camera). Inspect the overlays visually.
- **Downscaling to DRIVE scale loses the smallest vessels.** Microaneurysms sit on capillaries that are never visible (plan assumption A7).
- **The fovea position is estimated** (2.5 disc diameters from the disc), not measured.
- **The physics uses assumed boundary conditions,** not measurements. Rankings depend on them, which is why they are sampled.

## Software checks done before real data (synthetic eyes; `scripts/m7_lesion_association.py --synthetic`)
- **Null control** (lesions placed at random, 8 test eyes): AUROC 0.42, CI 0.27–0.58, verdict "no evidence of association". Over 200 random placements the mean was 0.50 (SD 0.03) for every score.
- **Positive control** (lesions planted in high-shear patches): AUROC 0.91, CI 0.84–0.98, verdict "positive association". It also beat vessel density and thinness in the paired tests.
