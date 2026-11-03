# Milestone 6: DR grade triage model

> Research prototype, not a medical device. The model ranks photos for research. It does not make referral decisions and gives no follow-up interval (PLAN.md use 2).

## What it does
One photo goes in. The model returns:
- five ICDR grade probabilities (0 none … 4 proliferative), calibrated;
- **P(referable)** = P(grade ≥ 2) (PLAN.md A15; macular oedema excluded because APTOS has no DME label);
- an **uncertainty** score (normalised predictive entropy);
- **image-quality flags**;
- a **Grad-CAM** map of the regions that raised P(referable).

Code: `src/drpinn/triage/` (quality, data, model, train, metrics, report). Run it with `scripts/m6_triage.py`, or from notebook step 15.

## Pre-registered analysis plan
**Fixed on 31 Oct 2026, before any real result was computed.** Any change made after seeing results must be reported, with its reason.

### Data
Rows come from `data/splits/manifest.csv` (M4: leakage-free, duplicate-grouped).

| Set | Source | Role |
|---|---|---|
| train | APTOS 2019, split train (70 %) | fitting |
| val | APTOS 2019, split val (15 %) | epoch selection, temperature, threshold, uncertainty limit |
| internal test | APTOS 2019, split test (15 %) | **primary** grading endpoint |
| external test | IDRiD grading, official TEST (103 photos) | **primary** referral endpoint |
| external (secondary) | IDRiD grading, official TRAIN | larger external check |

IDRiD is never used for training or tuning, so both IDRiD sets are external: different clinic and camera, same country. A test photo is dropped if it shares a near-duplicate group with any training or validation photo. The number dropped is reported.

### Model and training (preset `full`)
- **Model:** MobileNetV3-Large with ImageNet weights, five outputs.
- **Input:** the field of view is cropped, padded square and resized to 512 px.
- **Augmentation:** flips, rotation by any angle, ±15 % brightness and contrast.
- **Loss:** cross-entropy, label smoothing 0.05, class weights ∝ 1/√frequency.
- **Optimiser:** AdamW, learning rate 5e-4, 1-epoch warm-up then cosine decay; batch 32; 20 epochs.
- **Model selection:** the epoch with the highest **validation** QWK.

The `quick` preset (384 px, 6 epochs) is for checking the pipeline only. Its numbers are not results.

### Operating points (all fixed on validation, before testing)
1. **Temperature scaling** (one scalar, minimising validation NLL).
2. **Referral threshold** on calibrated P(referable): the highest threshold that gives **≥ 90 % sensitivity** on validation.
3. **Uncertainty limit:** the 90th percentile of validation normalised entropy, so the 10 % most uncertain photos are marked "needs a human grader".
4. **Quality reference:** the 1st–99th percentile range of each quality metric over the training photos (except the FOV fraction; see the changes below).

### Endpoints
**Primary** (95 % percentile-bootstrap CIs over photos, 2,000 resamples):
- **P1:** referable-DR AUC on the IDRiD external test;
- **P2:** quadratic weighted kappa (QWK) on the APTOS internal test.

**Secondary:**
- sensitivity and specificity at the fixed threshold (each test set);
- accuracy and macro-F1;
- per-grade sensitivity, specificity and F1; confusion matrices;
- top-label and referable expected calibration error (ECE), NLL and Brier score, before and after temperature scaling;
- selective prediction: accuracy, QWK and referable AUC when the 10 / 20 / 30 % most uncertain photos are set aside;
- results split by quality flag (flagged vs not).

**Exploratory:** Grad-CAM for four external photos: the most confident correct referable photo, the most confident correct non-referable photo, the worst missed referable photo and the worst false alarm.

No correction for multiple comparisons is applied to secondary results; they are labelled as secondary.

## Changes after the first real run (1 Nov 2026)
The pre-registration requires every later change to be reported with its reason. Neither change touches the model, the training data, the selected epoch, the temperature or the threshold: all of these come from APTOS only.

1. **IDRiD loader bug (fixed in 0.7.2).** IDRiD's grading training and test sets reuse the same file names: both start at `IDRiD_001`. The loader matched label rows to photos by name only. As a result:
   - training labels 001–103 were attached to the *test* photos 001–103;
   - those photos then had two different grades, and the duplicate check excluded most of them;
   - training photos 001–103 were never read.

   The first run therefore reported an "external test" of 53 rows, all grade ≥ 2, so its AUC was undefined, and a 300-photo "IDRiD train" set missing most of its referable photos. **Those IDRiD numbers are invalid and are not reported.** Photos are now matched by (set, name) and get unique ids (`IDRiD_001-train`, `IDRiD_001-test`). The synthetic test data now reuses names the same way, so the bug is caught by a test.
2. **FOV-fraction quality flag removed.** Every IDRiD photo was flagged "small field of view". That flag measured how a dataset was cropped (the APTOS working copy is cropped to the field of view; IDRiD originals are not), not image quality. Only the absolute rule (field of view under 15 % of the photo) remains.

3. **External test = IDRiD's official test set.** The external test is now chosen by the official split, as pre-registered. Before, it used the manifest split, into which M4 also moves training photos that duplicate a test photo.

The APTOS internal-test results of the first run were not affected.

**Safeguards added:**
- Re-running M4 now keeps the old manifest (`manifest_previous.csv`) and reports how many APTOS photos changed split. A test checks that adding or fixing other datasets cannot move APTOS photos.
- The M6 checkpoint now stores its training photo ids. Evaluation refuses to run if any of them is now a validation or test photo. Older checkpoints are checked by set size.

## Software checks (synthetic eyes, `--synthetic`)
The synthetic eyes are fake fundus photos with planted lesions whose number grows with grade. The "external" set has a different colour tint and resolution, to imitate a camera change.

| Control | Result (tiny CNN, 250 eyes, 12 epochs) |
|---|---|
| **Positive** (lesions follow grade) | internal QWK 0.94 [0.87, 0.98]; external referable AUC 1.00 |
| **Null** (lesions unrelated to grade) | external referable AUC 0.50 [0.33, 0.66]; QWK 0 |

The positive control also shows what a camera change does. On the tinted "external" photos, grading QWK fell from 0.95 to about 0.6, and the threshold chosen on APTOS-like validation photos gave 100 % sensitivity but only 33 % specificity. Calibration got *worse* after temperature scaling, and the quality check flagged most of those photos as "low contrast" or "too dark". The real IDRiD results should be read with this in mind: a threshold chosen on one camera does not have to carry over to another.

## Limitations (to state with the results)
- **Labels:** APTOS grades come from one grader per photo, with known label noise. IDRiD grades are from experts but the set is small: 103 test photos give wide CIs.
- **Patients:** APTOS does not identify patients, so both eyes of one person may be split across train and test, or both be in test. The near-duplicate grouping catches copies of the same photo, not a person's other eye. The CIs treat photos as independent, so they are somewhat too narrow.
- **Referable definition:** grade ≥ 2 only; diabetic macular oedema is not included.
- **Quality flags** are a heuristic relative to the training photos and are not validated against gradability labels. No dataset used here has such labels; EyeQ would be needed. Brightness, contrast and sharpness also shift with the camera, so many external photos can be flagged for that reason alone.
- **Grad-CAM** is a coarse localisation aid. It can look convincing even for a wrong prediction.
- No prospective or clinical validation. The results do not support any clinical use.

## Testing
`tests/test_m6.py` has 18 tests:
- QWK and F1 against scikit-learn;
- temperature recovery;
- ECE;
- the threshold rule;
- ROC area equal to AUC;
- quality flags for blur, darkness and a missing fundus, and no flag for an uncropped frame;
- the FOV crop;
- seeded augmentation;
- a resumable cache;
- model shapes;
- Grad-CAM on a model with a known answer;
- exact training resume;
- manifest sets with duplicate removal;
- operating point and triage card;
- the two synthetic controls, and re-evaluating a finished model without retraining;
- refusing to evaluate when a training photo has moved to validation or test.

Mutation checks were run: 10 deliberate bugs were all caught. One other change was not caught: using the same shuffle every epoch, which hurts training but is not a correctness error.
