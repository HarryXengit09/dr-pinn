# Datasets (Milestone 4)

Facts below were checked against each dataset's own page in October 2026. Anything not stated there is marked *not stated*. Re-check before you submit.

| | DRIVE | RITE | IDRiD | APTOS 2019 | Messidor-2 (optional) |
|---|---|---|---|---|---|
| **Used for** | vessel segmentation (M5) | artery/vein separation (M5) | DR grading, lesion masks (proxy validation, M7), disc and fovea positions | DR triage training (M6) | external test (M6), if grades are obtained |
| **Images** | 40 (20 train / 20 test) | 40 sets (the DRIVE images) | 516 graded (413 / 103); 81 with lesion masks; disc and fovea centres for the 516 | 3,662 with grades | 1,748 (874 exams) |
| **Labels** | expert vessel masks — **training 20 only** (test withheld); field-of-view masks | arteries red, veins blue, overlap green, uncertain white; vessel masks (test = DRIVE 2nd observer) | ICDR 0–4 + macular-oedema risk; masks MA 81, HE 80, EX 81, SE 40, OD 81 | ICDR 0–4 (`train.csv`: `id_code`, `diagnosis`) | **none in the official release**; third-party grades exist |
| **Camera / site** | field of view ≈ 540 px | as DRIVE | Kowa VX-10α, 50°, 4288 × 2848, India | Aravind Eye Hospital, India; camera *not stated* | Topcon TRC NW6, 45° (extension set), France |
| **Size** | small | small | 1.01 GB | 10.21 GB (train images 8.60 GB) | *not stated* |
| **Access** | grand-challenge account; Kaggle copy if the link is down | request form | IEEE DataPort login | Kaggle; accept competition rules | form + licence; link valid 7 days |
| **Terms** | *not stated* on the page | cite the paper | CC BY 4.0 | competition rules (no other licence stated) | research/education only; no redistribution |
| **Cite** | Staal et al., IEEE TMI 23(4):501–509, 2004 | Hu, Abràmoff & Garvin, MICCAI 2013 | Porwal et al., IEEE DataPort 2018, doi:10.21227/H25W98 | the Kaggle competition | Decencière et al. 2014; Abràmoff et al. 2013 |

Sources: [DRIVE](https://drive.grand-challenge.org/), [RITE](https://eye.medicine.uiowa.edu/rite-dataset), [IDRiD data page](https://idrid.grand-challenge.org/Data/), [IDRiD on IEEE DataPort](https://ieee-dataport.org/node/956), [APTOS file list](https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd), [IDRiD file list](https://academictorrents.com/details/3bb974ffdad31f9df9d26a63ed2aea2f1d789405), [Messidor-2](https://www.adcis.net/en/third-party/messidor2/).

## Project definitions
- **DR grade:** ICDR scale, 0 none · 1 mild · 2 moderate · 3 severe NPDR · 4 proliferative.
- **Referable DR:** grade ≥ 2. Macular oedema is not included, because APTOS has no oedema label. This must be stated wherever referable results are reported.
- **External test:** the plan is to train on APTOS and test on IDRiD-grading test. Both come from India, so this is a different camera but not a different population. Messidor-2 (France) would be stronger, but only with a documented third-party grade source; decide in M6.

## What `scripts/m4_prepare_datasets.py` guarantees
1. **Every image is decoded.** Corrupt or truncated files are excluded and listed.
2. **Labels are validated.** Grades must be 0–4. Labels without images, and images without labels, are reported.
3. **Near-duplicates are grouped across all datasets.** The signature is a band-passed green channel, field-of-view-centred. Pairs scoring ≥ 0.92 are grouped. Pairs scoring 0.80–0.92 are listed for inspection by eye. Calibration and limits are in `src/drpinn/datasets/signatures.py`.
4. **Splits are assigned per duplicate group.** A copy of an official test image goes to test everywhere. Groups with conflicting grades are excluded. APTOS is split 70/15/15 and IDRiD-grading-train 85/15, both stratified by grade. DRIVE and RITE get 5 shared CV folds.
5. **Leakage is checked.** No group may cross splits or folds; a violation stops the script.
6. **Splits are reproducible.** They are seeded, and the manifest fingerprint changes if any row, split or grade changes.
7. **IDRiD grading train and test reuse file names** (both sets start at `IDRiD_001`). Photos and label rows are matched by (set, name), and grading ids are made unique: `IDRiD_001-train`, `IDRiD_001-test`. Before 0.7.2 they were matched by name only, which paired training labels with test photos. Rebuild the manifest (step 11d) after upgrading.
8. **APTOS splits do not depend on other datasets.** Adding or fixing DRIVE, RITE or IDRiD records never moves an APTOS photo (tested). Each rebuild keeps the previous manifest as `manifest_previous.csv` and reports how many APTOS photos changed split.

## Known limits
- **No patient or eye IDs** exist in APTOS or IDRiD, so the two eyes of one person can land in different splits. This cannot be fixed with the public data; state it as a limitation.
- **The APTOS working copy is cropped and resized** to a 1024 px longest side. It is fine for grading, but not for vessel geometry; vessel geometry uses DRIVE, RITE and IDRiD at full resolution.
- **The duplicate detector doesn't target rotated, flipped or cropped copies.**
