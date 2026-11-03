# Data folder — setup

Datasets are **not** part of the code. Put them here (in Colab: `MyDrive/dr-pinn/data/`). Zips can stay zipped; the notebook unpacks them. Any folder layout inside each dataset folder works.

```
data/
  DRIVE/       training set zip from drive.grand-challenge.org (or a Kaggle copy)
  RITE/        zip from the RITE download-request form (artery/vein labels)
  IDRiD/       A. Segmentation.zip, B. Disease Grading.zip, C. Localization.zip (IEEE DataPort)
  APTOS2019/   made by notebook step 11b (cropped 1024 px copy); or put train.csv + train_images/ here
  splits/      written by scripts/m4_prepare_datasets.py — manifest.csv used by every later step
```

Full facts, licences and citations: `docs/datasets.md`.

**Synthetic mode (software testing only):**
`python scripts/m4_prepare_datasets.py --synthetic --data-dir /tmp/fake_data --out /tmp/fake_out` writes fake datasets with planted problems and runs every check. Never report numbers from it.
