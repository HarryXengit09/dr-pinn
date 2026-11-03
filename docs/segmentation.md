# Vessel and artery/vein segmentation (Milestone 5)

## Model
- **Network:** U-Net (Ronneberger et al., MICCAI 2015). It has 4 down-sampling levels and starts with 32 channels, doubling at each level (7.8 M parameters in the `full` preset). Each convolution is followed by batch normalisation and ReLU.
- **Inputs (4 channels):** R, G and B, each standardised inside the field of view, plus a CLAHE-equalised green channel. Everything outside the field of view is 0.
- **Outputs (3 sigmoid channels):** vessel, artery, vein. Artery and vein are separate yes/no outputs rather than one choice between them. A crossing pixel can therefore be both, so each tree stays continuous through crossings.
- **Loss:**
  - Vessel channel: binary cross-entropy plus soft Dice, inside the field of view.
  - Artery and vein channels: binary cross-entropy only on pixels RITE labels as artery, vein or overlap. Background and "uncertain" pixels are ignored.
- **Training:**
  - Patches are 96 × 96, batch 16, half of them centred on a vessel.
  - Augmentation: random 90° rotations, flips and intensity jitter.
  - Optimiser: AdamW at learning rate 1e-3, with a 100-step warm-up and cosine decay, for 4,000 steps. Mixed precision is used on the GPU.
  - Checkpoints save every 1,000 steps and training resumes after a disconnect.
- **Prediction:** 256 × 256 tiles with 50 % overlap, blended with a smooth window. A test shows this matches whole-image prediction exactly for a per-pixel model.

## Evaluation protocol
| | Photos | Vessel reference | A/V reference |
|---|---|---|---|
| Cross-validation | DRIVE 21–40, 5 folds of 4 (fold fixed per photo) | DRIVE 1st observer | RITE (training) |
| Test | DRIVE 1–20 (needs RITE) | RITE test = DRIVE 2nd observer | RITE (test) |

- **Pixel metrics** are computed inside the field of view at threshold 0.5, chosen in advance and never tuned on test photos: Dice, IoU, sensitivity, specificity, accuracy and AUC.
- **Graph metrics:**
  - Centreline recall and precision: the model's graph is compared with the expert-mask graph, within 2 px.
  - Median relative radius error: on segments matched between the two graphs.
- **A/V metrics:** computed on the expert's vessel centreline, skipping overlap and uncertain pixels. A pixel is called artery when p(artery) > p(vein). Accuracy is reported, along with sensitivity (arteries) and specificity (veins).
- **Artery/vein graphs:** crossings and connected pieces are counted in the combined, artery and vein graphs.

**Protocol note:** DRIVE's official test masks (1st observer) are withheld today. The test row uses the 2nd observer's masks, so it is not directly comparable with papers that report against the 1st observer.

## Verified so far (SYNTHETIC data only — real numbers come from the Colab run)
- **Separation:** on 12 synthetic images, artery/vein separation using the true masks cut crossing flags from 38 (combined graph) to 0 (artery graphs) and 5 (vein graphs). Every tree became a single connected piece; in 7 of the 12 combined graphs the two trees had been fused.
- **Training length:** a tiny model trained for 300 steps drew vessels about 50 % too wide. At 1,500 steps the error was 1.5–1.8 %, whichever patch-sampling setting was used. So under-training, not the sampling, caused the over-width.
- **End-to-end tiny CPU run:** Dice > 0.75, AUC > 0.97, A/V accuracy > 0.9, centreline recall > 0.9. Synthetic images are much easier than real ones.

## Limits
- **20 annotated photos is very few.** Expect fold-to-fold spread, and report mean ± SD.
- **DRIVE is low-resolution** (565 × 584). Transfer to IDRiD's 4288 × 2848 photos is unverified; audit it visually in M7.
- **The width reference is itself an estimate:** the area method applied to the expert mask.
