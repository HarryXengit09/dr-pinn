# Milestones 9–10: API backend and dashboard

> Research prototype, not a medical device. Every response carries this disclaimer, and the dashboard shows it at the top of the page.

## What it does
One fundus photo goes in. Back comes everything the pipeline can produce for it, with each output's assumptions:

| Component | Needs | Output |
|---|---|---|
| **quality** | nothing | metrics and flags (relative to the APTOS training photos once M6 results exist) |
| **triage** | `checkpoints/m6_full/best.pt` and `reports/m6_full/m6_results.json` (step 15) | calibrated grade probabilities, P(referable) vs the validation threshold, uncertainty, research priority, notes, Grad-CAM |
| **vessels** | `checkpoints/m5_full/final.pt` (step 12) | vessel, artery and vein overlay; artery, vein and combined graph summary |
| **shear** | the vessel model trained with artery/vein labels (RITE) | Monte Carlo wall-shear-stress ranking per tree, as a 32 px patch map; top patches; rank stability; the assumptions used |
| **disc** | — | the optic disc used: as given by the user, or estimated (`api/disc.py`) |

A missing component is reported with the reason and the notebook step that makes it, and the rest still runs. Nothing is faked or replaced by a default model.

The photo is processed exactly as in the milestones:
- triage on the ≤ 1024 px copy cropped to the field of view (M6);
- vessels and physics at the working scale where the field of view is 540 px across (M5, M7);
- the radius error σ comes from the M5 cross-validation, by the same rule as M7.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the dashboard (one self-contained HTML page) |
| GET | `/api/health` | version, device, which components are available and why not, radius σ and its source |
| POST | `/api/analyse` | multipart form: `image` (required); `disc_x`, `disc_y` (pixels in the uploaded photo, both or neither); `disc_diameter` (px, optional); `mc_samples` (5–200, default 50); `include_images` (default true) |

**Errors** (JSON `{"detail": "..."}`):
- 400: empty or unreadable file;
- 413: file larger than the limit (25 MB);
- 422: photo under 128 px a side, only one disc coordinate, disc outside the photo, or `mc_samples` out of range.

Every number in a response is valid JSON (NaN becomes `null`).

```bash
python scripts/m9_serve.py --project .          # then open http://127.0.0.1:8000
curl -F image=@photo.jpg -F disc_x=812 -F disc_y=640 http://127.0.0.1:8000/api/analyse > report.json
```
On Colab, notebook step 16 starts the server in the background and shows the dashboard (link and embedded view). Firefox often blocks Colab's port links, so step 16c runs the same analysis without a server and shows the same dashboard as a saved, self-contained HTML page (`api/report.py`). That page also opens offline in any browser.

## The optic disc
The physics needs the disc: it is the inlet of every tree and sets the pixel size. M7 used IDRiD's expert disc masks; an uploaded photo has none. The dashboard therefore asks the user to **click the disc**. Without a click, it is estimated:
- **position:** the brightest region, combined with the most vessel-dense region when the vessel model is present;
- **diameter:** 0.16 × the field-of-view diameter, the median measured on the 81 IDRiD photos (0.161). Until 0.8.1 the textbook value 0.14 was used.

**Measured on IDRiD (step 16b, 1 Nov 2026):** the estimated centre fell inside the expert disc in **99 %** of the 81 photos using brightness alone, and in **98 %** with the vessel map. The median error was 0.13 and 0.06 disc diameters respectively.

`scripts/m9_check_disc.py` (notebook step 16b) measures on the 81 IDRiD photos:
- how often the estimate lands inside the expert disc;
- the true disc-to-field-of-view ratio.

Report those numbers wherever the API is described.

## Dashboard (M10)
One page served by the backend: no build step and no external scripts, so it also runs inside Colab. It is bound to the real `/api/analyse` response and has ten sections:

1. system status;
2. photo upload with click-to-place disc;
3. image quality;
4. DR triage;
5. Grad-CAM;
6. vessels and artery/vein graphs;
7. shear-risk map with top patches and rank stability;
8. the assumptions behind the map;
9. how to read each output and its limits;
10. JSON report download and timings.

**Deviation from the plan:** the plan named a Next.js dashboard. A Node build cannot run in Colab, and a second codebase would add nothing the single page lacks, so the dashboard is plain HTML and JavaScript served by FastAPI.

## Tests
`tests/test_api.py`, 13 tests:
- no-model behaviour;
- a full analysis with a stand-in vessel model: every output present, valid JSON, sizes;
- a given disc is used;
- six bad-request cases;
- the upload size limit;
- the dashboard and health endpoints;
- the disc estimate on synthetic eyes;
- JSON cleaning.

`tests/test_m11.py` covers the disc check script.

Mutation checks: 6 deliberate bugs, all caught:
- ignoring the given disc;
- dropping the JSON cleaning;
- a wrong size limit;
- accepting half a disc position;
- an inverted disc score;
- keeping invalid IDRiD results in the summary.

## Limits
- Single-user prototype: requests are processed one at a time, there is no authentication, and nothing is stored.
- On CPU an analysis takes roughly 10–60 s, mostly the Monte Carlo physics; lower `mc_samples` for a quicker look.
- The triage threshold was set on APTOS validation photos. On other cameras its sensitivity can differ (see the M6 external results).
