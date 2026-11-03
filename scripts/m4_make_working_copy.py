"""Make a compact working copy of a large image folder (used for APTOS 2019).

Each image is cropped to its field of view and resized so its longest side is
at most --max-side px, then saved as high-quality JPEG with the same file
stem. CSV files next to the images are copied. Existing outputs are skipped,
so an interrupted run can be resumed. The crop changes the pixel scale; the
working copy is meant for DR-grade classification, not for vessel geometry.

    python scripts/m4_make_working_copy.py --src /content/aptos --dst data/APTOS2019 --max-side 1024
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drpinn.datasets.signatures import _crop_to_fov  # noqa: E402

_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _convert_one(job: tuple[str, str, int, int]) -> str:
    """Convert one image; returns '' on success or an error message."""
    from PIL import Image

    src, out, max_side, quality = job
    try:
        with Image.open(src) as im:
            rgb = np.asarray(im.convert("RGB"))
        crop = _crop_to_fov(rgb.astype(np.float32)).astype(np.uint8)
        img = Image.fromarray(crop)
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(out).with_suffix(".part")
        img.save(tmp, "JPEG", quality=quality)
        tmp.replace(out)  # never leave a half-written image that a resume would skip
        return ""
    except Exception as exc:  # keep going; report every failure
        return f"{type(exc).__name__}: {exc}"


def convert(src: Path, dst: Path, max_side: int = 1024, quality: int = 95, workers: int = 1,
            progress_every: int = 0) -> dict:
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in _EXT)
    jobs, skipped = [], 0
    for p in files:
        out = (dst / p.relative_to(src)).with_suffix(".jpg")
        if out.exists():
            skipped += 1
        else:
            jobs.append((str(p), str(out), max_side, quality))
    t0 = time.perf_counter()
    if progress_every:
        print(f"{len(files)} images found, {skipped} already converted, {len(jobs)} to do", flush=True)
    results = []
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_convert_one, jobs, chunksize=8):
                results.append(r)
                if progress_every and len(results) % progress_every == 0:
                    print(f"  {len(results)}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        for j in jobs:
            results.append(_convert_one(j))
            if progress_every and len(results) % progress_every == 0:
                print(f"  {len(results)}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    failed = [f"{Path(j[0]).relative_to(src)}: {r}" for j, r in zip(jobs, results) if r]
    for c in src.rglob("*.csv"):
        target = dst / c.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c, target)
    return {"converted": len(jobs) - len(failed), "skipped_existing": skipped, "failed": failed,
            "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    r = convert(Path(a.src), Path(a.dst), a.max_side, a.quality, a.workers, progress_every=200)
    print(f"converted {r['converted']}, skipped {r['skipped_existing']} existing, "
          f"{len(r['failed'])} failed, {r['seconds']:.0f} s")
    for f in r["failed"][:20]:
        print("  FAILED", f)


if __name__ == "__main__":
    main()
