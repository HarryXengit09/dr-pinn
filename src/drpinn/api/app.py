"""FastAPI app: GET / (dashboard), GET /api/health, POST /api/analyse.

Run:  python scripts/m9_serve.py --project .      (or notebook step 16 on Colab)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .. import DISCLAIMER, __version__
from .service import Analyser, ServiceConfig

STATIC = Path(__file__).parent / "static"
MIN_SIDE = 128


def read_upload(data: bytes, max_bytes: int) -> np.ndarray:
    """Decoded RGB photo, or an HTTPException with a message a user can act on."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(413, f"The file is larger than {max_bytes / 1e6:.0f} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"Could not read the image ({type(exc).__name__}). Use a JPEG, PNG or TIFF fundus "
                                 "photo; the file may be damaged.") from None
    if min(rgb.shape[:2]) < MIN_SIDE:
        raise HTTPException(422, f"The image is {rgb.shape[1]} x {rgb.shape[0]} px; at least {MIN_SIDE} px per side "
                                 "is needed.")
    return rgb


def create_app(analyser: Analyser | None = None, config: ServiceConfig | None = None) -> FastAPI:
    app = FastAPI(title="DR-PINN research prototype", version=__version__, description=DISCLAIMER)
    config = config or (analyser.cfg if analyser else ServiceConfig())
    holder = {"analyser": analyser}

    def get() -> Analyser:
        if holder["analyser"] is None:  # load models on first use, so the app starts instantly
            holder["analyser"] = Analyser(config)
        return holder["analyser"]

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return get().health()

    @app.post("/api/analyse")
    def analyse(image: UploadFile = File(...), disc_x: float | None = Form(None), disc_y: float | None = Form(None),
                disc_diameter: float | None = Form(None), mc_samples: int | None = Form(None),
                include_images: bool = Form(True)) -> JSONResponse:
        max_bytes = int(config.max_upload_mb * 1e6)
        rgb = read_upload(image.file.read(max_bytes + 1), max_bytes)
        if (disc_x is None) != (disc_y is None):
            raise HTTPException(422, "Give both disc_x and disc_y, or neither.")
        disc = None
        if disc_x is not None:
            h, w = rgb.shape[:2]
            if not (0 <= disc_x < w and 0 <= disc_y < h):
                raise HTTPException(422, f"The disc position must lie inside the {w} x {h} px photo.")
            if disc_diameter is not None and not 0 < disc_diameter < max(h, w):
                raise HTTPException(422, "disc_diameter must be a positive number of pixels.")
            disc = (disc_x, disc_y, disc_diameter)
        if mc_samples is not None and not 5 <= mc_samples <= 200:
            raise HTTPException(422, "mc_samples must be between 5 and 200.")
        return JSONResponse(get().analyse(rgb, disc, mc_samples, include_images))

    return app
