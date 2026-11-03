"""Saved dashboard: one analysed photo as a self-contained HTML page (no server, any browser, works offline).

Uses the same page as the live dashboard (static/index.html) with the result
embedded, so the two always look the same. Used by notebook step 16c, which
also shows it inside the notebook (no Colab port link needed, e.g. in Firefox).
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np

from .service import clean_json

STATIC = Path(__file__).parent / "static" / "index.html"


def _jpg_b64(rgb, side: int = 900) -> str:
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8))
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def static_dashboard(result: dict, health: dict, photo_rgb, name: str = "photo") -> str:
    """Full HTML document of the dashboard for one result."""
    data = {"result": clean_json(result), "health": clean_json(health), "photo_jpg": _jpg_b64(photo_rgb),
            "name": name, "when": dt.datetime.now().strftime("%d %b %Y %H:%M")}
    payload = json.dumps(data).replace("</", "<\\/")  # never close the script tag early
    page = STATIC.read_text(encoding="utf-8")
    marker = "<script>\nconst $ = "
    if marker not in page:
        raise RuntimeError("dashboard page layout changed: cannot embed the result")
    page = page.replace(marker, f"<script>window.DRPINN_STATIC = {payload};</script>\n{marker}", 1)
    return page.replace("<title>DR-PINN Research Dashboard</title>", f"<title>DR-PINN report · {name}</title>", 1)


def save_static_dashboard(path: str | Path, result: dict, health: dict, photo_rgb, name: str = "photo") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(static_dashboard(result, health, photo_rgb, name), encoding="utf-8")
    return path
