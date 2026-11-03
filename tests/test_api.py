"""M9-M10: analysis service, FastAPI endpoints, error states, disc estimate and dashboard (SYNTHETIC)."""

import base64
import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from drpinn.api.app import create_app  # noqa: E402
from drpinn.api.disc import estimate_disc  # noqa: E402
from drpinn.api.service import Analyser, ServiceConfig, clean_json  # noqa: E402
from drpinn.datasets.synthetic import fake_fundus_av  # noqa: E402
from drpinn.triage.data import synthetic_graded_eye  # noqa: E402
from drpinn.triage.quality import QualityReference, quality_metrics  # noqa: E402
from drpinn.triage.report import choose_operating_point  # noqa: E402
from drpinn.triage.train import TriageConfig, predict_logits, train_triage  # noqa: E402

SEED = 7


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project folder with a tiny trained triage model and its evaluation file (no vessel model)."""
    root = tmp_path_factory.mktemp("proj")
    y = np.array([0, 2, 4, 1, 3, 2] * 4)
    x = np.stack([synthetic_graded_eye(int(g), 64, k) for k, g in enumerate(y)])
    cfg = TriageConfig(arch="tiny", pretrained=False, image_size=64, epochs=1, batch=8, lr=2e-3)
    train_triage(x[:18], y[:18], x[18:], y[18:], cfg, root / "checkpoints" / "m6_full", log=lambda *_: None)
    from drpinn.triage.train import load_triage
    model, _ = load_triage(root / "checkpoints" / "m6_full" / "best.pt")
    op = choose_operating_point(predict_logits(model, x[18:]), y[18:], 1.0)
    qref = QualityReference.fit([quality_metrics(fake_fundus_av(256, k)[0]) for k in range(6)])
    res = root / "reports" / "m6_full"
    res.mkdir(parents=True)
    (res / "m6_results.json").write_text(json.dumps({"status": "complete", "drpinn_version": "test",
                                                     "operating_point": op.to_dict(),
                                                     "quality_reference": qref.to_dict()}))
    return root


def _truth(rgb_w, fov_w):
    """Stand-in for the vessel model: the synthetic eye's own vessel/artery/vein maps at the working size."""
    _, v, _, a, w = fake_fundus_av(rgb_w.shape[0], SEED)
    return np.stack([v, a, w]).astype(np.float32)


@pytest.fixture(scope="module")
def client(project):
    return TestClient(create_app(Analyser(ServiceConfig(project=project, mc_samples=5), seg_predictor=_truth)))


def test_without_models_every_component_says_what_is_missing(tmp_path):
    a = Analyser(ServiceConfig(project=tmp_path))
    h = a.health()
    assert not h["components"]["triage"]["available"] and "step 15" in h["components"]["triage"]["detail"]
    assert not h["components"]["vessels"]["available"] and "step 12" in h["components"]["vessels"]["detail"]
    r = TestClient(create_app(a)).post("/api/analyse", files={"image": ("e.png", _png(fake_fundus_av(300, 1)[0]))})
    assert r.status_code == 200
    j = r.json()
    assert j["quality"]["flags"] == [] and not j["triage"]["available"] and not j["shear"]["available"]
    assert j["disc"]["available"] and j["disc"]["source"].startswith("estimated")


def test_full_analysis_returns_every_output(client):
    img = fake_fundus_av(540, SEED)[0]
    r = client.post("/api/analyse", files={"image": ("eye.png", _png(img), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    json.dumps(j, allow_nan=False)  # valid JSON: no NaN anywhere
    t = j["triage"]
    assert t["available"] and sum(t["grade_probabilities"]) == pytest.approx(1, abs=1e-6)
    assert t["priority"] in ("higher priority", "lower priority") and "not a medical device" in j["disclaimer"]
    cam = Image.open(io.BytesIO(base64.b64decode(t["gradcam_png"])))
    assert cam.size == (64, 64)
    s = j["shear"]
    assert s["available"] and s["patches_analysed"] > 0 and s["trees"]["artery"]["solved"] > 0
    grid = np.array(s["high_shear"], dtype=float)
    assert grid.shape == (s["rows"], s["cols"]) and np.nanmax(grid) <= 1
    assert all(0 <= p["x"] < 540 and 0 <= p["y"] < 540 for p in s["top_patches"])
    assert any("Monte Carlo" in a for a in s["assumptions"])
    assert j["vessels"]["overlay_png"] and j["vessels"]["graphs"]["artery_segments"] > 0


def test_given_disc_is_used(client):
    img = fake_fundus_av(540, SEED, return_disc=True)
    r = client.post("/api/analyse", files={"image": ("eye.png", _png(img[0]))},
                    data={"disc_x": str(img[5][1]), "disc_y": str(img[5][0]), "include_images": "false"})
    j = r.json()
    assert j["disc"]["source"].startswith("given")
    assert (j["disc"]["x"], j["disc"]["y"]) == pytest.approx((img[5][1], img[5][0]))
    assert "gradcam_png" not in j["triage"] and "overlay_png" not in j["shear"]


@pytest.mark.parametrize("payload, data, code", [
    (b"", {}, 400),
    (b"this is not an image", {}, 400),
    (_png(np.zeros((64, 64, 3), np.uint8)), {}, 422),
    (None, {"disc_x": "10"}, 422),
    (None, {"mc_samples": "1"}, 422),
    (None, {"disc_x": "9999", "disc_y": "10"}, 422),
])
def test_bad_requests_get_clear_errors(client, payload, data, code):
    body = _png(fake_fundus_av(200, 1)[0]) if payload is None else payload
    r = client.post("/api/analyse", files={"image": ("x.png", body)}, data=data)
    assert r.status_code == code and isinstance(r.json()["detail"], (str, list))


def test_upload_size_limit(tmp_path):
    c = TestClient(create_app(config=ServiceConfig(project=tmp_path, max_upload_mb=0.001)))
    r = c.post("/api/analyse", files={"image": ("x.png", _png(fake_fundus_av(200, 1)[0]))})
    assert r.status_code == 413


def test_dashboard_and_health_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Research prototype" in page.text and "api/analyse" in page.text
    h = client.get("/api/health").json()
    assert h["components"]["triage"]["available"] and h["components"]["shear"]["available"]


def test_disc_estimate_lands_on_the_synthetic_disc():
    for s in range(10):
        img, v, fov, _, _, disc = fake_fundus_av(400, s, return_disc=True)
        for prob in (None, v.astype(float)):
            (r, c), d = estimate_disc(img, fov, prob)
            assert np.hypot(r - disc[0], c - disc[1]) < 0.5 * d


def test_clean_json_removes_nan_and_numpy():
    out = clean_json({"a": np.float32(np.nan), "b": [np.int64(3), np.inf], "c": np.array([1.5, np.nan])})
    assert out == {"a": None, "b": [3, None], "c": [1.5, None]}


def test_saved_dashboard_embeds_the_result_safely(client):
    from drpinn.api.report import static_dashboard

    img = fake_fundus_av(540, SEED)[0]
    r = client.post("/api/analyse", files={"image": ("eye.png", _png(img))}, data={"mc_samples": "5"}).json()
    r["triage"]["notes"].append("</script><script>alert(1)</script>")  # hostile text must not break out
    html = static_dashboard(r, client.get("/api/health").json(), img, "eye.png")
    assert html.count("window.DRPINN_STATIC = ") == 1 and "<title>DR-PINN report · eye.png</title>" in html
    payload = html.split("window.DRPINN_STATIC = ", 1)[1].split(";</script>", 1)[0]
    assert "</script" not in payload
    data = json.loads(payload)
    assert data["result"]["shear"]["available"] and data["photo_jpg"] and data["name"] == "eye.png"

