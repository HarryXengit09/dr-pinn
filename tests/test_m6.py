"""M6: triage metrics, quality checks, data pipeline, model, Grad-CAM, resume, and null/positive controls (SYNTHETIC)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from drpinn.datasets.synthetic import fake_fundus
from drpinn.triage.data import augment, build_cache, load_cached, read_working, square_fundus, synthetic_graded_eye
from drpinn.triage.metrics import (
    bootstrap_ci, expected_calibration_error, fit_temperature, macro_f1, nll, per_class, quadratic_weighted_kappa,
    referable_log_odds, referable_probability, roc_curve, softmax, threshold_for_sensitivity,
)
from drpinn.triage.model import build_model, grad_cam, referable_score
from drpinn.triage.quality import NO_FUNDUS, QualityReference, quality_metrics
from drpinn.triage.report import OperatingPoint, choose_operating_point, triage_card
from drpinn.triage.train import TriageConfig, class_weights, load_triage, train_triage
from drpinn.segmentation.metrics import roc_auc

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_triage.py"


# --- metrics --------------------------------------------------------------------------

def test_qwk_and_f1_match_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    pred = np.clip(y + rng.integers(-1, 2, 300), 0, 4)
    assert quadratic_weighted_kappa(y, pred) == pytest.approx(skm.cohen_kappa_score(y, pred, weights="quadratic"))
    assert macro_f1(y, pred) == pytest.approx(skm.f1_score(y, pred, average="macro"))
    se = skm.recall_score(y, pred, average=None)
    assert [c["sensitivity"] for c in per_class(y, pred)] == pytest.approx(se)
    assert quadratic_weighted_kappa(y, y) == 1.0
    assert quadratic_weighted_kappa([0, 0, 4, 4], [4, 4, 0, 0]) < 0


def test_temperature_recovers_known_scaling():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 2, (4000, 5))
    p = softmax(true_logits)
    y = np.array([rng.choice(5, p=q) for q in p])
    assert fit_temperature(3.0 * true_logits, y) == pytest.approx(3.0, rel=0.1)  # over-confident model -> T ~ 3
    np.testing.assert_allclose(softmax(3.0 * true_logits, 3.0), p)
    assert nll(3.0 * true_logits, y, 3.0) == pytest.approx(nll(true_logits, y))


def test_ece_is_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 20000)
    outcome = rng.uniform(0, 1, conf.size) < conf
    assert expected_calibration_error(conf, outcome) < 0.02
    assert expected_calibration_error(np.full(1000, 0.95), np.arange(1000) < 600) == pytest.approx(0.35)


def test_threshold_gives_at_least_the_target_sensitivity():
    s = np.r_[np.linspace(1, 0.1, 10), np.linspace(0.95, 0.05, 30)]
    y = np.r_[np.ones(10, bool), np.zeros(30, bool)]
    t = threshold_for_sensitivity(s, y, 0.9)
    assert t == pytest.approx(0.2) and (s[y] >= t).mean() >= 0.9
    assert (s[y] >= t + 1e-9).mean() < 0.9  # the highest threshold that does it
    rng = np.random.default_rng(5)
    for n_pos in range(3, 16):
        sc, yy = rng.random(n_pos + 20), np.r_[np.ones(n_pos, bool), np.zeros(20, bool)]
        t = threshold_for_sensitivity(sc, yy, 0.9)
        assert (sc[yy] >= t).mean() >= 0.9 and (sc[yy] > t).mean() < 0.9


def test_roc_curve_area_equals_auc_and_referable_helpers():
    rng = np.random.default_rng(3)
    y = rng.random(200) < 0.3
    s = y + rng.normal(0, 1, 200)
    s[:20] = np.round(s[:20])  # some ties
    fpr, tpr = roc_curve(s, y)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(tpr, fpr) == pytest.approx(roc_auc(s, y))
    logits = rng.normal(size=(50, 5))
    p = softmax(logits)
    np.testing.assert_allclose(referable_probability(p), p[:, 2:].sum(1))
    lo = referable_log_odds(logits)
    np.testing.assert_allclose(1 / (1 + np.exp(-lo)), referable_probability(p), rtol=1e-10)
    np.testing.assert_allclose(referable_score(torch.tensor(logits)).numpy(), lo, rtol=1e-6)
    est, a, b = bootstrap_ci(lambda i: float(s[i].mean()), s.size, 300)
    assert a < est < b


# --- quality --------------------------------------------------------------------------

def test_quality_flags_blur_darkness_and_missing_fundus():
    imgs = [fake_fundus(256, k)[0] for k in range(8)]
    ref = QualityReference.fit([quality_metrics(i) for i in imgs])
    assert ref.flags(quality_metrics(imgs[0])) == []
    blurred = ndimage.gaussian_filter(imgs[1].astype(float), (3, 3, 0)).astype(np.uint8)
    assert "blurry" in ref.flags(quality_metrics(blurred))
    dark = (imgs[2] * 0.35).astype(np.uint8)
    assert "too dark" in ref.flags(quality_metrics(dark))
    assert ref.flags(quality_metrics(np.zeros((200, 200, 3), np.uint8))) == [NO_FUNDUS]
    small = np.zeros((600, 600, 3), np.uint8)
    small[250:350, 250:350] = imgs[3][100:200, 100:200]
    assert ref.flags(quality_metrics(small)) == [NO_FUNDUS]
    assert QualityReference.from_dict(json.loads(json.dumps(ref.to_dict()))) == ref
    padded = np.zeros((400, 460, 3), np.uint8)  # same photo, uncropped camera frame (as IDRiD vs APTOS copy)
    padded[72:328, 102:358] = imgs[0]
    assert quality_metrics(padded)["fov_fraction"] < 0.5 * quality_metrics(imgs[0])["fov_fraction"]
    assert ref.flags(quality_metrics(padded)) == []


# --- data -----------------------------------------------------------------------------

def test_square_fundus_crops_pads_and_resizes():
    img = np.zeros((300, 500, 3), np.uint8)
    yy, xx = np.mgrid[:300, :500]
    img[(yy - 150) ** 2 + (xx - 250) ** 2 <= 140 ** 2] = (180, 80, 40)
    sq = square_fundus(img, 128)
    assert sq.shape == (128, 128, 3)
    assert sq[64, 64, 0] > 150 and sq[2, 2].sum() == 0  # FOV centred, corners black


def test_augment_is_seeded_keeps_background_black_and_shape():
    x = np.stack([synthetic_graded_eye(2, 96, k) for k in range(4)])
    a = augment(x, torch.Generator().manual_seed(0))
    b = augment(x, torch.Generator().manual_seed(0))
    assert a.shape == (4, 3, 96, 96) and torch.equal(a, b)
    black = (torch.zeros(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(a[0, :, 0, 0], black)
    c = augment(x, torch.Generator().manual_seed(1))
    assert not torch.equal(a, c)


def test_cache_resumes_and_returns_quality(tmp_path):
    from PIL import Image

    paths = []
    for k in range(3):
        p = tmp_path / "src" / f"im{k}.png"
        p.parent.mkdir(exist_ok=True)
        Image.fromarray(fake_fundus(160, k)[0]).save(p)
        paths.append((f"DS_{k}", p))
    paths.append(("DS_bad", tmp_path / "src" / "missing.png"))
    q, err = build_cache(paths, 64, tmp_path / "cache")
    assert set(q) == {"DS_0", "DS_1", "DS_2"} and set(err) == {"DS_bad"}
    stamp = (tmp_path / "cache" / "DS_0.png").stat().st_mtime_ns
    q2, _ = build_cache(paths[:3], 64, tmp_path / "cache")
    assert q2 == q and (tmp_path / "cache" / "DS_0.png").stat().st_mtime_ns == stamp
    assert load_cached(["DS_0", "DS_2"], tmp_path / "cache").shape == (2, 64, 64, 3)
    big = np.asarray(Image.fromarray(fake_fundus(300, 5)[0]).resize((2400, 1600)))
    Image.fromarray(big).save(tmp_path / "big.jpg", quality=95)
    assert read_working(tmp_path / "big.jpg").shape == (683, 1024, 3)  # decoded small, never below 1024 px


# --- model ----------------------------------------------------------------------------

def test_models_build_with_five_outputs():
    x = torch.zeros(2, 3, 96, 96)
    for arch in ("mobilenet_v3_large", "tiny"):
        m = build_model(arch, 5, pretrained=False).eval()
        assert m(x).shape == (2, 5)
    with pytest.raises(ValueError):
        build_model("vgg")


class _SpotModel(torch.nn.Module):
    """features = red channel (ReLU); output 2 (referable side) = its mean: Grad-CAM must equal the red map."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 1, 1, bias=False), torch.nn.ReLU())
        with torch.no_grad():
            self.features[0].weight.copy_(torch.tensor([1.0, 0, 0]).view(1, 3, 1, 1))

    def forward(self, x):
        f = self.features(x).mean((2, 3))
        return torch.cat([torch.zeros_like(f), torch.zeros_like(f), f, torch.zeros_like(f), torch.zeros_like(f)], 1)


def test_grad_cam_highlights_the_evidence():
    x = torch.zeros(1, 3, 32, 32)
    x[0, 0, 20:26, 4:10] = 1.0
    cam = grad_cam(_SpotModel(), x, 2)
    assert cam.shape == (1, 32, 32) and cam.max() == pytest.approx(1.0)
    assert cam[0, 22, 6] > 0.9 and cam[0, 5, 25] < 0.1
    assert grad_cam(_SpotModel(), x, "referable")[0, 22, 6] > 0.9


# --- training -------------------------------------------------------------------------

def _tiny_data(n=48, size=64):
    y = np.array([0, 2, 4] * (n // 3))
    x = np.stack([synthetic_graded_eye(int(g), size, k) for k, g in enumerate(y)])
    return x, y


def test_training_resumes_exactly_and_refuses_other_settings(tmp_path):
    x, y = _tiny_data()
    cfg = TriageConfig(arch="tiny", pretrained=False, image_size=64, epochs=2, batch=16, lr=2e-3)
    full = train_triage(x[:36], y[:36], x[36:], y[36:], cfg, tmp_path / "a", log=lambda *_: None)
    part = train_triage(x[:36], y[:36], x[36:], y[36:], cfg, tmp_path / "b", stop_after_epochs=1, log=lambda *_: None)
    assert not part["complete"] and part["epoch"] == 1
    part = train_triage(x[:36], y[:36], x[36:], y[36:], cfg, tmp_path / "b", log=lambda *_: None)
    assert part["complete"]
    assert [h["val_loss"] for h in part["history"]] == pytest.approx([h["val_loss"] for h in full["history"]], rel=1e-5)
    model, ck = load_triage(tmp_path / "b" / "best.pt")
    assert ck["epoch"] == part["best_epoch"]
    with pytest.raises(ValueError):
        train_triage(x[:36], y[:36], x[36:], y[36:], TriageConfig(arch="tiny", pretrained=False, epochs=3),
                     tmp_path / "b", log=lambda *_: None)


def test_class_weights_favour_rare_grades_and_average_one():
    y = np.array([0] * 80 + [1] * 5 + [2] * 15)
    w = class_weights(y, "sqrt")
    assert w[1] > w[2] > w[0] and w[3] == 0 and (w[y]).mean() == pytest.approx(1.0)
    assert np.all(class_weights(y, "none") == 1)


def _script():
    spec = importlib.util.spec_from_file_location("m6", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_sets_keep_idrid_external_and_drop_duplicates_of_training(tmp_path):
    import csv

    (tmp_path / "splits").mkdir()
    rows = [("APTOS2019", "grading", "a1", "train", "2", "1", "0"), ("APTOS2019", "grading", "a2", "val", "0", "2", "0"),
            ("APTOS2019", "grading", "a3", "test", "4", "3", "0"), ("APTOS2019", "grading", "a4", "test", "1", "1", "0"),
            ("IDRiD", "grading", "i1", "test", "3", "5", "0"), ("IDRiD", "grading", "i2", "train", "0", "6", "0"),
            ("IDRiD", "grading", "i3", "val", "2", "2", "0"), ("IDRiD", "segmentation", "s1", "test", "", "7", "0"),
            ("IDRiD", "grading", "i4-train", "test", "1", "7", "0"),  # moved to test as a duplicate: still not official test
            ("APTOS2019", "grading", "a5", "train", "3", "8", "1"), ("DRIVE", "vessels", "21", "cv", "", "9", "0")]
    with open(tmp_path / "splits" / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "part", "image_id", "rel_path", "official_split", "split", "dr_grade", "dup_group",
                    "label_conflict"])
        for ds, part, iid, split, g, grp, conf in rows:
            official = "train" if iid.endswith("-train") else (split if split == "test" else "train")
            w.writerow([ds, part, iid, f"{ds}/{iid}.jpg", official if ds == "IDRiD" else "", split, g, grp, conf])
    m = _script()
    sets, info = m.make_sets(m.read_manifest(tmp_path))
    ids = {k: [r["image_id"] for r in v] for k, v in sets.items()}
    assert ids == {"train": ["a1"], "val": ["a2"], "test_internal": ["a3"], "external_idrid_test": ["i1"],
                   "external_idrid_train": ["i2", "i4-train"]}  # a4 and i3 duplicate a training/validation photo
    assert info["dropped_as_duplicates_of_training"] == {"test_internal": 1, "external_idrid_test": 0,
                                                         "external_idrid_train": 1}
    assert sets["train"][0]["path"] == tmp_path / "APTOS2019" / "a1.jpg"


def test_operating_point_and_triage_card():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 5, 400)
    logits = np.eye(5)[y] * 2 + rng.normal(0, 1, (400, 5))
    op = choose_operating_point(logits, y, 1.0, 0.9, 0.1)
    p = softmax(logits)
    assert (referable_probability(p)[y >= 2] >= op.referral_threshold).mean() >= 0.9
    card = triage_card(logits[0], op, [])
    assert card["priority"] in ("higher priority", "lower priority") and "not a medical device" in card["disclaimer"]
    assert triage_card(logits[0], op, [NO_FUNDUS])["priority"].startswith("not assessed")
    op2 = OperatingPoint(1.0, 0.5, 0.9, 0.5, 0.1)
    unsure = triage_card(np.array([1.0, 0.5, 0, 0, 0]), op2, ["blurry"])  # normalised entropy ~0.9
    assert any("uncertain" in n for n in unsure["notes"]) and any("blurry" in n for n in unsure["notes"])
    sure = triage_card(np.array([9.0, 0, 0, 0, 0]), op2, [])
    assert sure["notes"] == [] and sure["priority"] == "lower priority"


# --- controls (the whole script) --------------------------------------------------------

def _run(scenario, tmp_path):
    from dataclasses import replace

    mod = _script()
    return mod.run(tmp_path, cfg=replace(mod.PRESETS["tiny"], epochs=12), preset="tiny", device="cpu",
                   synthetic=True, scenario=scenario, n_synthetic=250, n_boot=200)


def test_positive_control_is_detected(tmp_path, capsys):
    r = _run("planted", tmp_path)
    ext, internal = r["sets"]["external_idrid_test"], r["sets"]["test_internal"]
    assert ext["referable"]["auc"]["ci"][0] > 0.8 and internal["qwk"]["value"] > 0.6
    assert r["source"].startswith("SYNTHETIC")
    for f in ("m6_results.json", "m6_predictions.csv", "m6_confusion.png", "m6_gradcam.png"):
        assert (tmp_path / f).exists()
    # second run with a finished model: evaluation only, training photos not re-read, same results
    capsys.readouterr()
    again = _run("planted", tmp_path)
    out = capsys.readouterr().out
    assert "evaluating only" in out and "training photos not needed" in out
    assert again["sets"]["test_internal"]["qwk"] == internal["qwk"]


def test_evaluation_refuses_when_training_photos_moved_to_test():
    m = _script()
    sets = {"train": [{"dataset": "APTOS2019", "image_id": "a"}], "val": [{"dataset": "APTOS2019", "image_id": "b"}],
            "test_internal": [{"dataset": "APTOS2019", "image_id": "c"}]}
    m._check_no_leak({"train_keys": ["APTOS2019_a"]}, sets, Path("ck"))  # fine
    with pytest.raises(RuntimeError, match="trained on are now"):
        m._check_no_leak({"train_keys": ["APTOS2019_a", "APTOS2019_c"]}, sets, Path("ck"))
    m._check_no_leak({"sets": {"train": 1, "val": 1}}, sets, Path("ck"))  # old checkpoint: sizes match
    with pytest.raises(RuntimeError, match="split changed"):
        m._check_no_leak({"sets": {"train": 2, "val": 1}}, sets, Path("ck"))


def test_null_control_finds_nothing(tmp_path):
    r = _run("null", tmp_path)
    auc = r["sets"]["external_idrid_test"]["referable"]["auc"]
    assert auc["ci"][0] < 0.5 < auc["ci"][1]
