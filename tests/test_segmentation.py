"""M5: segmentation pipeline on SYNTHETIC data (CPU). Skipped where PyTorch is not installed."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from drpinn.datasets.splits import drive_fold
from drpinn.datasets.synthetic import fake_fundus_av, make_fake_drive_rite
from drpinn.segmentation.av import av_graphs, masks_from_probabilities
from drpinn.segmentation.metrics import av_metrics, roc_auc, segmentation_metrics
from drpinn.segmentation.preprocess import estimate_fov, preprocess

torch = pytest.importorskip("torch")

from drpinn.segmentation.data import SegSample, load_drive_rite  # noqa: E402
from drpinn.segmentation.infer import predict_image  # noqa: E402
from drpinn.segmentation.train import TrainConfig, load_checkpoint, loss_fn, train  # noqa: E402
from drpinn.segmentation.unet import UNet, UNetConfig  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m5_train_segmentation.py"
TINY = TrainConfig(iterations=60, batch=4, patch=64, save_every=20, unet=UNetConfig(base=4, depth=3, dropout=0.0))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5data")
    make_fake_drive_rite(root, n_train=20, n_test=3, size=128)
    return root


@pytest.fixture(scope="module")
def samples(fake):
    return load_drive_rite(fake / "DRIVE", fake / "RITE")


# --- numpy parts --------------------------------------------------------------------

def test_preprocess_standardises_inside_fov_and_zeroes_outside():
    img, _, fov, _, _ = fake_fundus_av(128, 3)
    x = preprocess(img, fov)
    assert x.shape == (4, 128, 128) and x.dtype == np.float32
    assert np.all(x[:, ~fov] == 0)
    np.testing.assert_allclose(x[:, fov].mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(x[:, fov].std(axis=1), 1, atol=1e-3)
    with pytest.raises(ValueError):
        preprocess(img[..., 0])


def test_fov_estimate_matches_true_aperture():
    img, _, fov, _, _ = fake_fundus_av(192, 4)
    est = estimate_fov(img)
    assert (est & fov).sum() / (est | fov).sum() > 0.97


def test_segmentation_metrics_hand_example():
    ref = np.array([[1, 1, 0, 0]], bool)
    prob = np.array([[0.9, 0.2, 0.7, 0.1]])
    m = segmentation_metrics(prob, ref, np.ones_like(ref))
    # tp=1 fp=1 fn=1 tn=1
    assert m["dice"] == pytest.approx(0.5) and m["iou"] == pytest.approx(1 / 3)
    assert m["sensitivity"] == pytest.approx(0.5) and m["specificity"] == pytest.approx(0.5)
    assert m["auc"] == pytest.approx(0.75)  # pairs: (0.9>0.7),(0.9>0.1),(0.2<0.7),(0.2>0.1) -> 3/4
    fov = np.array([[1, 1, 0, 1]], bool)  # pixel outside FOV is ignored
    assert segmentation_metrics(prob, ref, fov)["specificity"] == pytest.approx(1.0)


def test_auc_extremes():
    y = np.array([0, 0, 1, 1], bool)
    assert roc_auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert np.isnan(roc_auc([0.5, 0.5], np.array([1, 1], bool)))


def test_av_metrics_on_centreline():
    vessel = np.zeros((9, 20), bool)
    vessel[3:6, 1:19] = True                       # one horizontal vessel
    labels = np.zeros((9, 20), np.uint8)
    labels[3:6, 1:10] = 1                          # left half artery
    labels[3:6, 10:19] = 2                         # right half vein
    p_art = np.where(np.arange(20) < 10, 0.9, 0.1)[None].repeat(9, 0)
    p_art[[3, 5], 3:15] = 1 - p_art[[3, 5], 3:15]    # wrong only at the vessel edges: must not count
    m = av_metrics(p_art, 1 - p_art, labels, vessel)
    assert m["av_accuracy"] == 1.0 and 10 < m["av_centreline_pixels"] <= 18
    m2 = av_metrics(1 - p_art, p_art, labels, vessel)
    assert m2["av_accuracy"] == 0.0


def test_targets_handle_overlap_and_uncertain():
    lab = np.array([[0, 1, 2, 3, 4]], np.uint8)
    s = SegSample("21", "train", np.zeros((4, 1, 5), np.float32), np.zeros((1, 5, 3), np.uint8),
                  np.ones((1, 5), bool), lab > 0, "x", lab)
    y, w = s.targets()
    np.testing.assert_array_equal(y[1, 0], [0, 1, 0, 1, 0])   # artery: 1 and overlap
    np.testing.assert_array_equal(y[2, 0], [0, 0, 1, 1, 0])   # vein: 2 and overlap
    np.testing.assert_array_equal(w[1, 0], [0, 1, 1, 1, 0])   # background and uncertain not scored


def test_av_separation_removes_crossings_on_ground_truth():
    combined = per_tree = 0
    for seed in range(6):
        _, vessel, fov, art, vein = fake_fundus_av(320, seed)
        s = av_graphs(vessel, art, vein, fov=fov).summary()
        combined += s["combined_crossings_split"] + s["combined_possible_crossings"]
        per_tree += sum(s[f"{t}_crossings_split"] + s[f"{t}_possible_crossings"] for t in ("artery", "vein"))
        assert s["artery_components"] == 1 and s["vein_components"] == 1
    assert combined >= 10 and per_tree <= combined // 4


def test_masks_from_probabilities():
    prob = np.array([[[0.9, 0.9, 0.2]], [[0.8, 0.1, 0.9]], [[0.7, 0.2, 0.9]]])
    v, a, w = masks_from_probabilities(prob)
    np.testing.assert_array_equal(v[0], [1, 1, 0])
    np.testing.assert_array_equal(a[0], [1, 0, 0])  # artery only where a vessel is
    np.testing.assert_array_equal(w[0], [1, 0, 0])


def test_drive_fold_is_balanced_and_stable():
    folds = [drive_fold(i) for i in range(21, 41)]
    assert sorted(np.bincount(folds).tolist()) == [4] * 5
    assert folds == [drive_fold(i) for i in range(21, 41)]
    with pytest.raises(ValueError):
        drive_fold(41)


# --- torch parts --------------------------------------------------------------------

def test_unet_shapes_and_input_check():
    m = UNet(UNetConfig(base=4, depth=3))
    assert m(torch.zeros(2, 4, 64, 48)).shape == (2, 3, 64, 48)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 4, 60, 64))


class _Pointwise(torch.nn.Module):
    multiple = 16

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1)

    def forward(self, x):
        return self.conv(x)


def test_tiled_prediction_equals_whole_image_for_pointwise_model():
    torch.manual_seed(0)
    m = _Pointwise().eval()
    img = np.random.default_rng(0).normal(size=(4, 150, 173)).astype(np.float32)
    tiled = predict_image(m, img, tile=64, overlap=0.5)
    whole = torch.sigmoid(m(torch.from_numpy(img)[None])).detach().numpy()[0]
    np.testing.assert_allclose(tiled, whole, atol=1e-5)
    fov = np.zeros((150, 173), bool)
    fov[20:100, 30:120] = True
    assert np.all(predict_image(m, img, fov, tile=64)[:, ~fov] == 0)


def test_loss_ignores_unlabelled_av_pixels():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    y = torch.zeros(2, 3, 8, 8)
    w = torch.zeros(2, 3, 8, 8)
    w[:, 0] = 1
    a = loss_fn(logits, y, w)
    logits2 = logits.clone()
    logits2[:, 1:] = 50.0  # wildly wrong A/V outputs must not matter without A/V labels
    assert float(loss_fn(logits2, y, w)) == pytest.approx(float(a))
    w[:, 1:] = 1
    assert float(loss_fn(logits2, y, w)) > float(a) + 10


def test_load_drive_rite_attaches_av_and_test_set(samples):
    train_ids = [s.image_id for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    assert train_ids == [str(i) for i in range(21, 41)]
    assert all(s.has_av for s in samples) and len(test) == 3
    assert {s.vessel_source for s in test} == {"RITE test (DRIVE 2nd observer)"}


def test_checkpoint_stop_resume_and_reuse(samples, tmp_path, capsys):
    trn = [s for s in samples if s.split == "train"][:4]
    ck = tmp_path / "m.pt"
    _, c1 = train(trn, TINY, ck, log_every=10**9, stop_after=30)
    assert c1["step"] == 30 and not c1["complete"]
    _, c2 = train(trn, TINY, ck, log_every=10**9)
    assert "resuming m.pt from step 30" in capsys.readouterr().out
    assert c2["step"] == 60 and c2["complete"]
    _, c3 = train(trn, TINY, ck, log_every=10**9)  # complete: reused, no training
    assert c3["step"] == 60
    with pytest.raises(ValueError, match="different settings"):
        train(trn, replace(TINY, lr=5e-4), ck)
    model, meta = load_checkpoint(ck)
    assert meta["meta"]["train_ids"] == [s.image_id for s in trn]
    with pytest.raises(FileNotFoundError, match="train them first"):
        load_checkpoint(tmp_path / "missing.pt")


def test_end_to_end_training_learns_vessels_and_av(fake, tmp_path):
    spec = importlib.util.spec_from_file_location("m5", SCRIPT)
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)
    r = m5.run(fake / "DRIVE", fake / "RITE", tmp_path / "out", tmp_path / "ck", mode="both", folds=(0,),
               preset="tiny", device="cpu", overlays=1)
    cv, test = r["cv"]["summary"], r["test"]["summary"]
    assert cv["dice"]["n"] == 4 and test["dice"]["n"] == 3
    assert cv["dice"]["mean"] > 0.75 and test["dice"]["mean"] > 0.75
    assert cv["auc"]["mean"] > 0.97
    assert cv["av_accuracy"]["mean"] > 0.9
    assert cv["centreline_recall"]["mean"] > 0.9
    out = tmp_path / "out"
    assert json.loads((out / "m5_results.json").read_text())["use_av"] is True
    assert (out / "cv_per_image.csv").exists() and (out / "test_per_image.csv").exists()
    assert len(list((out / "probs").rglob("*_vessel.png"))) == 7
    assert len(list((out / "overlays").glob("*.png"))) == 2
    assert (tmp_path / "ck" / "fold0.pt").exists() and (tmp_path / "ck" / "final.pt").exists()


def test_vessel_only_model_when_rite_missing(fake, tmp_path):
    spec = importlib.util.spec_from_file_location("m5", SCRIPT)
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)
    r = m5.run(fake / "DRIVE", tmp_path / "no_rite", tmp_path / "out", tmp_path / "ck", mode="cv", folds=(1,),
               preset="tiny", device="cpu", overlays=0, iterations=40)
    assert r["use_av"] is False
    assert "av_accuracy" not in r["cv"]["summary"]
    assert not list((tmp_path / "out" / "probs").rglob("*_av.png"))
