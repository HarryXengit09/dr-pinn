"""DRIVE discovery + graph script, on a fake DRIVE folder built from synthetic masks."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from drpinn.datasets import find_drive_samples, load_binary
from drpinn.vessel_graph.synthetic_masks import fundus_like

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m3_drive_graphs.py"
SHAPE = (584, 565)  # DRIVE image size


def _save(arr, path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).convert(mode).save(path)


@pytest.fixture(scope="module")
def fake_drive(tmp_path_factory):
    root = tmp_path_factory.mktemp("DRIVE")
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    fov = ((yy - 292) ** 2 + (xx - 282) ** 2 <= 270**2).astype(np.uint8) * 255
    for iid, seed in (("21", 0), ("22", 1)):
        m = (fundus_like(shape=SHAPE, seed=seed, generations=5, segment_length=80).binary * 255).astype(np.uint8)
        rgb = np.dstack([m // 2 + 60, m // 3 + 30, np.full_like(m, 20)])
        if iid == "21":  # official layout and formats
            _save(m, root / "training" / "1st_manual" / f"{iid}_manual1.gif", "L")
            _save(fov, root / "training" / "mask" / f"{iid}_training_mask.gif", "L")
            _save(rgb, root / "training" / "images" / f"{iid}_training.tif", "RGB")
            _save(m, root / "training" / "2nd_manual" / f"{iid}_manual2.gif", "L")  # must be ignored
        else:  # a converted copy: flat folder, PNG
            _save(m, root / "png_copy" / f"{iid}_manual1.png", "L")
            _save(fov, root / "png_copy" / f"{iid}_training_mask.png", "L")
            _save(rgb, root / "png_copy" / f"{iid}_training.png", "RGB")
    return root


def test_discovery_handles_both_layouts(fake_drive):
    samples = find_drive_samples(fake_drive)
    assert [s.image_id for s in samples] == ["21", "22"]
    assert all(s.fov_path is not None and s.image_path is not None for s in samples)
    assert all("manual1" in s.manual_path.name for s in samples)
    assert load_binary(samples[0].manual_path).shape == SHAPE


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_drive_samples(tmp_path / "nope")


def test_script_writes_stats_summary_and_overlays(fake_drive, tmp_path):
    spec = importlib.util.spec_from_file_location("m3_drive_graphs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.run(fake_drive, tmp_path / "out", n_overlays=1)
    assert summary["images"] == 2 and summary["failures"] == []
    rows = list(csv.DictReader(open(tmp_path / "out" / "drive_graph_stats.csv")))
    assert len(rows) == 2 and all(int(r["segments"]) > 20 for r in rows)
    assert all(float(r["seconds"]) < 10 for r in rows)
    assert (tmp_path / "out" / "overlays" / "drive_21_graph.png").exists()
    assert json.loads((tmp_path / "out" / "drive_graph_summary.json").read_text())["images"] == 2


def test_script_reports_no_annotations_clearly(tmp_path):
    spec = importlib.util.spec_from_file_location("m3_drive_graphs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="manual1"):
        mod.run(tmp_path / "empty", tmp_path / "out")
