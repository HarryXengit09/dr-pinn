"""M9 disc check and M11 results collector (SYNTHETIC inputs)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from drpinn.datasets.synthetic import fake_fundus_av

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set(v, n=100):
    m = {"value": v, "ci": [v - 0.05, v + 0.05]}
    return {"description": "x", "images": n, "qwk": m, "accuracy": m, "macro_f1": m,
            "referable": {"auc": m, "sensitivity": m, "specificity": m}, "calibration": {"referable_ece_after": 0.03},
            "quality": {"quality_flagged": {"images": 3}}}


def test_collector_reports_missing_results(tmp_path):
    summary, md = _load("m11_collect_results").collect(tmp_path)
    assert len(summary["missing"]) == 6 and "Not available yet" in md


def test_collector_copies_numbers_and_drops_invalid_idrid_results(tmp_path):
    rep = tmp_path / "reports" / "m6_full"
    rep.mkdir(parents=True)
    res = {"source": "APTOS 2019 + IDRiD grading (manifest)", "drpinn_version": "0.7.1", "preset": "full",
           "temperature": 0.85, "operating_point": {"referral_threshold": 0.723, "target_sensitivity": 0.9},
           "training": {"best_epoch": 9},
           "sets": {"test_internal": _set(0.911), "external_idrid_test": _set(0.497, 53)}}
    (rep / "m6_results.json").write_text(json.dumps(res))
    summary, md = _load("m11_collect_results").collect(tmp_path)
    assert "0.911 [0.861, 0.961]" in md and "external_idrid_test" not in summary["m6"]["sets"]
    assert any("before the IDRiD fix" in w for w in summary["warnings"])
    res["drpinn_version"] = "0.7.2"
    (rep / "m6_results.json").write_text(json.dumps(res))
    summary, _ = _load("m11_collect_results").collect(tmp_path)
    assert summary["m6"]["sets"]["external_idrid_test"]["images"] == 53 and not summary["warnings"]


def test_collector_flags_synthetic_runs(tmp_path):
    rep = tmp_path / "reports" / "m7_full"
    rep.mkdir(parents=True)
    (rep / "m7_results.json").write_text(json.dumps({"source": "SYNTHETIC (null) - software check", "primary": {}}))
    summary, _ = _load("m11_collect_results").collect(tmp_path)
    assert any("synthetic" in w for w in summary["warnings"])


def test_disc_check_scores_the_estimate_against_expert_discs(tmp_path):
    eyes = []
    for s in range(4):
        img, v, fov, _, _, disc = fake_fundus_av(400, s, return_disc=True)
        yy, xx = np.mgrid[:400, :400]
        od = (yy - disc[0]) ** 2 + (xx - disc[1]) ** 2 <= (0.06 * 400) ** 2
        eyes.append((f"e{s}", "test", img, fov, od, {}))
    s = _load("m9_check_disc").run(Path("unused"), tmp_path, None, eyes)
    assert s["images"] == 4 and s["brightness"]["hit_rate"] == 1.0
    assert abs(s["true_disc_to_fov_median"] - 0.12 / 0.94) < 0.01  # disc 0.12, field of view 0.94 of the image
    assert (tmp_path / "disc_check.json").exists()


def test_collector_catches_old_manifest_and_foreign_code(tmp_path):
    import csv

    (tmp_path / "data" / "splits").mkdir(parents=True)
    with open(tmp_path / "data" / "splits" / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "part", "image_id", "split"])
        w.writerow(["IDRiD", "grading", "IDRiD_001", "test"])  # pre-0.7.2 id: no -train/-test suffix
    rep = tmp_path / "reports" / "m6_full"
    rep.mkdir(parents=True)
    (rep / "m6_results.json").write_text(json.dumps({"drpinn_version": "0.7.2", "preset": "full", "operating_point": {},
                                                     "sets": {"test_internal": _set(0.9), "external_idrid_test": _set(0.5, 22)}}))
    m7 = tmp_path / "reports" / "m7_full"
    m7.mkdir(parents=True)
    (m7 / "m7_results.json").write_text(json.dumps({"drpinn_version": "99.0.0", "primary": {}}))
    summary, _ = _load("m11_collect_results").collect(tmp_path)
    w = " | ".join(summary["warnings"])
    assert "before the IDRiD fix" in w and "99.0.0" in w and "external_idrid_test" not in summary["m6"]["sets"]


def test_collector_catches_results_from_a_different_manifest(tmp_path):
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "splits" / "manifest.csv").write_text("dataset,part,image_id,split\nIDRiD,grading,IDRiD_001-test,test\n")
    rep = tmp_path / "reports" / "m6_full"
    rep.mkdir(parents=True)
    (rep / "m6_results.json").write_text(json.dumps({"drpinn_version": "0.8.1", "preset": "full", "operating_point": {},
                                                     "manifest_sha": "0000000000000000",
                                                     "sets": {"external_idrid_test": _set(0.5, 103)}}))
    summary, _ = _load("m11_collect_results").collect(tmp_path)
    assert any("different split manifest" in x for x in summary["warnings"]) and not summary["m6"]["sets"]

