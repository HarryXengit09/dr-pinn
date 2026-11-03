"""M4: loaders, integrity, duplicates and leakage-free splits on SYNTHETIC dataset folders."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from drpinn.datasets import find_drive_samples
from drpinn.datasets.aptos import find_aptos
from drpinn.datasets.idrid import disc_diameter_px, find_idrid
from drpinn.datasets.records import ImageRecord
from drpinn.datasets.rite import decode_av, find_rite
from drpinn.datasets.signatures import SignatureCache, group_duplicates, image_signature, near_duplicate_pairs
from drpinn.datasets.splits import _allocate, check_no_leakage
from drpinn.datasets.synthetic import make_fake_datasets

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    planted = make_fake_datasets(root)
    return root, planted


@pytest.fixture(scope="module")
def prepared(fake, tmp_path_factory):
    root, planted = fake
    out = tmp_path_factory.mktemp("m4out")
    report = _load("m4_prepare_datasets").run(root, out)
    rows = list(csv.DictReader(open(root / "splits" / "manifest.csv")))
    return root, planted, report, rows, out


def test_idrid_discovery(fake):
    root, _ = fake
    idx = find_idrid(root / "IDRiD")
    assert len(idx.grading) == 10 and len(idx.segmentation) == 4
    assert sorted({r.official_split for r in idx.grading}) == ["test", "train"]
    assert sum(r.official_split == "test" for r in idx.grading) == 2
    assert all(0 <= r.dr_grade <= 4 and r.dme_grade in (0, 1, 2) for r in idx.grading)
    assert all("OD" in r.extra["masks"] for r in idx.segmentation)
    assert sum("SE" in r.extra["masks"] for r in idx.segmentation) == 2  # absent for some, as in IDRiD
    assert len(idx.od_centre) == 8 and idx.problems == []
    # train and test reuse file names (IDRiD_001 in both): each label must stay with its own set's photo
    assert len({r.image_id for r in idx.grading}) == 10
    assert all(("Testing" in str(r.path)) == (r.official_split == "test") for r in idx.grading)
    assert {r.image_id for r in idx.grading if r.official_split == "test"} == {"IDRiD_001-test", "IDRiD_002-test"}


def test_aptos_discovery_reports_problems(fake):
    root, planted = fake
    idx = find_aptos(root / "APTOS2019")
    assert len(idx.records) == 14
    assert idx.unlabeled_images == 1
    assert any(planted["aptos_missing_image"] in p for p in idx.problems)


def test_aptos_bad_grade_and_missing_csv(tmp_path):
    (tmp_path / "imgs").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "imgs" / "a.png")
    (tmp_path / "train.csv").write_text("id_code,diagnosis\na,7\n")
    idx = find_aptos(tmp_path)
    assert idx.records == [] and "out of range" in idx.problems[0]
    with pytest.raises(FileNotFoundError):
        find_aptos(tmp_path / "imgs")


def test_rite_discovery_and_colour_decoding(fake):
    root, _ = fake
    idx = find_rite(root / "RITE")
    assert len(idx.records) == 4
    av = np.zeros((2, 3, 3), np.uint8)
    av[0, 0] = (255, 0, 0)
    av[0, 1] = (0, 0, 255)
    av[0, 2] = (0, 255, 0)
    av[1, 0] = (255, 255, 255)
    np.testing.assert_array_equal(decode_av(av), [[1, 2, 3], [4, 0, 0]])
    labels = decode_av(np.asarray(Image.open(idx.records[0].extra["av"]).convert("RGB")))
    assert set(np.unique(labels)) <= {0, 1, 2}


def test_disc_diameter_from_mask(fake):
    root, _ = fake
    rec = find_idrid(root / "IDRiD").segmentation[0]
    od = np.asarray(Image.open(rec.extra["masks"]["OD"])) > 127
    d = disc_diameter_px(od)
    assert d["vertical"] == pytest.approx(2 * 0.08 * 256, abs=2.0)
    with pytest.raises(ValueError):
        disc_diameter_px(np.zeros((5, 5), bool))


def test_signatures_detect_corruption_and_duplicates(fake):
    root, planted = fake
    bad = image_signature(root / "APTOS2019" / "train_images" / f"{planted['aptos_corrupt']}.png")
    assert not bad[0].ok and bad[1] is None
    a = image_signature(root / "APTOS2019" / "train_images" / f"{planted['aptos_conflict'][0]}.png")[1]
    b = image_signature(root / "APTOS2019" / "train_images" / f"{planted['aptos_conflict'][1]}.png")[1]
    assert float(a @ b) > 0.95


def test_signature_calibration_separates_copies_from_different_eyes(tmp_path):
    """Different eyes stay well below the threshold; altered copies stay well above."""
    from drpinn.datasets.signatures import DEFAULT_THRESHOLD
    from drpinn.datasets.synthetic import _variant, fake_fundus

    orig, copies = [], []
    for k in range(30):
        img = fake_fundus(256, k)[0]
        Image.fromarray(img).save(tmp_path / f"{k}.png")
        Image.fromarray(_variant(img, scale=0.7, gain=1.08)).save(tmp_path / f"c{k}.jpg", quality=75)
        orig.append(image_signature(tmp_path / f"{k}.png")[1])
        copies.append(image_signature(tmp_path / f"c{k}.jpg")[1])
    v, c = np.array(orig), np.array(copies)
    sim = v @ v.T
    np.fill_diagonal(sim, -1)
    assert sim.max() < 0.85 < DEFAULT_THRESHOLD
    assert min(float(c[k] @ v[k]) for k in range(30)) > 0.95


def test_grouping_is_transitive():
    pairs = [(0, 1, 0.99), (1, 2, 0.95)]
    g = group_duplicates(4, pairs)
    assert g[0] == g[1] == g[2] != g[3]
    v = np.eye(3, dtype=np.float32)
    assert near_duplicate_pairs(v, 0.5) == []


def test_signature_cache_roundtrip(fake, tmp_path):
    root, _ = fake
    paths = sorted((root / "APTOS2019" / "train_images").glob("*.png"))[:3]
    c = SignatureCache(tmp_path / "cache")
    first = [c.get(p)[1] for p in paths]
    c.save()
    c2 = SignatureCache(tmp_path / "cache")
    second = [c2.get(p)[1] for p in paths]
    assert len(c2.meta) == 3
    for x, y in zip(first, second):
        np.testing.assert_allclose(x, y, atol=1e-3)


def test_prepare_finds_every_planted_problem(prepared):
    root, planted, report, rows, out = prepared
    assert len(report["corrupt_images"]) == 1
    assert report["duplicate_pairs"] == 5  # 3 DRIVE-RITE, 1 IDRiD cross-part, 1 APTOS
    assert report["duplicate_pairs_by_kind"]["DRIVE/vessels vs RITE/av"] == 3
    assert report["grouping"]["label_conflict_groups"] == 1
    assert report["grouping"]["moved_to_test_by_duplicates"] == 1
    assert report["leakage_check"] == "passed"
    ids = {(r["dataset"], r["image_id"]) for r in rows}
    assert ("APTOS2019", planted["aptos_corrupt"]) not in ids
    assert not any(("APTOS2019", x) in ids for x in planted["aptos_conflict"])
    pairs = list(csv.DictReader(open(out / "duplicate_pairs.csv")))
    assert sum(p["kind"] == "duplicate" for p in pairs) == 5


def test_duplicate_of_a_test_image_is_never_trained_on(prepared):
    _, planted, _, rows, _ = prepared
    seg_id, grading_id = planted["cross_part_duplicate"]
    by_id = {(r["part"], r["image_id"]): r for r in rows if r["dataset"] == "IDRiD"}
    assert by_id[("segmentation", seg_id)]["split"] == "test"
    assert by_id[("grading", grading_id)]["split"] == "test"
    assert by_id[("grading", grading_id)]["official_split"] == "train"


def test_shared_photos_share_a_cv_fold(prepared):
    _, _, _, rows, _ = prepared
    fold = {(r["dataset"], int(r["image_id"])): r["fold"] for r in rows if r["split"] == "cv"}
    for k in (21, 22, 23):
        assert fold[("DRIVE", k)] == fold[("RITE", k)]


def test_manifest_paths_are_relative_and_exist(prepared):
    root, _, _, rows, _ = prepared
    for r in rows:
        assert not Path(r["rel_path"]).is_absolute() and (root / r["rel_path"]).exists()
        for v in json.loads(r["extra"]).get("masks", {}).values():
            assert (root / v).exists()
        if r["referable"] != "":
            assert int(r["referable"]) == int(int(r["dr_grade"]) >= 2)


def test_same_seed_gives_identical_manifest(prepared, tmp_path):
    root, _, report, _, _ = prepared
    again = _load("m4_prepare_datasets").run(root, tmp_path / "again", seed=0)
    assert again["manifest_fingerprint"] == report["manifest_fingerprint"]
    c = again["aptos_split_vs_previous_manifest"]
    assert c["changed_split"] == 0 and c["compared"] > 0 and (root / "splits" / "manifest_previous.csv").exists()


def test_aptos_split_does_not_depend_on_other_datasets():
    """Fixing or adding IDRiD/DRIVE records must not move APTOS photos (a trained model relies on the split)."""
    from drpinn.datasets.splits import assign_splits

    rng = np.random.default_rng(3)
    aptos = [ImageRecord("APTOS2019", "grading", f"a{i}", Path(f"a{i}.png"), "", int(g))
             for i, g in enumerate(rng.choice(5, 300))]
    splits = []
    for n_other in (0, 7, 40):
        other = [ImageRecord("IDRiD", "grading", f"IDRiD_{i:03d}-train", Path(f"i{i}.jpg"), "train", int(i % 5))
                 for i in range(n_other)]
        recs = other + aptos
        rows, _ = assign_splits(recs, np.arange(len(recs)), seed=0)
        splits.append({r["image_id"]: r["split"] for r in rows if r["dataset"] == "APTOS2019"})
    assert splits[0] == splits[1] == splits[2]


def test_stratified_allocation_matches_fractions_in_every_grade():
    grades = np.random.default_rng(1).choice(5, 3662, p=[0.49, 0.10, 0.28, 0.05, 0.08])
    out = _allocate(list(enumerate(grades)), {"train": 0.70, "val": 0.15, "test": 0.15}, np.random.default_rng(0))
    for gr in range(5):
        sel = [out[i] for i in range(len(grades)) if grades[i] == gr]
        assert abs(sel.count("test") / len(sel) - 0.15) < 0.01


def test_leakage_checker_catches_a_split_group():
    rows = [{"dup_group": 1, "split": "train", "fold": ""}, {"dup_group": 1, "split": "test", "fold": ""}]
    assert check_no_leakage(rows)


def test_referable_definition():
    assert ImageRecord("x", "grading", "a", Path("a"), "", dr_grade=2).referable is True
    assert ImageRecord("x", "grading", "a", Path("a"), "", dr_grade=1).referable is False
    assert ImageRecord("x", "segmentation", "a", Path("a"), "").referable is None


def test_working_copy_crops_resizes_and_resumes(fake, tmp_path):
    root, _ = fake
    mod = _load("m4_make_working_copy")
    src = root / "APTOS2019"
    r1 = mod.convert(src, tmp_path / "copy", max_side=128)
    assert r1["converted"] == 14 and len(r1["failed"]) == 1  # the corrupt PNG is reported
    out = sorted((tmp_path / "copy").rglob("*.jpg"))
    assert all(max(Image.open(p).size) <= 128 for p in out)
    assert (tmp_path / "copy" / "train.csv").exists()
    r2 = mod.convert(src, tmp_path / "copy", max_side=128)
    assert r2["converted"] == 0 and r2["skipped_existing"] == 14
    assert len(find_aptos(tmp_path / "copy").records) == 13  # corrupt one not copied


def test_drive_finder_on_fake_drive(fake):
    root, _ = fake
    assert [s.image_id for s in find_drive_samples(root / "DRIVE")] == ["21", "22", "23"]
