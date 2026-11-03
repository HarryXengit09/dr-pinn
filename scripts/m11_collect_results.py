"""M11: collect every milestone's results into one machine-readable summary and one readable report.

Reads what the Colab runs wrote under reports/ and never recomputes or edits a
number. It records where each number came from (file, code version, date) and
warns about results that must not be reported:
  - synthetic runs (software checks);
  - 'quick' presets (pipeline checks only);
  - M6 IDRiD results made before code version 0.7.2 (IDRiD loader bug, docs/m6_triage.md).

    python scripts/m11_collect_results.py --project .
Outputs: reports/summary/results_summary.json and reports/summary/results_summary.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpinn import DISCLAIMER, __version__  # noqa: E402


def _version(v) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except ValueError:
        return (0,)


def _load(project: Path, candidates: list[str]) -> tuple[dict | None, dict]:
    for rel in candidates:
        p = project / rel
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except ValueError as exc:
                return None, {"file": rel, "error": f"unreadable JSON: {exc}"}
            when = dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")
            return data, {"file": rel, "modified": when}
    return None, {"file": candidates[0], "missing": True}


def _ms(s: dict, k: str, d: int = 3) -> str:
    v = s.get(k)
    return "–" if not v or v.get("mean") is None else f"{v['mean']:.{d}f} ± {v.get('sd', 0):.{d}f}"


def _f(x, d: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def _ci(m: dict | None, d: int = 3) -> str:
    if not m or _f(m.get("value")) == "–":
        return "–"
    lo, hi = (m.get("ci") or [None, None])[:2]
    if _f(lo) == "–" or _f(hi) == "–":
        return _f(m["value"], d)
    return f"{m['value']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def collect(project: Path) -> tuple[dict, str]:
    project = Path(project)
    out = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "drpinn_version": __version__,
           "disclaimer": DISCLAIMER, "provenance": {}, "warnings": [], "missing": []}
    md = [f"# Results summary\n\nGenerated {out['generated']} by `scripts/m11_collect_results.py` "
          f"(code {__version__}). Every number is copied from the file named in its section; nothing is recomputed.\n",
          f"> {DISCLAIMER}\n"]

    def note(section: str, prov: dict, data: dict | None, label: str) -> bool:
        out["provenance"][section] = prov
        if data is None:
            out["missing"].append(f"{label}: {prov['file']}" + (f" ({prov['error']})" if "error" in prov else ""))
            md.append(f"## {label}\nNot available yet: `{prov['file']}`.\n")
            return False
        src = str(data.get("source", ""))
        if "SYNTHETIC" in src.upper():
            out["warnings"].append(f"{label}: synthetic software check, not a result ({prov['file']})")
        v = data.get("drpinn_version")
        if v and _version(v) > _version(__version__):
            out["warnings"].append(f"{label}: made by code version {v}, which is not a version of this project "
                                   f"(installed: {__version__}). Re-run that step with the installed code.")
        return True

    man = project / "data" / "splits" / "manifest.csv"
    man_sha, old_idrid = None, 0
    if man.exists():
        import csv
        import hashlib

        man_sha = hashlib.sha256(man.read_bytes()).hexdigest()[:16]
        with open(man, newline="") as f:
            old_idrid = sum(1 for r in csv.DictReader(f) if r.get("dataset") == "IDRiD" and r.get("part") == "grading"
                            and not r.get("image_id", "").endswith(("-train", "-test")))
        if old_idrid:
            out["warnings"].append(f"M4: the split manifest was built before the IDRiD fix ({old_idrid} IDRiD grading rows "
                                   "with old ids). Re-run step 11d, then step 15; until then every IDRiD triage number is invalid")

    # M4
    d, prov = _load(project, ["reports/m4_data/integrity_report.json"])
    if note("m4", prov, d, "M4 · Datasets and splits"):
        out["m4"] = {"manifest_fingerprint": d.get("manifest_fingerprint"), "splits": d.get("splits"),
                     "grouping": d.get("grouping"), "corrupt_images": len(d.get("corrupt_images", [])),
                     "aptos_split_vs_previous_manifest": d.get("aptos_split_vs_previous_manifest")}
        md.append("## M4 · Datasets and splits\n" + f"Source `{prov['file']}`. Manifest fingerprint "
                  f"**{d.get('manifest_fingerprint')}** (quote it: it identifies the exact split).\n")
        md.append("| dataset | split | photos | grades 0/1/2/3/4 |\n|---|---|---|---|")
        for ds, sp in (d.get("splits") or {}).items():
            for s, c in sp.items():
                md.append(f"| {ds} | {s} | {sum(c.values())} | {' / '.join(str(c.get(str(g), 0)) for g in range(5))} |")
        md.append("")

    # M5
    d, prov = _load(project, ["reports/m5_full/m5_results.json", "reports/m5_quick/m5_results.json"])
    if note("m5", prov, d, "M5 · Vessel and artery/vein segmentation"):
        if d.get("preset") != "full":
            out["warnings"].append(f"M5: preset '{d.get('preset')}' is a pipeline check, not a thesis result")
        keys = [("dice", "Dice"), ("auc", "AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                ("centreline_recall", "Centreline recall"), ("centreline_precision", "Centreline precision"),
                ("radius_median_abs_error", "Radius error (median, relative)"), ("av_accuracy", "Artery/vein accuracy")]
        cv, te = d.get("cv", {}).get("summary", {}), d.get("test", {}).get("summary", {})
        out["m5"] = {"preset": d.get("preset"), "artery_vein": d.get("use_av"),
                     "cv": {k: cv.get(k) for k, _ in keys}, "test": {k: te.get(k) for k, _ in keys},
                     "test_note": d.get("test", {}).get("note")}
        md.append(f"## M5 · Vessel and artery/vein segmentation\nSource `{prov['file']}` (preset {d.get('preset')}, "
                  f"artery/vein trained: {d.get('use_av')}). Mean ± SD over photos.\n")
        md.append("| metric | 5-fold CV (DRIVE, 1st observer) | test (DRIVE test photos, 2nd observer) |\n|---|---|---|")
        md += [f"| {name} | {_ms(cv, k)} | {_ms(te, k)} |" for k, name in keys]
        md.append("")

    # M6
    d, prov = _load(project, ["reports/m6_full/m6_results.json", "reports/m6_quick/m6_results.json"])
    if note("m6", prov, d, "M6 · DR grade triage"):
        prov["code_version"] = d.get("drpinn_version")
        sets = dict(d.get("sets", {}))
        stale = (_version(d.get("drpinn_version")) < (0, 7, 2) or bool(old_idrid)
                 or (d.get("manifest_sha") is not None and man_sha is not None and d["manifest_sha"] != man_sha))
        if d.get("manifest_sha") and man_sha and d["manifest_sha"] != man_sha:
            out["warnings"].append("M6: these results used a different split manifest than the current one; re-run step 15")
        if stale:
            dropped = [k for k in sets if k.startswith("external_idrid")]
            for k in dropped:
                sets.pop(k)
            if dropped:
                out["warnings"].append("M6: IDRiD results are left out (made before the IDRiD fix, or from an old or different "
                                       "manifest); re-run step 11d, then step 15, to get valid external results")
        if d.get("preset") != "full":
            out["warnings"].append(f"M6: preset '{d.get('preset')}' is a pipeline check, not a thesis result")
        out["m6"] = {"preset": d.get("preset"), "selected_epoch": d.get("training", {}).get("best_epoch"),
                     "temperature": d.get("temperature"), "operating_point": d.get("operating_point"),
                     "sets": {k: {"images": s["images"], "qwk": s["qwk"], "accuracy": s["accuracy"],
                                  "macro_f1": s["macro_f1"], "referable_auc": s["referable"]["auc"],
                                  "sensitivity": s["referable"]["sensitivity"],
                                  "specificity": s["referable"]["specificity"],
                                  "referable_ece_after": s.get("calibration", {}).get("referable_ece_after"),
                                  "quality_flagged": s.get("quality", {}).get("quality_flagged", {}).get("images")}
                              for k, s in sets.items()}}
        op = d.get("operating_point", {})
        md.append(f"## M6 · DR grade triage\nSource `{prov['file']}` (code {d.get('drpinn_version')}, preset "
                  f"{d.get('preset')}). Fixed on validation: epoch {out['m6']['selected_epoch']}, temperature "
                  f"{_f(d.get('temperature'), 2)}, referral threshold {_f(op.get('referral_threshold'))} "
                  f"(target sensitivity {op.get('target_sensitivity')}). 95 % bootstrap CIs.\n")
        md.append("| set | photos | QWK | accuracy | macro-F1 | referable AUC | sensitivity | specificity | "
                  "referable ECE (calibrated) |\n|---|---|---|---|---|---|---|---|---|")
        for k, s in out["m6"]["sets"].items():
            md.append(f"| {sets[k].get('description', k)} | {s['images']} | {_ci(s['qwk'])} | {_ci(s['accuracy'])} | "
                      f"{_ci(s['macro_f1'])} | {_ci(s['referable_auc'])} | {_ci(s['sensitivity'])} | "
                      f"{_ci(s['specificity'])} | {_f(s['referable_ece_after'])} |")
        md.append("\nPrimary endpoints (pre-registered): referable AUC on the IDRiD official test; QWK on the APTOS "
                  "internal test.\n")

    # M7
    d, prov = _load(project, ["reports/m7_full/m7_results.json", "reports/m7_quick/m7_results.json"])
    if note("m7", prov, d, "M7 · Shear risk vs lesions (core experiment)"):
        prov["code_version"] = d.get("drpinn_version")
        if "m5_quick" in str(d.get("model", {}).get("checkpoint", "")):
            out["warnings"].append("M7: run with the 'quick' vessel model; thesis results must use the 'full' model")
        p = d.get("primary", {})
        cm = d.get("results", {}).get("combined_models", {})
        out["m7"] = {"images": d.get("images"), "test_images": d.get("test_images"), "patches": d.get("patches"),
                     "radius_sigma": d.get("config", {}).get("mc", {}).get("radius_sigma"),
                     "primary": p, "physics_added": cm.get("physics added (paired)"),
                     "median_rank_stability": d.get("median_rank_stability"),
                     "red_lesion_pixels_labelled_vessel_median": d.get("red_lesion_pixels_labelled_vessel_median")}
        md.append(f"## M7 · Shear risk vs lesions (core experiment)\nSource `{prov['file']}` (code "
                  f"{d.get('drpinn_version')}): {d.get('images')} photos ({d.get('test_images')} official test), "
                  f"{d.get('patches')} vessel patches, radius σ {out['m7']['radius_sigma']}.\n")
        if p.get("available"):
            md.append(f"**Primary (pre-registered):** within-image AUROC of the Monte Carlo high-shear score for red "
                      f"lesions = **{_f(p['within_image_auroc'])} [{_f(p['ci'][0])}, {_f(p['ci'][1])}]**, "
                      f"p = {_f(p.get('p_vs_0.5'))} vs 0.5, {p['images']} test photos → **{p['verdict']}**.\n")
            md.append("| baseline | high-shear minus baseline (AUROC) | 95 % CI | p | photos |\n|---|---|---|---|---|")
            for b, x in p.get("vs_baselines", {}).items():
                md.append(f"| {b} | {x['difference']:+.3f} | [{_f(x['ci'][0])}, {_f(x['ci'][1])}] | "
                          f"{_f(x['p_value'])} | {x['images']} |")
            md.append("")
        pa = out["m7"]["physics_added"]
        if pa:
            md.append(f"Exploratory: adding physics to position + geometry (logistic, trained on the 54 train photos) "
                      f"changes the AUROC by {pa['difference']:+.3f} [{_f(pa['ci'][0])}, {_f(pa['ci'][1])}].\n")
        md.append(f"Median rank stability: {out['m7']['median_rank_stability']}. Median share of red-lesion pixels "
                  f"drawn as vessel: {_f(out['m7']['red_lesion_pixels_labelled_vessel_median'])}.\n")

    # M8
    d, prov = _load(project, ["reports/m8_full/m8_results.json", "reports/m8_quick/m8_results.json"])
    if note("m8", prov, d, "M8 · PINN vs classical solver"):
        if "m8_quick" in prov["file"]:
            out["warnings"].append("M8: 'quick' run (fewer networks); the 'full' run is the one to report")
        out["m8"] = {"settings": d.get("settings"), "summary": d.get("summary")}
        md.append(f"## M8 · PINN vs classical solver\nSource `{prov['file']}`. Pressure errors are fractions of the "
                  "inlet–outlet pressure drop.\n")
        for st, methods in (d.get("summary") or {}).items():
            n = next(iter(methods.values()), {}).get("networks")
            md.append(f"**{st} networks (n = {n})**\n\n| method | pressure error (median) | pressure error (worst) | "
                      "WSS top-10 % kept | time per network |\n|---|---|---|---|---|")
            md += [f"| {m} | {s['pressure_error_median']:.1e} | {s['pressure_error_worst']:.1e} | "
                   f"{s['wss_top10_overlap_median']:.2f} | {1000 * s['seconds_median']:.1f} ms |" for m, s in methods.items()]
            md.append("")

    # M9 disc check
    d, prov = _load(project, ["reports/m9_disc/disc_check.json"])
    if note("m9_disc", prov, d, "M9 · Optic-disc estimate used by the API"):
        s = d.get("summary", {})
        out["m9_disc"] = s
        md.append(f"## M9 · Optic-disc estimate used by the API\nSource `{prov['file']}` ({s.get('images')} IDRiD "
                  f"photos). Expert disc diameter / field of view: median {_f(s.get('true_disc_to_fov_median'))} "
                  f"(assumed {s.get('assumed_disc_to_fov')}).\n")
        md.append("| estimate | hit rate (inside expert disc) | median error (disc diameters) |\n|---|---|---|")
        for k in ("brightness", "brightness + vessels"):
            if k in s:
                md.append(f"| {k} | {s[k]['hit_rate']:.0%} | {_f(s[k]['median_error_disc_diameters'], 2)} |")
        md.append("")

    head = []
    if out["warnings"]:
        head.append("## Warnings\n" + "\n".join(f"- {w}" for w in out["warnings"]) + "\n")
    if out["missing"]:
        head.append("## Still missing\n" + "\n".join(f"- {m}" for m in out["missing"]) + "\n")
    md = md[:2] + head + md[2:]
    return out, "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=str(ROOT))
    ap.add_argument("--out", default=None, help="default <project>/reports/summary")
    a = ap.parse_args()
    out_dir = Path(a.out or Path(a.project) / "reports" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, md = collect(Path(a.project))
    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "results_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
