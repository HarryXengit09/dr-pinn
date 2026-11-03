"""Grading, referral, calibration and uncertainty metrics (numpy/scipy only).

Grades follow the ICDR scale 0-4; "referable" = grade >= 2 (docs/PLAN.md A15).
Confidence intervals: percentile bootstrap over images. Unlike the M7 patches,
images here are the units of analysis (hundreds per test set), where the
percentile bootstrap behaves well. Photos of both eyes of one person may sit in
one test set (APTOS does not identify patients), which makes the intervals
somewhat too narrow; this is stated with the results.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp

from ..datasets.records import REFERABLE_MIN_GRADE
from ..segmentation.metrics import roc_auc

N_GRADES = 5


def confusion_matrix(y: NDArray, pred: NDArray, n: int = N_GRADES) -> NDArray[np.int64]:
    """Rows = true grade, columns = predicted grade."""
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return cm


def quadratic_weighted_kappa(y: NDArray, pred: NDArray, n: int = N_GRADES) -> float:
    o = confusion_matrix(y, pred, n).astype(float)
    if o.sum() == 0:
        return float("nan")
    i, j = np.mgrid[:n, :n]
    w = (i - j) ** 2 / (n - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return float("nan") if den == 0 else float(1 - (w * o).sum() / den)


def per_class(y: NDArray, pred: NDArray, n: int = N_GRADES) -> list[dict]:
    cm = confusion_matrix(y, pred, n)
    out = []
    for k in range(n):
        tp, fn = cm[k, k], cm[k].sum() - cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        tn = cm.sum() - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else float("nan")
        sp = tn / (tn + fp) if tn + fp else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        out.append({"grade": k, "n": int(cm[k].sum()), "sensitivity": float(se), "specificity": float(sp),
                    "f1": float(f1)})
    return out


def macro_f1(y: NDArray, pred: NDArray, n: int = N_GRADES) -> float:
    """Mean F1 over grades that occur in the reference or the prediction."""
    present = set(np.asarray(y, int)) | set(np.asarray(pred, int))
    f = [c["f1"] for c in per_class(y, pred, n) if c["grade"] in present]
    return float(np.mean(f)) if f else float("nan")


def binary_rates(y: NDArray, positive: NDArray) -> dict[str, float]:
    y, p = np.asarray(y, bool), np.asarray(positive, bool)
    tp, fn, fp, tn = (y & p).sum(), (y & ~p).sum(), (~y & p).sum(), (~y & ~p).sum()

    def r(a, b):
        return float(a / (a + b)) if a + b else float("nan")

    return {"sensitivity": r(tp, fn), "specificity": r(tn, fp), "ppv": r(tp, fp), "npv": r(tn, fn)}


def softmax(logits: NDArray, temperature: float = 1.0) -> NDArray:
    return np.exp(log_softmax(np.asarray(logits, float) / temperature, axis=1))


def nll(logits: NDArray, y: NDArray, temperature: float = 1.0) -> float:
    lp = log_softmax(np.asarray(logits, float) / temperature, axis=1)
    return float(-lp[np.arange(len(lp)), np.asarray(y, int)].mean())


def fit_temperature(logits: NDArray, y: NDArray) -> float:
    """Scalar temperature minimising validation NLL (Guo et al. 2017). T > 1 softens over-confidence."""
    res = minimize_scalar(lambda t: nll(logits, y, float(np.exp(t))), bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def referable_probability(probs: NDArray) -> NDArray:
    return np.asarray(probs)[:, REFERABLE_MIN_GRADE:].sum(axis=1)


def referable_log_odds(logits: NDArray) -> NDArray:
    z = np.asarray(logits, float)
    return logsumexp(z[:, REFERABLE_MIN_GRADE:], axis=1) - logsumexp(z[:, :REFERABLE_MIN_GRADE], axis=1)


def normalised_entropy(probs: NDArray) -> NDArray:
    """Predictive entropy / log(classes): 0 = certain, 1 = uniform."""
    p = np.clip(np.asarray(probs, float), 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def expected_calibration_error(confidence: NDArray, correct: NDArray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE of a confidence against 0/1 outcomes."""
    c, ok = np.asarray(confidence, float), np.asarray(correct, float)
    if c.size == 0:
        return float("nan")
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in np.unique(bins):
        m = bins == b
        err += m.mean() * abs(c[m].mean() - ok[m].mean())
    return float(err)


def top_label_ece(probs: NDArray, y: NDArray, n_bins: int = 15) -> float:
    p = np.asarray(probs)
    return expected_calibration_error(p.max(1), p.argmax(1) == np.asarray(y), n_bins)


def reliability_bins(confidence: NDArray, outcome: NDArray, n_bins: int = 10) -> list[dict]:
    c, o = np.asarray(confidence, float), np.asarray(outcome, float)
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    return [{"bin": int(b), "n": int((bins == b).sum()), "mean_confidence": float(c[bins == b].mean()),
             "observed": float(o[bins == b].mean())} for b in np.unique(bins)]


def brier_multiclass(probs: NDArray, y: NDArray) -> float:
    p = np.asarray(probs, float)
    onehot = np.eye(p.shape[1])[np.asarray(y, int)]
    return float(((p - onehot) ** 2).sum(1).mean())


def roc_curve(scores: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """(false-positive rate, true-positive rate) at every distinct threshold, from (0, 0) to (1, 1)."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(~y)[last]
    return np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]


def threshold_for_sensitivity(scores: NDArray, y: NDArray, target: float = 0.90) -> float:
    """Highest threshold t with sensitivity(score >= t) >= target (chosen on validation data)."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    if not y.any():
        raise ValueError("no positive case to set a sensitivity threshold")
    pos = np.sort(s[y])[::-1]
    k = int(np.ceil(target * pos.size))  # need at least k positives at or above t
    return float(pos[k - 1])


def bootstrap_ci(stat: Callable[[NDArray], float], n: int, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(estimate, lower, upper): percentile bootstrap over the n images (stat takes an index array)."""
    est = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    vals = [v for _ in range(n_boot) if np.isfinite(v := stat(rng.integers(0, n, n)))]
    if not vals:
        return est, float("nan"), float("nan")
    a = (1 - level) / 2 * 100
    return float(est), float(np.percentile(vals, a)), float(np.percentile(vals, 100 - a))


def selective_table(probs: NDArray, y: NDArray, coverages=(1.0, 0.9, 0.8, 0.7)) -> list[dict]:
    """Keep the most certain fraction (lowest entropy); the rest would go to a human grader."""
    p, y = np.asarray(probs), np.asarray(y, int)
    order = np.argsort(normalised_entropy(p), kind="stable")
    out = []
    for cov in coverages:
        keep = order[: max(1, int(round(cov * len(y))))]
        pred = p[keep].argmax(1)
        ref_y = y[keep] >= REFERABLE_MIN_GRADE
        ref_auc = roc_auc(referable_probability(p[keep]), ref_y)
        out.append({"coverage": cov, "images": int(keep.size), "accuracy": float((pred == y[keep]).mean()),
                    "qwk": quadratic_weighted_kappa(y[keep], pred), "referable_auc": ref_auc})
    return out


def summarise(probs: NDArray, y: NDArray, threshold: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """All headline metrics for one set of images, with bootstrap CIs for the main ones."""
    p, y = np.asarray(probs, float), np.asarray(y, int)
    pred = p.argmax(1)
    ref_y = y >= REFERABLE_MIN_GRADE
    ref_p = referable_probability(p)
    flagged = ref_p >= threshold

    def ci(fn):
        e, lo, hi = bootstrap_ci(fn, len(y), n_boot, seed)
        return {"value": e, "ci": [lo, hi]}

    return {
        "images": int(len(y)),
        "grade_counts": np.bincount(y, minlength=N_GRADES).tolist(),
        "qwk": ci(lambda i: quadratic_weighted_kappa(y[i], pred[i])),
        "accuracy": ci(lambda i: float((pred[i] == y[i]).mean())),
        "macro_f1": ci(lambda i: macro_f1(y[i], pred[i])),
        "per_class": per_class(y, pred),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "referable": {
            "prevalence": float(ref_y.mean()),
            "auc": ci(lambda i: roc_auc(ref_p[i], ref_y[i])),
            "threshold": threshold,
            "sensitivity": ci(lambda i: binary_rates(ref_y[i], flagged[i])["sensitivity"]),
            "specificity": ci(lambda i: binary_rates(ref_y[i], flagged[i])["specificity"]),
            **{k: v for k, v in binary_rates(ref_y, flagged).items() if k in ("ppv", "npv")},
            "ece": expected_calibration_error(ref_p, ref_y, 10),
        },
        "top_label_ece": top_label_ece(p, y),
        "brier": brier_multiclass(p, y),
        "mean_normalised_entropy": float(normalised_entropy(p).mean()),
        "selective": selective_table(p, y),
    }
