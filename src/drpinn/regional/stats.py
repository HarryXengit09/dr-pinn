"""Statistics for patch-level association, respecting that patches of one eye are not independent.

- Within-image AUROC: computed per image (needs >= 1 lesion and >= 1 lesion-free
  patch), then averaged. Matches the claim being tested: a WITHIN-EYE ranking.
- Pooled AUROC / average precision across all patches (secondary).
- Uncertainty is always over IMAGES (never patches), so correlated patches of
  one eye move together. For the mean within-image AUROC and for paired
  differences: a t-interval / paired t-test over per-image values. A simulation
  under the null (lesions placed at random) showed the bootstrap-percentile
  interval rejects far too often with few images (23 % at 4 images, 10 % at 10;
  target 5 %), while the t-interval stayed at 4-6 %. Pooled AUROC keeps an
  image-bootstrap interval (secondary).
- Logistic regression (L2), for the exploratory combined models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _st
from scipy.optimize import minimize

from ..segmentation.metrics import roc_auc


def average_precision(scores: NDArray, labels: NDArray) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / y.sum())


def _clean(scores, labels, groups):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    g = np.asarray(groups)
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1 if np.isfinite(s).any() else 0.0)
    return s, y, g


def per_image_auroc(scores, labels, groups) -> dict:
    s, y, g = _clean(scores, labels, groups)
    out = {}
    for img in np.unique(g):
        m = g == img
        if y[m].any() and (~y[m]).any():
            out[img] = roc_auc(s[m], y[m])
    return out


def image_bootstrap(stat_fn, groups, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, 2.5 %, 97.5 %) of stat_fn(list_of_images) with images resampled."""
    imgs = np.unique(np.asarray(groups))
    est = stat_fn(imgs)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat_fn(rng.choice(imgs, imgs.size, replace=True))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def t_interval(values) -> tuple[float, float, float]:
    """(mean, lower, upper) 95 % t-interval; NaN when fewer than 2 values."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return (float(v.mean()) if v.size else float("nan")), float("nan"), float("nan")
    half = _st.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def evaluate_score(scores, labels, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Within-image and pooled discrimination of one score for one lesion label."""
    s, y, g = _clean(scores, labels, groups)
    per = per_image_auroc(s, y, g)
    mean, lo, hi = t_interval(list(per.values()))
    p_vs_half = (float(_st.ttest_1samp(list(per.values()), 0.5).pvalue) if len(per) > 1 else float("nan"))

    def pooled(imgs):
        m = np.concatenate([np.flatnonzero(g == i) for i in imgs])
        return roc_auc(s[m], y[m])

    p = image_bootstrap(pooled, g, n_boot, seed)
    return {"within_image_auroc": mean, "within_ci": [lo, hi], "p_vs_0.5": p_vs_half, "images_scored": len(per),
            "pooled_auroc": p[0], "pooled_ci": [p[1], p[2]], "average_precision": average_precision(s, y),
            "prevalence": float(y.mean()) if y.size else float("nan"), "patches": int(y.size)}


def paired_difference(scores_a, scores_b, labels, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean within-image AUROC of A minus B: paired t-interval and two-sided paired t-test over images."""
    a = per_image_auroc(scores_a, labels, groups)
    b = per_image_auroc(scores_b, labels, groups)
    imgs = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in imgs])
    mean, lo, hi = t_interval(d)
    p = float(_st.ttest_1samp(d, 0.0).pvalue) if d.size > 1 and d.std() > 0 else float("nan")
    return {"difference": mean, "ci": [lo, hi], "p_value": p, "images": int(d.size)}


class Logistic:
    """L2-regularised logistic regression with feature standardisation (fit on training data only)."""

    def __init__(self, c: float = 1.0):
        self.c = c

    def fit(self, x: NDArray, y: NDArray) -> "Logistic":
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[self.std == 0] = 1.0
        z = np.nan_to_num((x - self.mean) / self.std)
        z = np.column_stack([np.ones(len(z)), z])

        def f(wb):
            t = z @ wb
            loss = np.sum(np.logaddexp(0, t) - y * t) / len(y) + 0.5 / (self.c * len(y)) * np.sum(wb[1:] ** 2)
            p = 1 / (1 + np.exp(-t))
            grad = z.T @ (p - y) / len(y)
            grad[1:] += wb[1:] / (self.c * len(y))
            return loss, grad

        self.w = minimize(f, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def decision(self, x: NDArray) -> NDArray:
        z = np.nan_to_num((np.asarray(x, float) - self.mean) / self.std)
        return self.w[0] + z @ self.w[1:]
