"""One research triage card per photo: quality, calibrated grade probabilities, uncertainty, priority.

All operating points (temperature, referral threshold, uncertainty limit) are
fixed on the VALIDATION set before any test image is scored. The card ranks
photos for research; it is not a referral decision, and it gives no follow-up
interval (PLAN.md use 2: no longitudinal data to support one).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import DISCLAIMER
from ..datasets.records import DR_GRADE_NAMES
from .metrics import normalised_entropy, referable_probability, softmax, threshold_for_sensitivity
from .quality import NO_FUNDUS


@dataclass(frozen=True)
class OperatingPoint:
    temperature: float
    referral_threshold: float      # on calibrated P(referable); target sensitivity on validation
    target_sensitivity: float
    uncertainty_limit: float       # normalised entropy above which a photo is marked "uncertain"
    uncertain_share: float         # share of validation photos above that limit

    def to_dict(self) -> dict:
        return asdict(self)


def choose_operating_point(val_logits, val_y, temperature: float, target_sensitivity: float = 0.90,
                           uncertain_share: float = 0.10) -> OperatingPoint:
    p = softmax(val_logits, temperature)
    thr = threshold_for_sensitivity(referable_probability(p), np.asarray(val_y) >= 2, target_sensitivity)
    lim = float(np.quantile(normalised_entropy(p), 1 - uncertain_share))
    return OperatingPoint(temperature, thr, target_sensitivity, lim, uncertain_share)


def triage_card(logits, op: OperatingPoint, quality_flags: list[str] | None = None) -> dict:
    """Card for one photo from its raw logits (length 5)."""
    p = softmax(np.asarray(logits, float)[None], op.temperature)[0]
    p_ref = float(referable_probability(p[None])[0])
    u = float(normalised_entropy(p[None])[0])
    flags = list(quality_flags or [])
    if NO_FUNDUS in flags:
        priority = "not assessed: no fundus found"
    else:
        priority = "higher priority" if p_ref >= op.referral_threshold else "lower priority"
    notes = []
    if u > op.uncertainty_limit:
        notes.append("model uncertain: needs a human grader")
    if flags and NO_FUNDUS not in flags:
        notes.append("quality outside the training range (" + ", ".join(flags) + "): result less reliable")
    g = int(p.argmax())
    return {"grade": g, "grade_name": DR_GRADE_NAMES[g], "grade_probabilities": [float(x) for x in p],
            "p_referable": p_ref, "uncertainty": u, "priority": priority, "notes": notes,
            "quality_flags": flags, "disclaimer": DISCLAIMER}
