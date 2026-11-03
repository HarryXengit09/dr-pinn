"""Unit conversions and vitals-derived pressure helpers.

Convention: every physics routine in this package works in SI units
(pressure Pa, length m, viscosity Pa*s, flow m^3/s, wall shear stress Pa).
Convert only at the edges (user input, reports, plots).

The vitals helpers below are standard *approximations*, not measurements of
retinal pressure. Their outputs are modelling assumptions and must be labelled
as such wherever they are used.
"""

from __future__ import annotations

import math

MMHG_TO_PA: float = 133.322387415  # conventional millimetre of mercury
UM_TO_M: float = 1e-6
CP_TO_PA_S: float = 1e-3  # 1 centipoise = 1 mPa*s
DYN_PER_CM2_TO_PA: float = 0.1
M3_S_TO_UL_PER_MIN: float = 1e9 * 60.0  # 1 m^3 = 1e9 microlitres


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def mmhg_to_pa(p_mmhg: float) -> float:
    return float(p_mmhg) * MMHG_TO_PA


def pa_to_mmhg(p_pa: float) -> float:
    return float(p_pa) / MMHG_TO_PA


def um_to_m(x_um: float) -> float:
    return float(x_um) * UM_TO_M


def m_to_um(x_m: float) -> float:
    return float(x_m) / UM_TO_M


def cp_to_pa_s(mu_cp: float) -> float:
    return float(mu_cp) * CP_TO_PA_S


def pa_to_dyn_per_cm2(tau_pa: float) -> float:
    return float(tau_pa) / DYN_PER_CM2_TO_PA


def m3_s_to_ul_per_min(q_m3_s: float) -> float:
    return float(q_m3_s) * M3_S_TO_UL_PER_MIN


def mean_arterial_pressure(systolic_mmhg: float, diastolic_mmhg: float) -> float:
    """Resting MAP approximation: DBP + (SBP - DBP) / 3, in mmHg.

    ASSUMPTION: the one-third rule is a resting-heart-rate approximation of the
    time-averaged brachial pressure. It is not a retinal pressure.
    """
    sbp = _finite_positive("systolic_mmhg", systolic_mmhg)
    dbp = _finite_positive("diastolic_mmhg", diastolic_mmhg)
    if sbp < dbp:
        raise ValueError(f"systolic ({sbp}) must be >= diastolic ({dbp})")
    return dbp + (sbp - dbp) / 3.0


def mean_ocular_perfusion_pressure(
    map_mmhg: float, iop_mmhg: float, hydrostatic_factor: float = 2.0 / 3.0
) -> float:
    """Mean ocular perfusion pressure, MOPP = factor * MAP - IOP, in mmHg.

    The default factor 2/3 is the commonly used seated-position correction for
    the height difference between the brachial artery and the eye.
    ASSUMPTION: an index of perfusion, not the pressure at the central retinal
    artery, and it ignores autoregulation.
    """
    map_mmhg = _finite_positive("map_mmhg", map_mmhg)
    iop = float(iop_mmhg)
    if not math.isfinite(iop) or iop < 0:
        raise ValueError(f"iop_mmhg must be finite and >= 0, got {iop_mmhg!r}")
    factor = _finite_positive("hydrostatic_factor", hydrostatic_factor)
    mopp = factor * map_mmhg - iop
    if mopp <= 0:
        raise ValueError(
            f"non-positive perfusion pressure ({mopp:.2f} mmHg) from MAP={map_mmhg}, IOP={iop}"
        )
    return mopp


def pixel_size_um_from_disc(disc_diameter_px: float, assumed_disc_diameter_um: float) -> float:
    """Micrometres per pixel, using the optic disc as a size reference.

    The physical disc diameter is NOT observable in a fundus photograph; it is
    an assumed population value with real inter-individual spread, so callers
    should treat it (and therefore the returned scale) as an uncertain input.
    Without a disc measurement there is no physical scale: results must then be
    reported in pixel units or as scale-free (relative) quantities.
    """
    px = _finite_positive("disc_diameter_px", disc_diameter_px)
    um = _finite_positive("assumed_disc_diameter_um", assumed_disc_diameter_um)
    return um / px
