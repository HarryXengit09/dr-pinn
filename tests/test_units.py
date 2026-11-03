import math

import pytest

from drpinn.hemodynamics import units as u


def test_pressure_conversion_roundtrip():
    assert u.mmhg_to_pa(1.0) == pytest.approx(133.322387415)
    assert u.pa_to_mmhg(u.mmhg_to_pa(47.5)) == pytest.approx(47.5)


def test_viscosity_shear_and_flow_conversions():
    assert u.cp_to_pa_s(3.5) == pytest.approx(3.5e-3)
    assert u.pa_to_dyn_per_cm2(1.0) == pytest.approx(10.0)
    assert u.m3_s_to_ul_per_min(1e-9) == pytest.approx(60.0)  # 1 uL/s
    assert u.um_to_m(u.m_to_um(2.5e-5)) == pytest.approx(2.5e-5)


def test_mean_arterial_pressure():
    assert u.mean_arterial_pressure(120, 80) == pytest.approx(80 + 40 / 3)


def test_mean_ocular_perfusion_pressure():
    map_ = u.mean_arterial_pressure(120, 80)
    assert u.mean_ocular_perfusion_pressure(map_, 15) == pytest.approx(2 / 3 * map_ - 15)


@pytest.mark.parametrize("sbp,dbp", [(80, 120), (0, 0), (-1, -5), (math.nan, 80), (math.inf, 80)])
def test_invalid_blood_pressure_rejected(sbp, dbp):
    with pytest.raises(ValueError):
        u.mean_arterial_pressure(sbp, dbp)


def test_nonpositive_perfusion_rejected():
    with pytest.raises(ValueError):
        u.mean_ocular_perfusion_pressure(80, 60)
    with pytest.raises(ValueError):
        u.mean_ocular_perfusion_pressure(90, -1)


def test_pixel_size_from_disc():
    assert u.pixel_size_um_from_disc(300, 1800) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        u.pixel_size_um_from_disc(0, 1800)
