"""Tests for SPD construction, sampling, and arithmetic."""

from __future__ import annotations

import math
import pytest

from chromatone.spd import SPD, SPDError


def test_from_mapping_sorted():
    spd = SPD.from_mapping({500: 1.0, 400: 0.5, 600: 0.2}, name="demo")
    assert spd.wavelengths == (400.0, 500.0, 600.0)
    assert spd.values == (0.5, 1.0, 0.2)
    assert spd.name == "demo"


def test_rejects_non_increasing():
    with pytest.raises(SPDError):
        SPD((400, 400), (1.0, 1.0))
    with pytest.raises(SPDError):
        SPD((500, 400), (1.0, 1.0))


def test_rejects_short():
    with pytest.raises(SPDError):
        SPD((400,), (1.0,))


def test_sample_and_resample():
    spd = SPD((400.0, 500.0, 600.0), (0.0, 1.0, 0.0))
    assert spd.sample(450.0) == pytest.approx(0.5)
    assert spd.sample(350.0) == 0.0
    assert spd.sample(350.0, extrapolate=True) == 0.0
    rs = spd.resample_regular(400, 600, 50)
    assert len(rs) == 5
    assert rs.sample(500.0) == pytest.approx(1.0)


def test_gaussian_peak():
    g = SPD.gaussian(550, 20, amplitude=2.0)
    assert g.sample(550.0) == pytest.approx(2.0)
    assert g.sample(550.0 + 10.0) == pytest.approx(1.0, rel=1e-3)


def test_normalize_peak_and_at():
    spd = SPD((400.0, 560.0, 700.0), (1.0, 2.0, 0.5))
    assert max(spd.normalize_peak(1.0).values) == pytest.approx(1.0)
    n = spd.normalize_at(560.0, 100.0)
    assert n.sample(560.0) == pytest.approx(100.0)


def test_arithmetic_align():
    a = SPD((400.0, 500.0), (1.0, 1.0))
    b = SPD((450.0, 550.0), (1.0, 1.0))
    s = a + b
    assert 450.0 in s.wavelengths
    assert (a * 2).values == (2.0, 2.0)
    assert (a - 0.5).values == (0.5, 0.5)


def test_multiply_reflectance():
    illum = SPD.ones()
    refl = SPD.ones().scale(0.25)
    prod = illum * refl
    assert all(v == pytest.approx(0.25) for v in prod.values)


def test_clip_and_nonneg():
    spd = SPD((400.0, 500.0, 600.0, 700.0), (1.0, -0.5, 0.2, 0.1))
    clipped = spd.clip_range(450, 650)
    assert clipped.start_nm == 500.0
    assert min(spd.nonneg().values) >= 0.0


def test_integral_triangle():
    spd = SPD((0.0 + 400, 500.0, 600.0), (0.0, 1.0, 0.0))
    # Triangle area base 200 height 1 → 100
    assert spd.integral() == pytest.approx(100.0)


def test_to_from_dict_roundtrip():
    spd = SPD.gaussian(600, 30, name="orange")
    restored = SPD.from_dict(spd.to_dict())
    assert restored.wavelengths == spd.wavelengths
    assert restored.values == spd.values
    assert restored.name == "orange"


def test_zeros_ones_regular():
    z = SPD.zeros(step_nm=10)
    assert z.is_regular
    assert z.step_nm == pytest.approx(10.0)
    assert sum(z.values) == 0
    assert all(v == 1.0 for v in SPD.ones().values)


def test_scale_div():
    spd = SPD.ones().scale(10)
    assert (spd / 2).values[0] == pytest.approx(5.0)
    with pytest.raises(SPDError):
        _ = spd / 0.0
