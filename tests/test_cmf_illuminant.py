"""Tests for CMF tables and standard illuminants."""

from __future__ import annotations

import pytest

from chromatone.cmf import CMFError, cie_1931_2deg, load_cmf, validate_cmf_coverage
from chromatone.illuminant import (
    IlluminantError,
    blackbody_spd,
    daylight_locus_xy,
    get_illuminant,
    illuminant_description,
    list_illuminants,
    resolve_illuminant_name,
)
from chromatone.spd import SPD, SPDError


def test_cie1931_shape_and_peak():
    cmf = cie_1931_2deg()
    assert len(cmf) == 81
    assert cmf.start_nm == 380.0
    assert cmf.stop_nm == 780.0
    # y_bar peaks near 555 nm (photopic V(λ))
    idx = cmf.wavelengths.index(555.0)
    assert cmf.y_bar[idx] == max(cmf.y_bar)
    assert cmf.y_bar[idx] == pytest.approx(1.0000, abs=1e-3)


def test_load_cmf_aliases():
    a = load_cmf("CIE1931_2")
    b = load_cmf("2deg")
    assert a.x_bar == b.x_bar
    with pytest.raises(CMFError):
        load_cmf("CIE2015")


def test_cmf_resample_and_sample():
    cmf = cie_1931_2deg()
    x, y, z = cmf.sample(555.0)
    assert y == pytest.approx(1.0, abs=1e-3)
    rs = cmf.resample([400.0, 500.0, 600.0, 700.0])
    assert len(rs) == 4
    assert rs.sample(500.0)[1] == pytest.approx(cmf.sample(500.0)[1])


def test_illuminant_list_and_aliases():
    names = list_illuminants()
    assert "D65" in names and "A" in names and "FL2" in names
    assert resolve_illuminant_name("F2") == "FL2"
    assert resolve_illuminant_name("f11") == "FL11"
    with pytest.raises(IlluminantError):
        get_illuminant("Z99")


def test_d65_normalized_at_560(d65):
    assert d65.sample(560.0) == pytest.approx(100.0)
    assert d65.name == "D65"


def test_illuminant_a_matches_blackbody():
    tabulated = get_illuminant("A")
    bb = blackbody_spd(2856.0)
    # Relative shapes should be close across the visible band.
    for w in (400.0, 500.0, 560.0, 700.0):
        assert tabulated.sample(w) == pytest.approx(bb.sample(w), rel=0.05)


def test_blackbody_rejects_bad_temp():
    with pytest.raises(IlluminantError):
        blackbody_spd(-10)


def test_daylight_locus_d65_region():
    x, y = daylight_locus_xy(6504)
    # D65 chromaticity ≈ (0.3127, 0.3290)
    assert x == pytest.approx(0.3127, abs=0.005)
    assert y == pytest.approx(0.3290, abs=0.01)


def test_descriptions_nonempty():
    for name in ("A", "D65", "F7", "E"):
        assert len(illuminant_description(name)) > 10


def test_validate_coverage():
    cmf = cie_1931_2deg()
    ok = SPD((400.0, 500.0, 600.0), (1.0, 1.0, 1.0))
    validate_cmf_coverage(ok, cmf)
    bad = SPD((200.0, 250.0), (1.0, 1.0))
    with pytest.raises(SPDError):
        validate_cmf_coverage(bad, cmf)


def test_fluorescent_positive_energy():
    for name in ("F2", "F7", "F11"):
        spd = get_illuminant(name)
        assert spd.integral() > 0
        assert max(spd.values) > 0
