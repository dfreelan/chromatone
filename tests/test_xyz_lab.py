"""XYZ / Lab conversions and round-trip tests."""

from __future__ import annotations

import math
import pytest

from chromatone.illuminant import get_illuminant
from chromatone.lab import Lab, LCh, lab_to_xyz, xyz_to_lab, xyz_to_lch
from chromatone.spd import SPD
from chromatone.xyz import XYZ, spd_to_xyz, white_point, xyY


def test_d65_white_point_chromaticity(d65):
    xyz = spd_to_xyz(d65)
    assert xyz.Y == pytest.approx(100.0)
    x, y = xyz.to_xy()
    # CIE D65 2°: x≈0.3127, y≈0.3290
    assert x == pytest.approx(0.3127, abs=0.002)
    assert y == pytest.approx(0.3290, abs=0.002)


def test_illuminant_a_chromaticity(illuminant_a):
    xyz = spd_to_xyz(illuminant_a)
    x, y = xyz.to_xy()
    # CIE A: x≈0.4476, y≈0.4074
    assert x == pytest.approx(0.4476, abs=0.003)
    assert y == pytest.approx(0.4074, abs=0.003)


def test_equal_energy_near_e():
    xyz = white_point("E")
    x, y = xyz.to_xy()
    assert x == pytest.approx(1 / 3, abs=0.01)
    assert y == pytest.approx(1 / 3, abs=0.01)


def test_perfect_reflector_lab(d65):
    refl = SPD.ones()
    xyz = spd_to_xyz(refl, illuminant=d65)
    lab = xyz_to_lab(xyz, "D65")
    assert lab.L == pytest.approx(100.0, abs=0.05)
    assert lab.a == pytest.approx(0.0, abs=0.5)
    assert lab.b == pytest.approx(0.0, abs=0.5)


def test_grey_reflectance_lstar(d65, grey_reflectance):
    xyz = spd_to_xyz(grey_reflectance, illuminant=d65)
    lab = xyz_to_lab(xyz, "D65")
    # Y=50 → L* ≈ 76.07
    assert lab.L == pytest.approx(76.0693, abs=0.5)


def test_xyz_lab_roundtrip_grid():
    white = white_point("D65")
    samples = [
        XYZ(white.X, white.Y, white.Z),
        XYZ(white.X * 0.2, white.Y * 0.2, white.Z * 0.2),
        XYZ(40.0, 20.0, 10.0),
        XYZ(10.0, 50.0, 30.0),
        XYZ(5.0, 5.0, 5.0),
        XYZ(80.0, 90.0, 60.0),
    ]
    for xyz in samples:
        lab = xyz_to_lab(xyz, white)
        back = lab_to_xyz(lab, white)
        assert back.X == pytest.approx(xyz.X, rel=1e-9, abs=1e-9)
        assert back.Y == pytest.approx(xyz.Y, rel=1e-9, abs=1e-9)
        assert back.Z == pytest.approx(xyz.Z, rel=1e-9, abs=1e-9)


def test_lab_lch_roundtrip():
    lab = Lab(60.0, -20.0, 35.0)
    lch = lab.to_lch()
    assert lch.C == pytest.approx(math.hypot(-20.0, 35.0))
    back = lch.to_lab()
    assert back.L == pytest.approx(lab.L)
    assert back.a == pytest.approx(lab.a)
    assert back.b == pytest.approx(lab.b)


def test_xyY_roundtrip():
    xyz = white_point("D65")
    xyy = xyz.chromaticity()
    back = xyy.to_xyz()
    assert back.X == pytest.approx(xyz.X)
    assert back.Y == pytest.approx(xyz.Y)
    assert back.Z == pytest.approx(xyz.Z)


def test_xyY_construct():
    c = xyY(0.3127, 0.3290, 100.0)
    xyz = c.to_xyz()
    assert xyz.Y == 100.0
    assert xyz.to_xy()[0] == pytest.approx(0.3127)


def test_raw_vs_normalized(d65):
    raw = spd_to_xyz(d65, normalize_y=None)
    norm = spd_to_xyz(d65, normalize_y=100.0)
    scale = 100.0 / raw.Y
    assert norm.X == pytest.approx(raw.X * scale)
    assert norm.Y == pytest.approx(100.0)


def test_spd_to_xyz_under_a():
    refl = SPD.ones().scale(0.8)
    xyz = spd_to_xyz(refl, illuminant=get_illuminant("A"))
    assert xyz.Y == pytest.approx(80.0, abs=0.1)


def test_normalize_y_on_xyz():
    xyz = XYZ(1.0, 2.0, 3.0).normalize_y(100)
    assert xyz.Y == pytest.approx(100.0)
    assert xyz.X == pytest.approx(50.0)


def test_xyz_to_lch_path():
    xyz = white_point("D50")
    lch = xyz_to_lch(xyz, "D50")
    assert isinstance(lch, LCh)
    assert lch.L == pytest.approx(100.0, abs=0.05)
