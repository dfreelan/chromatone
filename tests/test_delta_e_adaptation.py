"""Delta-E formulae and chromatic adaptation tests."""

from __future__ import annotations

import pytest

from chromatone.adaptation import (
    adapt_xyz,
    adapt_xyz_bradford,
    available_methods,
    build_transform,
    chromatic_adaptation_matrix,
)
from chromatone.delta_e import (
    delta_e,
    delta_e_cie76,
    delta_e_cie94,
    delta_e_ciede2000,
    delta_e_xyz,
)
from chromatone.lab import Lab, xyz_to_lab
from chromatone.xyz import XYZ, white_point


def test_cie76_identical_zero():
    lab = Lab(50, 10, -5)
    assert delta_e_cie76(lab, lab) == 0.0


def test_cie76_known():
    # ΔE76 = 1 along L*
    assert delta_e_cie76(Lab(50, 0, 0), Lab(51, 0, 0)) == pytest.approx(1.0)


def test_cie94_positive():
    d = delta_e_cie94(Lab(50, 2.5, 0), Lab(50, 0, -2.5))
    assert d > 0
    d_tex = delta_e_cie94(
        Lab(50, 2.5, 0), Lab(50, 0, -2.5), application="textiles"
    )
    assert d_tex > 0



# Sharma et al. (2005) CIEDE2000 reference pairs: (L1,a1,b1, L2,a2,b2, dE00)
SHARMA_RAW = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]

def test_ciede2000_sharma_cases():
    for L1, a1, b1, L2, a2, b2, expected in SHARMA_RAW:
        got = delta_e_ciede2000(Lab(L1, a1, b1), Lab(L2, a2, b2))
        assert got == pytest.approx(expected, abs=0.0001)


def test_delta_e_dispatcher():
    a, b = Lab(50, 0, 0), Lab(50, 3, 4)
    assert delta_e(a, b, method="cie76") == pytest.approx(5.0)
    assert delta_e(a, b, method="94") > 0
    assert delta_e(a, b, method="cie00") > 0


def test_delta_e_xyz_path():
    w = white_point("D65")
    d = delta_e_xyz(w, w.scale(0.5), white="D65", method="cie76")
    assert d > 0


def test_bradford_adapts_a_to_d65():
    src = white_point("A")
    dst_white = white_point("D65")
    adapted = adapt_xyz_bradford(src, "A", "D65")
    # Adapted source white should land on destination white.
    assert adapted.X == pytest.approx(dst_white.X, rel=1e-6)
    assert adapted.Y == pytest.approx(dst_white.Y, rel=1e-6)
    assert adapted.Z == pytest.approx(dst_white.Z, rel=1e-6)


def test_adaptation_methods_roundtrip_white():
    src = white_point("D50")
    for method in available_methods():
        out = adapt_xyz(src, "D50", "D65", method=method)
        target = white_point("D65")
        assert out.X == pytest.approx(target.X, rel=1e-5)
        assert out.Y == pytest.approx(target.Y, rel=1e-5)
        assert out.Z == pytest.approx(target.Z, rel=1e-5)


def test_build_transform_reusable():
    xf = build_transform("A", "D65", method="bradford")
    c = XYZ(50, 40, 20)
    assert xf.apply(c).as_tuple() == adapt_xyz(c, "A", "D65").as_tuple()


def test_chromatic_adaptation_matrix_shape():
    m = chromatic_adaptation_matrix("A", "D65")
    assert len(m) == 3 and len(m[0]) == 3
    assert m[0][0] != 0.0


def test_lab_delta_consistency_after_adaptation():
    # A neutral under A, adapted to D65, should be near Lab neutral under D65.
    xyz_a = white_point("A")
    adapted = adapt_xyz_bradford(xyz_a, "A", "D65")
    lab = xyz_to_lab(adapted, "D65")
    assert lab.L == pytest.approx(100.0, abs=0.05)
    assert abs(lab.a) < 0.05
    assert abs(lab.b) < 0.05
