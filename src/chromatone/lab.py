"""CIELAB and CIELCh colour spaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from chromatone.xyz import XYZ, cached_white_point, white_point


class LabError(ValueError):
    """Raised for invalid Lab / LCh operations."""


# CIE standard constants
_DELTA = 6.0 / 29.0
_DELTA_CUBE = _DELTA ** 3
_KAPPA = 24389.0 / 27.0  # (29/3)^3
_EPSILON = _DELTA_CUBE  # 216/24389


def _lab_f(t: float) -> float:
    """CIE f(t) used in L*a*b* (piecewise cube root)."""
    if t > _EPSILON:
        return math.pow(t, 1.0 / 3.0)
    return (t * _KAPPA + 16.0) / 116.0


def _lab_f_inv(t: float) -> float:
    """Inverse of :func:`_lab_f`."""
    t3 = t * t * t
    if t3 > _EPSILON:
        return t3
    return (116.0 * t - 16.0) / _KAPPA


@dataclass(frozen=True)
class Lab:
    """CIE 1976 L*a*b* colour."""

    L: float
    a: float
    b: float

    def __post_init__(self) -> None:
        for name, value in (("L", self.L), ("a", self.a), ("b", self.b)):
            if not math.isfinite(value):
                raise LabError(f"non-finite {name}: {value}")

    def __iter__(self):
        yield self.L
        yield self.a
        yield self.b

    def __repr__(self) -> str:
        return f"Lab(L={self.L:.6g}, a={self.a:.6g}, b={self.b:.6g})"

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.L, self.a, self.b)

    def to_lch(self) -> "LCh":
        """Convert to cylindrical CIELCh."""
        c = math.hypot(self.a, self.b)
        h = math.degrees(math.atan2(self.b, self.a)) % 360.0
        return LCh(self.L, c, h)

    def delta_e76(self, other: "Lab") -> float:
        """CIE76 ΔE*ab (Euclidean distance in Lab)."""
        return math.sqrt(
            (self.L - other.L) ** 2
            + (self.a - other.a) ** 2
            + (self.b - other.b) ** 2
        )


@dataclass(frozen=True)
class LCh:
    """CIE L*C*h° cylindrical representation of Lab."""

    L: float
    C: float
    h: float  # degrees, [0, 360)

    def __post_init__(self) -> None:
        for name, value in (("L", self.L), ("C", self.C), ("h", self.h)):
            if not math.isfinite(value):
                raise LabError(f"non-finite {name}: {value}")
        if self.C < 0:
            raise LabError(f"chroma C must be non-negative, got {self.C}")

    def __repr__(self) -> str:
        return f"LCh(L={self.L:.6g}, C={self.C:.6g}, h={self.h:.6g})"

    def to_lab(self) -> Lab:
        hr = math.radians(self.h)
        return Lab(self.L, self.C * math.cos(hr), self.C * math.sin(hr))


def _resolve_white(white: Union[str, XYZ, None]) -> XYZ:
    if white is None:
        return cached_white_point("D65")
    if isinstance(white, str):
        return cached_white_point(white)
    if isinstance(white, XYZ):
        return white
    raise LabError(f"unsupported white reference type: {type(white)!r}")


def xyz_to_lab(xyz: XYZ, white: Union[str, XYZ, None] = "D65") -> Lab:
    """Convert CIE XYZ to CIELAB relative to ``white``."""
    wn = _resolve_white(white)
    if wn.X <= 0 or wn.Y <= 0 or wn.Z <= 0:
        raise LabError("reference white must have positive XYZ")
    xr = xyz.X / wn.X
    yr = xyz.Y / wn.Y
    zr = xyz.Z / wn.Z
    fx = _lab_f(xr)
    fy = _lab_f(yr)
    fz = _lab_f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return Lab(L, a, b)


def lab_to_xyz(lab: Lab, white: Union[str, XYZ, None] = "D65") -> XYZ:
    """Convert CIELAB to CIE XYZ relative to ``white``."""
    wn = _resolve_white(white)
    fy = (lab.L + 16.0) / 116.0
    fx = fy + lab.a / 500.0
    fz = fy - lab.b / 200.0
    xr = _lab_f_inv(fx)
    yr = _lab_f_inv(fy)
    zr = _lab_f_inv(fz)
    return XYZ(xr * wn.X, yr * wn.Y, zr * wn.Z)


def xyz_to_lch(xyz: XYZ, white: Union[str, XYZ, None] = "D65") -> LCh:
    return xyz_to_lab(xyz, white).to_lch()


def lch_to_xyz(lch: LCh, white: Union[str, XYZ, None] = "D65") -> XYZ:
    return lab_to_xyz(lch.to_lab(), white)


def lab_from_sequence(values: Sequence[float]) -> Lab:
    if len(values) != 3:
        raise LabError("expected 3 Lab components")
    return Lab(float(values[0]), float(values[1]), float(values[2]))


def is_lab_in_gamut_approx(lab: Lab) -> bool:
    """Heuristic check: L* in [0, 100] and finite a*, b*."""
    return 0.0 <= lab.L <= 100.0 and math.isfinite(lab.a) and math.isfinite(lab.b)


def lab_to_string(lab: Lab, *, precision: int = 3) -> str:
    fmt = f"{{:.{precision}f}}"
    return f"L*={fmt.format(lab.L)} a*={fmt.format(lab.a)} b*={fmt.format(lab.b)}"
