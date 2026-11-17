"""CIE XYZ tristimulus integration and chromaticity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from chromatone._mathutil import as_vec3, trapz
from chromatone.cmf import CMF, cie_1931_2deg, validate_cmf_coverage
from chromatone.illuminant import get_illuminant
from chromatone.spd import SPD


class XYZError(ValueError):
    """Raised for invalid XYZ / chromaticity operations."""


@dataclass(frozen=True)
class XYZ:
    """CIE XYZ tristimulus triple.

    Components are typically relative (Y of the reference white ≈ 100 or 1).
    """

    X: float
    Y: float
    Z: float

    def __post_init__(self) -> None:
        for name, value in (("X", self.X), ("Y", self.Y), ("Z", self.Z)):
            if not math.isfinite(value):
                raise XYZError(f"non-finite {name}: {value}")

    def __iter__(self):
        yield self.X
        yield self.Y
        yield self.Z

    def __repr__(self) -> str:
        return f"XYZ(X={self.X:.6g}, Y={self.Y:.6g}, Z={self.Z:.6g})"

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.X, self.Y, self.Z)

    def scale(self, factor: float) -> "XYZ":
        return XYZ(self.X * factor, self.Y * factor, self.Z * factor)

    def normalize_y(self, target_y: float = 100.0) -> "XYZ":
        """Scale so that Y equals ``target_y``."""
        if abs(self.Y) < 1e-15:
            raise XYZError("cannot normalize XYZ with Y ≈ 0")
        return self.scale(target_y / self.Y)

    @property
    def sum(self) -> float:
        return self.X + self.Y + self.Z

    def chromaticity(self) -> "xyY":
        """Return xyY chromaticity (Y preserved)."""
        s = self.sum
        if abs(s) < 1e-15:
            raise XYZError("cannot compute chromaticity of zero XYZ")
        return xyY(self.X / s, self.Y / s, self.Y)

    def to_xy(self) -> tuple[float, float]:
        c = self.chromaticity()
        return (c.x, c.y)


@dataclass(frozen=True)
class xyY:
    """CIE xyY chromaticity + luminance."""

    x: float
    y: float
    Y: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("Y", self.Y)):
            if not math.isfinite(value):
                raise XYZError(f"non-finite {name}: {value}")

    def __repr__(self) -> str:
        return f"xyY(x={self.x:.6g}, y={self.y:.6g}, Y={self.Y:.6g})"

    def to_xyz(self) -> XYZ:
        if abs(self.y) < 1e-15:
            raise XYZError("cannot convert xyY with y ≈ 0")
        X = self.x * self.Y / self.y
        Z = (1.0 - self.x - self.y) * self.Y / self.y
        return XYZ(X, self.Y, Z)


def _integrate_component(
    spd_values: Sequence[float],
    cmf_values: Sequence[float],
    wavelengths: Sequence[float],
) -> float:
    product = [s * c for s, c in zip(spd_values, cmf_values)]
    return trapz(product, wavelengths)


def spd_to_xyz(
    spd: SPD,
    *,
    cmf: Optional[CMF] = None,
    illuminant: Optional[SPD] = None,
    normalize_y: Optional[float] = 100.0,
) -> XYZ:
    """Compute CIE XYZ for an SPD (optionally as reflectance under an illuminant).

    Parameters
    ----------
    spd:
        Spectral power distribution. Interpreted as a light source unless
        ``illuminant`` is provided, in which case ``spd`` is treated as a
        reflectance / transmittance factor in [0, 1].
    cmf:
        Colour-matching functions (default: CIE 1931 2°).
    illuminant:
        Optional illuminant SPD. When set, the product illuminant×spd is
        integrated and normalized so the illuminant alone yields Y =
        ``normalize_y``.
    normalize_y:
        If not None, scale the result so a perfect reflector (or the bare
        illuminant / source) has this Y. Pass ``None`` to return raw
        integrals.
    """
    table = cmf or cie_1931_2deg()
    validate_cmf_coverage(spd, table)

    if illuminant is not None:
        validate_cmf_coverage(illuminant, table)
        # Align reflectance and illuminant, then form stimulus.
        refl, illum = spd.align_with(illuminant)
        stimulus = SPD(
            refl.wavelengths,
            tuple(r * i for r, i in zip(refl.values, illum.values)),
            name="stimulus",
        )
        white = illum
    else:
        stimulus = spd
        white = spd

    cmf_aligned = table.align_to_spd(stimulus)
    wavelengths = stimulus.wavelengths
    X = _integrate_component(stimulus.values, cmf_aligned.x_bar, wavelengths)
    Y = _integrate_component(stimulus.values, cmf_aligned.y_bar, wavelengths)
    Z = _integrate_component(stimulus.values, cmf_aligned.z_bar, wavelengths)
    result = XYZ(X, Y, Z)

    if normalize_y is not None:
        # Normalize against the white (illuminant or source itself).
        cmf_w = table.align_to_spd(white)
        Y_n = _integrate_component(white.values, cmf_w.y_bar, white.wavelengths)
        if abs(Y_n) < 1e-15:
            raise XYZError("reference white Y integral is ~0")
        result = result.scale(normalize_y / Y_n)
    return result


def white_point(
    illuminant: Union[str, SPD] = "D65",
    *,
    cmf: Optional[CMF] = None,
    normalize_y: float = 100.0,
) -> XYZ:
    """XYZ of a standard or custom illuminant (perfect reflecting diffuser)."""
    if isinstance(illuminant, str):
        spd = get_illuminant(illuminant)
    else:
        spd = illuminant
    return spd_to_xyz(spd, cmf=cmf, normalize_y=normalize_y)


# Common white points (CIE 1931 2°, Y=100), computed at import for convenience
# via lazy cache to keep import light.
_WHITE_POINT_CACHE: dict[str, XYZ] = {}


def cached_white_point(name: str = "D65") -> XYZ:
    """Memoized white-point lookup for standard illuminants."""
    key = name.strip().upper()
    if key not in _WHITE_POINT_CACHE:
        _WHITE_POINT_CACHE[key] = white_point(key)
    return _WHITE_POINT_CACHE[key]


def xyz_from_sequence(values: Sequence[float]) -> XYZ:
    """Build XYZ from a length-3 sequence."""
    v = as_vec3(values)
    return XYZ(v[0], v[1], v[2])


def chromaticity_distance(a: XYZ, b: XYZ) -> float:
    """Euclidean distance in xy chromaticity space."""
    ax, ay = a.to_xy()
    bx, by = b.to_xy()
    return math.hypot(ax - bx, ay - by)
