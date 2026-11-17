"""Chromatic adaptation transforms (Bradford, CAT02, von Kries)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from chromatone._mathutil import Mat3, Vec3, as_vec3, diag3, inv3, matmul, matvec
from chromatone.xyz import XYZ, cached_white_point


class AdaptationError(ValueError):
    """Raised for singular or unsupported adaptation transforms."""


# Bradford cone-response matrix (CIECAM97s / Bradford).
BRADFORD_M: Mat3 = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)

# Hunt-Pointer-Estevez (von Kries / "XYZ scaling" often uses identity;
# HPE is a common cone fundamentals matrix).
HPE_M: Mat3 = (
    (0.38971, 0.68898, -0.07868),
    (-0.22981, 1.18340, 0.04641),
    (0.00000, 0.00000, 1.00000),
)

# CAT02 matrix (CIE CAM02).
CAT02_M: Mat3 = (
    (0.7328, 0.4296, -0.1624),
    (-0.7036, 1.6975, 0.0061),
    (0.0030, 0.0136, 0.9834),
)

# Identity (pure XYZ scaling / "wrong von Kries").
IDENTITY_M: Mat3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

CATMethod = Literal["bradford", "von_kries", "cat02", "xyz_scaling"]

_METHOD_MATRICES: dict[str, Mat3] = {
    "bradford": BRADFORD_M,
    "von_kries": HPE_M,
    "cat02": CAT02_M,
    "xyz_scaling": IDENTITY_M,
}


@dataclass(frozen=True)
class AdaptationTransform:
    """Precomputed chromatic adaptation matrix (source → destination white)."""

    matrix: Mat3
    source_white: XYZ
    destination_white: XYZ
    method: str

    def apply(self, xyz: XYZ) -> XYZ:
        """Adapt a single XYZ colour."""
        out = matvec(self.matrix, xyz.as_tuple())
        return XYZ(out[0], out[1], out[2])

    def apply_many(self, colours: list[XYZ]) -> list[XYZ]:
        return [self.apply(c) for c in colours]

    def __repr__(self) -> str:
        return (
            f"AdaptationTransform(method={self.method!r}, "
            f"src=XYZ{self.source_white.as_tuple()}, "
            f"dst=XYZ{self.destination_white.as_tuple()})"
        )


def _resolve_white(white: Union[str, XYZ]) -> XYZ:
    if isinstance(white, str):
        return cached_white_point(white)
    if isinstance(white, XYZ):
        return white
    raise AdaptationError(f"unsupported white type: {type(white)!r}")


def get_cat_matrix(method: str = "bradford") -> Mat3:
    """Return the cone-response matrix for a named CAT method."""
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bradford": "bradford",
        "vonkries": "von_kries",
        "von_kries": "von_kries",
        "hpe": "von_kries",
        "cat02": "cat02",
        "ciecat02": "cat02",
        "xyz": "xyz_scaling",
        "xyz_scaling": "xyz_scaling",
        "scaling": "xyz_scaling",
    }
    if key not in aliases:
        raise AdaptationError(
            f"unknown CAT method {method!r}; "
            f"choose from: {', '.join(sorted(_METHOD_MATRICES))}"
        )
    return _METHOD_MATRICES[aliases[key]]


def cone_response(xyz: XYZ, method: str = "bradford") -> Vec3:
    """Project XYZ into the cone-response domain of ``method``."""
    m = get_cat_matrix(method)
    return matvec(m, xyz.as_tuple())


def chromatic_adaptation_matrix(
    source_white: Union[str, XYZ],
    destination_white: Union[str, XYZ],
    *,
    method: str = "bradford",
) -> Mat3:
    """Build the 3×3 matrix that adapts XYZ from source to destination white.

    Uses the von Kries-style diagonal gain in cone space::

        M_adapt = M^{-1} · diag(ρ_d/ρ_s, γ_d/γ_s, β_d/β_s) · M
    """
    src = _resolve_white(source_white)
    dst = _resolve_white(destination_white)
    m = get_cat_matrix(method)
    m_inv = inv3(m)
    src_cone = matvec(m, src.as_tuple())
    dst_cone = matvec(m, dst.as_tuple())
    gains: list[float] = []
    for s, d in zip(src_cone, dst_cone):
        if abs(s) < 1e-15:
            raise AdaptationError("source white has a zero cone response")
        gains.append(d / s)
    gain_mat = diag3(gains[0], gains[1], gains[2])
    return matmul(m_inv, matmul(gain_mat, m))


def build_transform(
    source_white: Union[str, XYZ],
    destination_white: Union[str, XYZ],
    *,
    method: str = "bradford",
) -> AdaptationTransform:
    """Return a reusable :class:`AdaptationTransform`."""
    src = _resolve_white(source_white)
    dst = _resolve_white(destination_white)
    key = method.strip().lower().replace("-", "_")
    matrix = chromatic_adaptation_matrix(src, dst, method=method)
    return AdaptationTransform(matrix=matrix, source_white=src, destination_white=dst, method=key)


def adapt_xyz(
    xyz: XYZ,
    source_white: Union[str, XYZ],
    destination_white: Union[str, XYZ],
    *,
    method: str = "bradford",
) -> XYZ:
    """Adapt a single XYZ colour between illuminants."""
    return build_transform(source_white, destination_white, method=method).apply(xyz)


def adapt_xyz_bradford(
    xyz: XYZ,
    source_white: Union[str, XYZ],
    destination_white: Union[str, XYZ],
) -> XYZ:
    """Convenience wrapper for Bradford chromatic adaptation."""
    return adapt_xyz(xyz, source_white, destination_white, method="bradford")



def available_methods() -> list[str]:
    """Return supported CAT method names."""
    return sorted(_METHOD_MATRICES.keys())
