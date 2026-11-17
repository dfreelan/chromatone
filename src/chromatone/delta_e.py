"""Colour-difference formulae: CIE76, CIE94, CIEDE2000."""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

from chromatone.lab import Lab, xyz_to_lab
from chromatone.xyz import XYZ


class DeltaEError(ValueError):
    """Raised for invalid delta-E parameters."""


DeltaEMethod = Literal["cie76", "cie94", "cie00", "ciede2000", "76", "94", "00"]


def delta_e_cie76(lab1: Lab, lab2: Lab) -> float:
    """CIE 1976 ΔE*ab — Euclidean distance in L*a*b*."""
    return math.sqrt(
        (lab1.L - lab2.L) ** 2
        + (lab1.a - lab2.a) ** 2
        + (lab1.b - lab2.b) ** 2
    )


def delta_e_cie94(
    lab1: Lab,
    lab2: Lab,
    *,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
    application: Literal["graphic_arts", "textiles"] = "graphic_arts",
) -> float:
    """CIE 1994 colour difference ΔE*94.

    Default parametric factors are for graphic arts (kL=1, K1=0.045,
    K2=0.015). Pass ``application="textiles"`` for kL=2, K1=0.048, K2=0.014.
    """
    if application == "graphic_arts":
        K1, K2 = 0.045, 0.015
        if kL == 1.0 and kC == 1.0 and kH == 1.0:
            pass  # defaults already set
    elif application == "textiles":
        K1, K2 = 0.048, 0.014
        if kL == 1.0:
            kL = 2.0
    else:
        raise DeltaEError(f"unknown application {application!r}")

    L1, a1, b1 = lab1.L, lab1.a, lab1.b
    L2, a2, b2 = lab2.L, lab2.a, lab2.b

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    dL = L1 - L2
    dC = C1 - C2
    dA = a1 - a2
    dB = b1 - b2
    dH_sq = dA * dA + dB * dB - dC * dC
    dH = math.sqrt(max(0.0, dH_sq))

    SL = 1.0
    SC = 1.0 + K1 * C1
    SH = 1.0 + K2 * C1

    return math.sqrt(
        (dL / (kL * SL)) ** 2
        + (dC / (kC * SC)) ** 2
        + (dH / (kH * SH)) ** 2
    )


def _hp_deg(a: float, b: float) -> float:
    """Hue angle in degrees, atan2(b, a), range [0, 360)."""
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return 0.0
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return h


def delta_e_ciede2000(
    lab1: Lab,
    lab2: Lab,
    *,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """CIEDE2000 colour difference ΔE00 (Sharma et al. 2005).

    Parametric factors ``kL``, ``kC``, ``kH`` default to 1 (graphic arts).
    """
    L1, a1, b1 = lab1.L, lab1.a, lab1.b
    L2, a2, b2 = lab2.L, lab2.a, lab2.b

    C1_ab = math.hypot(a1, b1)
    C2_ab = math.hypot(a2, b2)
    C_bar_ab = 0.5 * (C1_ab + C2_ab)

    C_bar_ab7 = C_bar_ab ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar_ab7 / (C_bar_ab7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    h1p = _hp_deg(a1p, b1)
    h2p = _hp_deg(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        dhp = dh

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    L_bar_p = 0.5 * (L1 + L2)
    C_bar_p = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        h_bar_p = h1p + h2p
    else:
        h_sum = h1p + h2p
        h_diff = abs(h1p - h2p)
        if h_diff <= 180.0:
            h_bar_p = 0.5 * h_sum
        elif h_sum < 360.0:
            h_bar_p = 0.5 * (h_sum + 360.0)
        else:
            h_bar_p = 0.5 * (h_sum - 360.0)

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    C_bar_p7 = C_bar_p ** 7
    R_C = 2.0 * math.sqrt(C_bar_p7 / (C_bar_p7 + 25.0 ** 7))
    S_L = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / math.sqrt(
        20.0 + (L_bar_p - 50.0) ** 2
    )
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * d_theta)) * R_C

    dE = math.sqrt(
        (dLp / (kL * S_L)) ** 2
        + (dCp / (kC * S_C)) ** 2
        + (dHp / (kH * S_H)) ** 2
        + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )
    return dE


# Alias matching common naming.
delta_e_cie00 = delta_e_ciede2000


def delta_e(
    lab1: Lab,
    lab2: Lab,
    *,
    method: str = "cie00",
    **kwargs: float,
) -> float:
    """Dispatch colour difference by ``method`` name.

    Supported methods: ``cie76`` / ``76``, ``cie94`` / ``94``,
    ``cie00`` / ``ciede2000`` / ``00``.
    """
    key = method.strip().lower().replace("-", "").replace("_", "").replace("*", "")
    mapping = {
        "cie76": "76",
        "76": "76",
        "deltae76": "76",
        "cie94": "94",
        "94": "94",
        "deltae94": "94",
        "cie00": "00",
        "cie2000": "00",
        "ciede2000": "00",
        "00": "00",
        "de00": "00",
        "deltae00": "00",
    }
    if key not in mapping:
        raise DeltaEError(
            f"unknown delta-E method {method!r}; use cie76, cie94, or cie00"
        )
    kind = mapping[key]
    if kind == "76":
        return delta_e_cie76(lab1, lab2)
    if kind == "94":
        app = kwargs.pop("application", None)
        params = {k: float(v) for k, v in kwargs.items() if k in {"kL", "kC", "kH"}}
        if app is not None:
            return delta_e_cie94(lab1, lab2, application=str(app), **params)  # type: ignore[arg-type]
        return delta_e_cie94(lab1, lab2, **params)
    params = {k: float(v) for k, v in kwargs.items() if k in {"kL", "kC", "kH"}}
    return delta_e_ciede2000(lab1, lab2, **params)


def delta_e_xyz(
    xyz1: XYZ,
    xyz2: XYZ,
    *,
    white: Union[str, XYZ] = "D65",
    method: str = "cie00",
    **kwargs: float,
) -> float:
    """Compute delta-E after converting XYZ pairs to Lab under ``white``."""
    lab1 = xyz_to_lab(xyz1, white)
    lab2 = xyz_to_lab(xyz2, white)
    return delta_e(lab1, lab2, method=method, **kwargs)


def pairwise_delta_e(
    colours: list[Lab],
    *,
    method: str = "cie00",
) -> list[list[float]]:
    """Return an n×n matrix of pairwise colour differences."""
    n = len(colours)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = delta_e(colours[i], colours[j], method=method)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix
