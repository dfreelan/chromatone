"""CIE standard illuminants and Planckian (blackbody) spectra."""

from __future__ import annotations

import math
from typing import Optional

from chromatone.data.illuminants import ALIASES, ILLUMINANTS, WAVELENGTHS_NM
from chromatone.spd import (
    DEFAULT_WAVELENGTH_MAX_NM,
    DEFAULT_WAVELENGTH_MIN_NM,
    DEFAULT_WAVELENGTH_STEP_NM,
    SPD,
    SPDError,
)

# Second radiation constant in nm·K (CIE convention uses 1.4388e7).
_C2_NM_K = 1.4388e7

# Nominal CCT values for common illuminants (informational).
ILLUMINANT_CCT_K: dict[str, float] = {
    "A": 2856.0,
    "C": 6774.0,
    "D50": 5003.0,
    "D55": 5503.0,
    "D65": 6504.0,
    "D75": 7504.0,
    "E": 5454.0,  # equal-energy; CCT approx.
    "FL1": 6430.0,
    "FL2": 4230.0,
    "FL3": 3450.0,
    "FL4": 2940.0,
    "FL5": 6350.0,
    "FL6": 4150.0,
    "FL7": 6500.0,
    "FL8": 5000.0,
    "FL9": 4150.0,
    "FL10": 5000.0,
    "FL11": 4000.0,
    "FL12": 3000.0,
}


class IlluminantError(ValueError):
    """Raised for unknown illuminants or invalid CCT requests."""


def list_illuminants() -> list[str]:
    """Return sorted canonical illuminant names available in package data."""
    return sorted(ILLUMINANTS.keys())


def resolve_illuminant_name(name: str) -> str:
    """Map aliases (e.g. ``F2``) to canonical keys (e.g. ``FL2``)."""
    key = name.strip().upper().replace(" ", "")
    if key.startswith("ILLUMINANT"):
        key = key[len("ILLUMINANT") :]
    if key in ILLUMINANTS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    # Accept lowercase f-series like "f2"
    lower_aliases = {k.upper(): v for k, v in ALIASES.items()}
    if key in lower_aliases:
        return lower_aliases[key]
    raise IlluminantError(
        f"unknown illuminant {name!r}; available: {', '.join(list_illuminants())}"
    )


def get_illuminant(name: str = "D65", *, normalize_560: bool = False) -> SPD:
    """Return a standard illuminant as an :class:`~chromatone.spd.SPD`.

    Parameters
    ----------
    name:
        Illuminant identifier (``D65``, ``A``, ``F2`` / ``FL2``, …).
    normalize_560:
        If True, rescale so the value at 560 nm equals 100 (CIE daylight
        convention). Most tabulated D-series data already satisfy this.
    """
    key = resolve_illuminant_name(name)
    values = ILLUMINANTS[key]
    wavelengths = tuple(float(w) for w in WAVELENGTHS_NM)
    spd = SPD(wavelengths, values, name=key)
    if normalize_560:
        spd = spd.normalize_at(560.0, 100.0, name=key)
    return spd


def illuminant_description(name: str) -> str:
    """Short description of a standard illuminant."""
    key = resolve_illuminant_name(name)
    descriptions = {
        "A": "CIE standard illuminant A — Planckian radiator at 2856 K (incandescent)",
        "C": "CIE illuminant C — average daylight (obsolete; prefer D65)",
        "D50": "CIE illuminant D50 — horizon daylight ≈ 5003 K",
        "D55": "CIE illuminant D55 — mid-morning daylight ≈ 5503 K",
        "D65": "CIE standard illuminant D65 — average noon daylight ≈ 6504 K",
        "D75": "CIE illuminant D75 — north-sky daylight ≈ 7504 K",
        "E": "CIE illuminant E — equal-energy spectrum",
        "FL1": "CIE fluorescent F1 — daylight fluorescent",
        "FL2": "CIE fluorescent F2 — cool white fluorescent",
        "FL3": "CIE fluorescent F3 — white fluorescent",
        "FL4": "CIE fluorescent F4 — warm white fluorescent",
        "FL5": "CIE fluorescent F5 — daylight fluorescent",
        "FL6": "CIE fluorescent F6 — light white fluorescent",
        "FL7": "CIE fluorescent F7 — D65 simulator broadband fluorescent",
        "FL8": "CIE fluorescent F8 — D50 simulator broadband fluorescent",
        "FL9": "CIE fluorescent F9 — cool white deluxe fluorescent",
        "FL10": "CIE fluorescent F10 — three-band fluorescent (5000 K)",
        "FL11": "CIE fluorescent F11 — three-band fluorescent (4000 K)",
        "FL12": "CIE fluorescent F12 — three-band fluorescent (3000 K)",
    }
    base = descriptions.get(key, f"CIE illuminant {key}")
    cct = ILLUMINANT_CCT_K.get(key)
    if cct is not None and key not in ("A", "D50", "D55", "D65", "D75"):
        return f"{base} (nominal CCT ≈ {cct:.0f} K)"
    return base


def blackbody_spd(
    temperature_k: float,
    *,
    start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
    stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
    step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
    normalize_560: bool = True,
    name: Optional[str] = None,
) -> SPD:
    """Planckian radiator relative SPD.

    Uses the CIE form of Planck's law with ``c2 = 1.4388×10^7 nm·K``.
    When ``normalize_560`` is True (default), the spectrum is scaled to
    100 at 560 nm, matching CIE illuminant A convention.
    """
    if temperature_k <= 0:
        raise IlluminantError("temperature_k must be positive")
    if stop_nm <= start_nm:
        raise IlluminantError("stop_nm must exceed start_nm")
    if step_nm <= 0:
        raise IlluminantError("step_nm must be positive")

    wavelengths: list[float] = []
    values: list[float] = []
    n = int(round((stop_nm - start_nm) / step_nm))
    for i in range(n + 1):
        lam = start_nm + i * step_nm
        if i == n:
            lam = stop_nm
        # Relative spectral concentration: (1/λ^5) / (exp(c2/(λ T)) - 1)
        # Avoid overflow for very short λ / low T.
        exponent = _C2_NM_K / (lam * temperature_k)
        if exponent > 700:
            spectral = 0.0
        else:
            spectral = (1.0 / (lam ** 5)) / (math.exp(exponent) - 1.0)
        wavelengths.append(lam)
        values.append(spectral)

    label = name or f"blackbody({temperature_k:.0f}K)"
    spd = SPD(tuple(wavelengths), tuple(values), name=label)
    if normalize_560:
        # Evaluate continuous Planck at 560 for normalization reference.
        ref = spd.sample(560.0, extrapolate=True)
        if ref <= 0:
            raise SPDError("blackbody sample at 560 nm is non-positive")
        spd = spd.scale(100.0 / ref, name=label)
    return spd


def illuminant_a(**kwargs: float) -> SPD:
    """CIE standard illuminant A (Planckian at 2856 K)."""
    # Prefer tabulated data for consistency with package tables.
    if not kwargs:
        return get_illuminant("A")
    return blackbody_spd(2856.0, name="A", **kwargs)  # type: ignore[arg-type]


def daylight_locus_xy(cct_k: float) -> tuple[float, float]:
    """Approximate CIE daylight locus chromaticity for a given CCT.

    Implements the CIE daylight locus polynomials valid roughly for
    4000–25000 K. Returns (x, y) chromaticity.
    """
    if cct_k < 4000 or cct_k > 25000:
        raise IlluminantError("daylight locus CCT must be in [4000, 25000] K")
    t = cct_k
    if t <= 7000:
        x = (
            -4.6070e9 / (t ** 3)
            + 2.9678e6 / (t ** 2)
            + 0.09911e3 / t
            + 0.244063
        )
    else:
        x = (
            -2.0064e9 / (t ** 3)
            + 1.9018e6 / (t ** 2)
            + 0.24748e3 / t
            + 0.237040
        )
    y = -3.0 * x * x + 2.870 * x - 0.275
    return (x, y)


def d_series_cct(name: str) -> float:
    """Return the nominal CCT for a D-series illuminant name."""
    key = resolve_illuminant_name(name)
    if key not in ILLUMINANT_CCT_K:
        raise IlluminantError(f"no CCT recorded for {key}")
    return ILLUMINANT_CCT_K[key]
