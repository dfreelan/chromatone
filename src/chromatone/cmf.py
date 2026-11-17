"""CIE colour-matching functions (1931 2° observer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from chromatone._mathutil import lerp
from chromatone.data.cmf_cie1931 import CMF_ROWS, WAVELENGTHS_NM, X_BAR, Y_BAR, Z_BAR
from chromatone.spd import SPD, SPDError


class CMFError(ValueError):
    """Raised for CMF lookup / sampling errors."""


@dataclass(frozen=True)
class CMF:
    """Colour-matching function triplet on a shared wavelength grid.

    Attributes
    ----------
    wavelengths:
        Wavelength samples in nm.
    x_bar, y_bar, z_bar:
        CIE matching-function values.
    observer:
        Descriptive label (e.g. ``"CIE 1931 2-deg"``).
    """

    wavelengths: tuple[float, ...]
    x_bar: tuple[float, ...]
    y_bar: tuple[float, ...]
    z_bar: tuple[float, ...]
    observer: str = "CIE 1931 2-deg"

    def __post_init__(self) -> None:
        n = len(self.wavelengths)
        if n < 2:
            raise CMFError("CMF requires at least two samples")
        if not (len(self.x_bar) == len(self.y_bar) == len(self.z_bar) == n):
            raise CMFError("CMF component lengths must match wavelengths")
        for i in range(1, n):
            if self.wavelengths[i] <= self.wavelengths[i - 1]:
                raise CMFError("CMF wavelengths must be strictly increasing")

    def __len__(self) -> int:
        return len(self.wavelengths)

    def __repr__(self) -> str:
        return (
            f"CMF(observer={self.observer!r}, n={len(self)}, "
            f"range=[{self.wavelengths[0]:.0f}, {self.wavelengths[-1]:.0f}] nm)"
        )

    @property
    def start_nm(self) -> float:
        return self.wavelengths[0]

    @property
    def stop_nm(self) -> float:
        return self.wavelengths[-1]

    def as_spd_triplet(self) -> tuple[SPD, SPD, SPD]:
        """Return (x_bar, y_bar, z_bar) as SPD objects."""
        return (
            SPD(self.wavelengths, self.x_bar, name=f"{self.observer} x_bar"),
            SPD(self.wavelengths, self.y_bar, name=f"{self.observer} y_bar"),
            SPD(self.wavelengths, self.z_bar, name=f"{self.observer} z_bar"),
        )

    def sample(self, wavelength_nm: float) -> tuple[float, float, float]:
        """Linearly interpolate (x_bar, y_bar, z_bar) at a wavelength."""
        wl = self.wavelengths
        if wavelength_nm < wl[0] or wavelength_nm > wl[-1]:
            return (0.0, 0.0, 0.0)
        lo, hi = 0, len(wl) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if wl[mid] <= wavelength_nm:
                lo = mid
            else:
                hi = mid
        if abs(wl[lo] - wavelength_nm) < 1e-12:
            return (self.x_bar[lo], self.y_bar[lo], self.z_bar[lo])
        if abs(wl[hi] - wavelength_nm) < 1e-12:
            return (self.x_bar[hi], self.y_bar[hi], self.z_bar[hi])
        t = (wavelength_nm - wl[lo]) / (wl[hi] - wl[lo])
        return (
            lerp(self.x_bar[lo], self.x_bar[hi], t),
            lerp(self.y_bar[lo], self.y_bar[hi], t),
            lerp(self.z_bar[lo], self.z_bar[hi], t),
        )

    def resample(self, wavelengths: Sequence[float], *, observer: Optional[str] = None) -> "CMF":
        """Resample CMF components onto ``wavelengths``."""
        if len(wavelengths) < 2:
            raise CMFError("need at least two wavelengths")
        wl = tuple(float(w) for w in wavelengths)
        for i in range(1, len(wl)):
            if wl[i] <= wl[i - 1]:
                raise CMFError("wavelengths must be strictly increasing")
        xs, ys, zs = [], [], []
        for w in wl:
            x, y, z = self.sample(w)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return CMF(
            wl,
            tuple(xs),
            tuple(ys),
            tuple(zs),
            observer=observer or self.observer,
        )

    def align_to_spd(self, spd: SPD) -> "CMF":
        """Resample this CMF onto an SPD's wavelength grid."""
        return self.resample(spd.wavelengths)


def cie_1931_2deg() -> CMF:
    """Return the embedded CIE 1931 2° observer at 5 nm (380–780 nm)."""
    return CMF(
        wavelengths=tuple(float(w) for w in WAVELENGTHS_NM),
        x_bar=tuple(X_BAR),
        y_bar=tuple(Y_BAR),
        z_bar=tuple(Z_BAR),
        observer="CIE 1931 2-deg",
    )


def load_cmf(name: str = "CIE1931_2") -> CMF:
    """Load a named CMF table.

    Currently supported names (case-insensitive)::

        CIE1931_2, CIE1931, 1931, 2deg, 2°
    """
    key = name.strip().lower().replace(" ", "").replace("°", "deg")
    aliases = {
        "cie1931_2",
        "cie1931",
        "1931",
        "2deg",
        "cie19312deg",
        "cie_1931_2",
        "standardobserver",
        "standard_observer",
    }
    if key in aliases:
        return cie_1931_2deg()
    raise CMFError(
        f"unknown CMF {name!r}; available: CIE1931_2 (CIE 1931 2-degree observer)"
    )


def cmf_rows() -> tuple[tuple[float, float, float, float], ...]:
    """Return raw (wavelength, x_bar, y_bar, z_bar) rows from package data."""
    return CMF_ROWS


def validate_cmf_coverage(spd: SPD, cmf: Optional[CMF] = None) -> None:
    """Ensure an SPD overlaps the CMF domain sufficiently.

    Raises :class:`SPDError` if the SPD lies entirely outside the CMF
    support or covers less than two overlapping samples.
    """
    table = cmf or cie_1931_2deg()
    overlap = [
        w for w in spd.wavelengths if table.start_nm <= w <= table.stop_nm
    ]
    if len(overlap) < 2:
        raise SPDError(
            f"SPD [{spd.start_nm}, {spd.stop_nm}] nm does not overlap CMF "
            f"[{table.start_nm}, {table.stop_nm}] nm with ≥2 samples"
        )
