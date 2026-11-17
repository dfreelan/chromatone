"""Spectral power distribution (SPD) type and utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Union

from chromatone._mathutil import arange_inclusive, lerp, trapz


DEFAULT_WAVELENGTH_MIN_NM = 380.0
DEFAULT_WAVELENGTH_MAX_NM = 780.0
DEFAULT_WAVELENGTH_STEP_NM = 5.0


class SPDError(ValueError):
    """Raised when an SPD fails validation or sampling constraints."""


def _validate_wavelengths(wavelengths: Sequence[float]) -> tuple[float, ...]:
    if len(wavelengths) < 2:
        raise SPDError("SPD requires at least two wavelength samples")
    wl = tuple(float(w) for w in wavelengths)
    for w in wl:
        if not math.isfinite(w):
            raise SPDError(f"non-finite wavelength: {w}")
        if w <= 0:
            raise SPDError(f"wavelength must be positive (nm), got {w}")
    for i in range(1, len(wl)):
        if wl[i] <= wl[i - 1]:
            raise SPDError(
                f"wavelengths must be strictly increasing; "
                f"index {i}: {wl[i - 1]} -> {wl[i]}"
            )
    return wl


def _validate_values(values: Sequence[float], n: int) -> tuple[float, ...]:
    if len(values) != n:
        raise SPDError(f"expected {n} values, got {len(values)}")
    out = tuple(float(v) for v in values)
    for v in out:
        if not math.isfinite(v):
            raise SPDError(f"non-finite SPD value: {v}")
    return out


@dataclass(frozen=True)
class SPD:
    """Immutable spectral power distribution (wavelengths_nm, values, name)."""

    wavelengths: tuple[float, ...]
    values: tuple[float, ...]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        wl = _validate_wavelengths(self.wavelengths)
        vals = _validate_values(self.values, len(wl))
        object.__setattr__(self, "wavelengths", wl)
        object.__setattr__(self, "values", vals)


    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[float, float],
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Build an SPD from a wavelength→value mapping."""
        items = sorted((float(k), float(v)) for k, v in mapping.items())
        if not items:
            raise SPDError("empty mapping")
        wavelengths = tuple(w for w, _ in items)
        values = tuple(v for _, v in items)
        return cls(wavelengths, values, name=name)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[float, float]],
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Build an SPD from an iterable of (wavelength, value) pairs."""
        items = [(float(w), float(v)) for w, v in pairs]
        items.sort(key=lambda p: p[0])
        if not items:
            raise SPDError("empty pair list")
        dedup: dict[float, float] = {}
        for w, v in items:
            dedup[w] = v
        wavelengths = tuple(sorted(dedup))
        values = tuple(dedup[w] for w in wavelengths)
        return cls(wavelengths, values, name=name)

    @classmethod
    def zeros(
        cls,
        start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
        stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
        step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Create a zero-valued SPD on a regular grid."""
        wavelengths = tuple(arange_inclusive(start_nm, stop_nm, step_nm))
        values = tuple(0.0 for _ in wavelengths)
        return cls(wavelengths, values, name=name)

    @classmethod
    def ones(
        cls,
        start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
        stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
        step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Create a unit-valued (equal-energy) SPD on a regular grid."""
        wavelengths = tuple(arange_inclusive(start_nm, stop_nm, step_nm))
        values = tuple(1.0 for _ in wavelengths)
        return cls(wavelengths, values, name=name or "E")

    @classmethod
    def gaussian(
        cls,
        peak_nm: float,
        fwhm_nm: float,
        *,
        amplitude: float = 1.0,
        start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
        stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
        step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
        name: Optional[str] = None,
    ) -> "SPD":
        """Synthesize a Gaussian lobe (FWHM-based; useful for LED models)."""
        if fwhm_nm <= 0:
            raise SPDError("fwhm_nm must be positive")
        if amplitude < 0:
            raise SPDError("amplitude must be non-negative")
        sigma = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        wavelengths = tuple(arange_inclusive(start_nm, stop_nm, step_nm))
        values = []
        for w in wavelengths:
            x = (w - peak_nm) / sigma
            values.append(amplitude * math.exp(-0.5 * x * x))
        label = name or f"gaussian({peak_nm:.0f},{fwhm_nm:.0f})"
        return cls(wavelengths, tuple(values), name=label)


    def __len__(self) -> int:
        return len(self.wavelengths)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return zip(self.wavelengths, self.values)

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        return (
            f"SPD(n={len(self)}, "
            f"range=[{self.wavelengths[0]:.1f}, {self.wavelengths[-1]:.1f}] nm"
            f"{label})"
        )

    @property
    def start_nm(self) -> float:
        return self.wavelengths[0]

    @property
    def stop_nm(self) -> float:
        return self.wavelengths[-1]

    @property
    def step_nm(self) -> Optional[float]:
        """Return the common step if the grid is regular, else None."""
        if len(self) < 2:
            return None
        steps = [
            self.wavelengths[i] - self.wavelengths[i - 1]
            for i in range(1, len(self))
        ]
        first = steps[0]
        if all(abs(s - first) <= 1e-9 for s in steps):
            return first
        return None

    @property
    def is_regular(self) -> bool:
        return self.step_nm is not None

    def copy(self, *, name: Optional[str] = None) -> "SPD":
        return SPD(self.wavelengths, self.values, name=name if name is not None else self.name)

    def rename(self, name: str) -> "SPD":
        return SPD(self.wavelengths, self.values, name=name)


    def sample(self, wavelength_nm: float, *, extrapolate: bool = False) -> float:
        """Linear interpolate at wavelength_nm (0 outside range unless extrapolate)."""
        wl = self.wavelengths
        vals = self.values
        if wavelength_nm < wl[0]:
            return vals[0] if extrapolate else 0.0
        if wavelength_nm > wl[-1]:
            return vals[-1] if extrapolate else 0.0
        lo, hi = 0, len(wl) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if wl[mid] <= wavelength_nm:
                lo = mid
            else:
                hi = mid
        if abs(wl[lo] - wavelength_nm) < 1e-12:
            return vals[lo]
        if abs(wl[hi] - wavelength_nm) < 1e-12:
            return vals[hi]
        t = (wavelength_nm - wl[lo]) / (wl[hi] - wl[lo])
        return lerp(vals[lo], vals[hi], t)

    def resample(
        self,
        wavelengths: Sequence[float],
        *,
        extrapolate: bool = False,
        name: Optional[str] = None,
    ) -> "SPD":
        """Resample onto an arbitrary wavelength grid via linear interpolation."""
        wl = _validate_wavelengths(wavelengths)
        values = tuple(self.sample(w, extrapolate=extrapolate) for w in wl)
        return SPD(wl, values, name=name if name is not None else self.name)

    def resample_regular(
        self,
        start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
        stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
        step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
        *,
        extrapolate: bool = False,
        name: Optional[str] = None,
    ) -> "SPD":
        """Resample onto a regular wavelength grid."""
        wavelengths = arange_inclusive(start_nm, stop_nm, step_nm)
        return self.resample(wavelengths, extrapolate=extrapolate, name=name)

    def clip_range(
        self,
        start_nm: float,
        stop_nm: float,
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Keep samples inside ``[start_nm, stop_nm]`` inclusive."""
        if stop_nm <= start_nm:
            raise SPDError("stop_nm must exceed start_nm")
        pairs = [(w, v) for w, v in self if start_nm <= w <= stop_nm]
        if len(pairs) < 2:
            raise SPDError("clipped SPD has fewer than two samples")
        return SPD(
            tuple(w for w, _ in pairs),
            tuple(v for _, v in pairs),
            name=name if name is not None else self.name,
        )


    def scale(self, factor: float, *, name: Optional[str] = None) -> "SPD":
        return SPD(
            self.wavelengths,
            tuple(v * factor for v in self.values),
            name=name if name is not None else self.name,
        )

    def normalize_peak(self, peak: float = 1.0, *, name: Optional[str] = None) -> "SPD":
        """Scale so the maximum sample equals ``peak``."""
        m = max(self.values)
        if m <= 0:
            raise SPDError("cannot normalize SPD with non-positive peak")
        return self.scale(peak / m, name=name)

    def normalize_at(
        self,
        wavelength_nm: float,
        target: float = 100.0,
        *,
        name: Optional[str] = None,
    ) -> "SPD":
        """Scale so the (interpolated) value at ``wavelength_nm`` equals ``target``."""
        current = self.sample(wavelength_nm, extrapolate=True)
        if abs(current) < 1e-15:
            raise SPDError(f"SPD is ~0 at {wavelength_nm} nm; cannot normalize")
        return self.scale(target / current, name=name)

    def integral(self) -> float:
        """Trapezoidal integral of the SPD over its support."""
        return trapz(self.values, self.wavelengths)

    def align_with(self, other: "SPD", *, extrapolate: bool = False) -> tuple["SPD", "SPD"]:
        """Resample both SPDs onto the union of their wavelength grids."""
        merged = sorted(set(self.wavelengths) | set(other.wavelengths))
        a = self.resample(merged, extrapolate=extrapolate)
        b = other.resample(merged, extrapolate=extrapolate)
        return a, b

    def add(self, other: "SPD", *, name: Optional[str] = None) -> "SPD":
        a, b = self.align_with(other)
        return SPD(
            a.wavelengths,
            tuple(x + y for x, y in zip(a.values, b.values)),
            name=name,
        )

    def subtract(self, other: "SPD", *, name: Optional[str] = None) -> "SPD":
        a, b = self.align_with(other)
        return SPD(
            a.wavelengths,
            tuple(x - y for x, y in zip(a.values, b.values)),
            name=name,
        )

    def multiply(self, other: "SPD", *, name: Optional[str] = None) -> "SPD":
        """Pointwise product (e.g. illuminant × reflectance)."""
        a, b = self.align_with(other)
        return SPD(
            a.wavelengths,
            tuple(x * y for x, y in zip(a.values, b.values)),
            name=name,
        )

    def __add__(self, other: object) -> "SPD":
        if isinstance(other, SPD):
            return self.add(other)
        if isinstance(other, (int, float)):
            return SPD(self.wavelengths, tuple(v + float(other) for v in self.values), name=self.name)
        return NotImplemented

    def __radd__(self, other: object) -> "SPD":
        return self.__add__(other)

    def __sub__(self, other: object) -> "SPD":
        if isinstance(other, SPD):
            return self.subtract(other)
        if isinstance(other, (int, float)):
            return SPD(self.wavelengths, tuple(v - float(other) for v in self.values), name=self.name)
        return NotImplemented

    def __mul__(self, other: object) -> "SPD":
        if isinstance(other, SPD):
            return self.multiply(other)
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: object) -> "SPD":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "SPD":
        if isinstance(other, (int, float)):
            denom = float(other)
            if abs(denom) < 1e-15:
                raise SPDError("division by zero")
            return self.scale(1.0 / denom)
        if isinstance(other, SPD):
            a, b = self.align_with(other)
            values = []
            for x, y in zip(a.values, b.values):
                if abs(y) < 1e-15:
                    values.append(0.0)
                else:
                    values.append(x / y)
            return SPD(a.wavelengths, tuple(values))
        return NotImplemented

    def nonneg(self, *, name: Optional[str] = None) -> "SPD":
        """Clamp negative samples to zero."""
        return SPD(
            self.wavelengths,
            tuple(max(0.0, v) for v in self.values),
            name=name if name is not None else self.name,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "wavelengths_nm": list(self.wavelengths),
            "values": list(self.values),
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SPD":
        """Deserialize from :meth:`to_dict` output."""
        if "wavelengths_nm" in data and "values" in data:
            wl = data["wavelengths_nm"]
            vals = data["values"]
        elif "wavelengths" in data and "values" in data:
            wl = data["wavelengths"]
            vals = data["values"]
        else:
            raise SPDError("dict must contain wavelengths(_nm) and values")
        if not isinstance(wl, Sequence) or not isinstance(vals, Sequence):
            raise SPDError("wavelengths and values must be sequences")
        name = data.get("name")
        return cls(
            tuple(float(x) for x in wl),  # type: ignore[arg-type]
            tuple(float(x) for x in vals),  # type: ignore[arg-type]
            name=str(name) if name is not None else None,
        )


def ensure_spd(obj: Union[SPD, Mapping[float, float]]) -> SPD:
    """Coerce mappings to :class:`SPD`."""
    if isinstance(obj, SPD):
        return obj
    return SPD.from_mapping(obj)
