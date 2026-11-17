"""SPD / LED spectral mixing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

from chromatone.spd import (
    DEFAULT_WAVELENGTH_MAX_NM,
    DEFAULT_WAVELENGTH_MIN_NM,
    DEFAULT_WAVELENGTH_STEP_NM,
    SPD,
    SPDError,
)
from chromatone.xyz import XYZ, spd_to_xyz


class MixError(ValueError):
    """Raised for invalid mixing weights or empty channel sets."""


@dataclass(frozen=True)
class LEDChannel:
    """A named LED-like spectral channel."""

    name: str
    spd: SPD
    peak_nm: Optional[float] = None
    fwhm_nm: Optional[float] = None

    def __repr__(self) -> str:
        peak = f", peak={self.peak_nm:.0f}nm" if self.peak_nm else ""
        return f"LEDChannel({self.name!r}{peak})"


def mix_spds(
    spectra: Sequence[SPD],
    weights: Optional[Sequence[float]] = None,
    *,
    name: Optional[str] = None,
    normalize_weights: bool = False,
) -> SPD:
    """Linear combination of SPDs.

    Parameters
    ----------
    spectra:
        One or more SPD components (aligned automatically).
    weights:
        Per-component weights (default: all ones). Length must match
        ``spectra``.
    normalize_weights:
        If True, divide weights by their sum (convex combination).
    """
    if not spectra:
        raise MixError("need at least one spectrum to mix")
    if weights is None:
        w = [1.0] * len(spectra)
    else:
        if len(weights) != len(spectra):
            raise MixError(
                f"weights length {len(weights)} != spectra length {len(spectra)}"
            )
        w = [float(x) for x in weights]
    if any(x < 0 for x in w):
        raise MixError("weights must be non-negative")
    total = sum(w)
    if normalize_weights:
        if total <= 0:
            raise MixError("cannot normalize weights that sum to zero")
        w = [x / total for x in w]

    # Align all spectra onto the union grid of the first, iteratively.
    result = spectra[0].scale(w[0])
    for spd, weight in zip(spectra[1:], w[1:]):
        if weight == 0.0:
            continue
        result = result.add(spd.scale(weight))
    return result.rename(name or "mix")


def mix_weighted(
    pairs: Iterable[tuple[SPD, float]],
    *,
    name: Optional[str] = None,
    normalize_weights: bool = False,
) -> SPD:
    """Mix ``(SPD, weight)`` pairs."""
    spectra: list[SPD] = []
    weights: list[float] = []
    for spd, weight in pairs:
        spectra.append(spd)
        weights.append(float(weight))
    return mix_spds(spectra, weights, name=name, normalize_weights=normalize_weights)


def gaussian_led(
    peak_nm: float,
    fwhm_nm: float = 20.0,
    *,
    amplitude: float = 1.0,
    name: Optional[str] = None,
    start_nm: float = DEFAULT_WAVELENGTH_MIN_NM,
    stop_nm: float = DEFAULT_WAVELENGTH_MAX_NM,
    step_nm: float = DEFAULT_WAVELENGTH_STEP_NM,
) -> LEDChannel:
    """Create a Gaussian LED channel."""
    label = name or f"LED_{peak_nm:.0f}"
    spd = SPD.gaussian(
        peak_nm,
        fwhm_nm,
        amplitude=amplitude,
        start_nm=start_nm,
        stop_nm=stop_nm,
        step_nm=step_nm,
        name=label,
    )
    return LEDChannel(name=label, spd=spd, peak_nm=peak_nm, fwhm_nm=fwhm_nm)


# Typical narrowband LED peaks used in imaging / vision setups.
DEFAULT_LED_PEAKS_NM: tuple[float, ...] = (450.0, 530.0, 660.0)


def rgb_led_set(
    peaks_nm: Sequence[float] = DEFAULT_LED_PEAKS_NM,
    *,
    fwhm_nm: float = 20.0,
    names: Optional[Sequence[str]] = None,
) -> list[LEDChannel]:
    """Build a small set of Gaussian LED channels (default R/G/B-ish)."""
    if names is None:
        default_names = ("B", "G", "R", "C4", "C5", "C6")
        labels = list(default_names[: len(peaks_nm)])
        while len(labels) < len(peaks_nm):
            labels.append(f"C{len(labels)+1}")
    else:
        if len(names) != len(peaks_nm):
            raise MixError("names length must match peaks_nm")
        labels = list(names)
    return [
        gaussian_led(peak, fwhm_nm, name=label)
        for peak, label in zip(peaks_nm, labels)
    ]


def mix_leds(
    channels: Sequence[LEDChannel],
    weights: Sequence[float],
    *,
    name: Optional[str] = None,
    normalize_weights: bool = True,
) -> SPD:
    """Mix LED channels by drive weights."""
    spectra = [ch.spd for ch in channels]
    return mix_spds(
        spectra,
        weights,
        name=name or "led_mix",
        normalize_weights=normalize_weights,
    )


def match_xyz_with_leds(
    target: XYZ,
    channels: Sequence[LEDChannel],
    *,
    iterations: int = 200,
    learning_rate: float = 0.05,
    nonneg: bool = True,
) -> tuple[SPD, list[float], XYZ]:
    """Approximate a target XYZ by non-negative LED weights (projected gradient).

    This is a simple iterative solver intended for demos and unit tests—not
    a production colorimetric optimizer. Returns ``(mixed_spd, weights, achieved_xyz)``.
    """
    if not channels:
        raise MixError("need at least one LED channel")
    n = len(channels)
    weights = [1.0 / n] * n

    # Precompute channel XYZ (Y-normalized relative scale).
    channel_xyz = [spd_to_xyz(ch.spd, normalize_y=None) for ch in channels]

    def mix_xyz(w: list[float]) -> XYZ:
        X = Y = Z = 0.0
        for wi, xyz in zip(w, channel_xyz):
            X += wi * xyz.X
            Y += wi * xyz.Y
            Z += wi * xyz.Z
        return XYZ(X, Y, Z)

    # Scale target to a comfortable magnitude if needed.
    target_vec = target.as_tuple()
    for _ in range(iterations):
        current = mix_xyz(weights)
        err = (
            target_vec[0] - current.X,
            target_vec[1] - current.Y,
            target_vec[2] - current.Z,
        )
        # Gradient of 0.5||Aw - t||^2 w.r.t. weight i is -A_i · err
        for i, xyz in enumerate(channel_xyz):
            grad = -(xyz.X * err[0] + xyz.Y * err[1] + xyz.Z * err[2])
            weights[i] -= learning_rate * grad
            if nonneg and weights[i] < 0.0:
                weights[i] = 0.0
        # Soft renormalize to keep Y roughly on target.
        achieved = mix_xyz(weights)
        if achieved.Y > 1e-12 and target.Y > 1e-12:
            scale = target.Y / achieved.Y
            weights = [w * scale for w in weights]

    mixed = mix_leds(channels, weights, normalize_weights=False, name="led_match")
    achieved = spd_to_xyz(mixed, normalize_y=None)
    # Final Y match scale for reporting consistency.
    if achieved.Y > 1e-12 and target.Y > 1e-12:
        s = target.Y / achieved.Y
        mixed = mixed.scale(s)
        weights = [w * s for w in weights]
        achieved = spd_to_xyz(mixed, normalize_y=None)
    return mixed, weights, achieved


def convex_combine(a: SPD, b: SPD, t: float, *, name: Optional[str] = None) -> SPD:
    """Return ``(1-t)*a + t*b`` with ``t`` clamped to [0, 1]."""
    if t < 0.0 or t > 1.0:
        raise MixError("t must be in [0, 1]")
    return mix_spds([a, b], [1.0 - t, t], name=name or f"blend({t:.3f})")


def scale_to_luminance(spd: SPD, target_y: float = 100.0) -> SPD:
    """Scale an SPD so its CIE Y equals ``target_y``."""
    xyz = spd_to_xyz(spd, normalize_y=None)
    if abs(xyz.Y) < 1e-15:
        raise SPDError("SPD has Y ≈ 0; cannot scale to luminance")
    return spd.scale(target_y / xyz.Y)
