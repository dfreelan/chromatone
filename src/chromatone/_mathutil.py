"""Minimal stdlib linear-algebra helpers for chromatone."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def as_vec3(values: Sequence[float]) -> Vec3:
    """Coerce a length-3 sequence to a float triple."""
    if len(values) != 3:
        raise ValueError(f"expected 3 components, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def add3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def sub3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def scale3(a: Sequence[float], s: float) -> Vec3:
    return (float(a[0]) * s, float(a[1]) * s, float(a[2]) * s)


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def norm3(a: Sequence[float]) -> float:
    return math.sqrt(dot3(a, a))


def matvec(m: Mat3, v: Sequence[float]) -> Vec3:
    """Multiply a 3×3 matrix (row-major) by a length-3 vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def matmul(a: Mat3, b: Mat3) -> Mat3:
    """Multiply two 3×3 matrices."""
    bt = transpose(b)
    return (
        (dot3(a[0], bt[0]), dot3(a[0], bt[1]), dot3(a[0], bt[2])),
        (dot3(a[1], bt[0]), dot3(a[1], bt[1]), dot3(a[1], bt[2])),
        (dot3(a[2], bt[0]), dot3(a[2], bt[1]), dot3(a[2], bt[2])),
    )


def transpose(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def det3(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def inv3(m: Mat3) -> Mat3:
    """Invert a 3×3 matrix. Raises ValueError if singular."""
    d = det3(m)
    if abs(d) < 1e-15:
        raise ValueError("singular 3x3 matrix")
    inv_d = 1.0 / d
    return (
        (
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
        ),
        (
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
        ),
        (
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
        ),
    )


def diag3(x: float, y: float, z: float) -> Mat3:
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def trapz(y: Sequence[float], x: Sequence[float]) -> float:
    """Trapezoidal integral of y with respect to x."""
    if len(y) != len(x):
        raise ValueError("x and y must have the same length")
    if len(y) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(y)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def linspace(start: float, stop: float, num: int) -> list[float]:
    if num < 2:
        raise ValueError("num must be >= 2")
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def arange_inclusive(start: float, stop: float, step: float) -> list[float]:
    """Inclusive arithmetic sequence with floating wavelengths."""
    if step <= 0:
        raise ValueError("step must be positive")
    values: list[float] = []
    # Use integer counting to avoid cumulative float drift.
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        values.append(start + i * step)
    # Snap endpoint if within tolerance.
    if abs(values[-1] - stop) > 1e-9:
        values.append(stop)
    else:
        values[-1] = stop
    return values


def nearly_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def all_nearly_equal(a: Iterable[float], b: Iterable[float], tol: float = 1e-9) -> bool:
    aa = list(a)
    bb = list(b)
    if len(aa) != len(bb):
        return False
    return all(nearly_equal(x, y, tol) for x, y in zip(aa, bb))
