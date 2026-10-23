"""Shared validated scientific helpers for source-auditable analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nearest_sample_index(frequency_GHz: np.ndarray, requested_GHz: float) -> int:
    """Choose the nearest exported sample; an exact tie selects the first row."""
    frequencies = np.asarray(frequency_GHz, dtype=float)
    if (frequencies.ndim != 1 or frequencies.size == 0
            or not np.isfinite(frequencies).all() or not np.isfinite(requested_GHz)
            or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("Frequency grid must be finite, nonempty and strictly increasing")
    if not frequencies[0] <= requested_GHz <= frequencies[-1]:
        raise ValueError(f"Marker {requested_GHz} GHz lies outside the exported grid")
    return int(np.argmin(np.abs(frequencies - requested_GHz)))


def sample_complex_frequency(
    frequency: ArrayLike, values: ArrayLike, targets_ghz: ArrayLike, *, frequency_unit: str
) -> np.ndarray:
    """Interpolate complex samples on a sorted, unique grid without extrapolation."""
    scales = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}
    if frequency_unit.lower() not in scales:
        raise ValueError(f"Unsupported frequency unit: {frequency_unit}")
    grid = np.asarray(frequency, dtype=float) * scales[frequency_unit.lower()]
    samples = np.asarray(values, dtype=complex)
    targets = np.asarray(targets_ghz, dtype=float)
    if (grid.ndim != 1 or grid.size < 2 or samples.shape != grid.shape
            or not np.isfinite(grid).all() or not np.isfinite(samples).all()
            or targets.ndim != 1 or not np.isfinite(targets).all()):
        raise ValueError("Frequency, complex values and targets must be finite one-dimensional arrays")
    order = np.argsort(grid, kind="stable")
    grid, samples = grid[order], samples[order]
    if np.any(np.diff(grid) <= 0):
        raise ValueError("Frequency grid must contain unique samples")
    # Unit conversion may move an exact endpoint by a few floating-point ulps.
    # Permit only that rounding uncertainty; interpolation never extrapolates.
    tolerance = 8 * np.finfo(float).eps * max(abs(grid[0]), abs(grid[-1]))
    if np.any((targets < grid[0] - tolerance) | (targets > grid[-1] + tolerance)):
        raise ValueError("Requested frequency lies outside the exported grid")
    targets = np.clip(targets, grid[0], grid[-1])
    return np.interp(targets, grid, samples.real) + 1j * np.interp(targets, grid, samples.imag)
