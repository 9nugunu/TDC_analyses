"""Shared validated scientific helpers for source-auditable analysis."""

from __future__ import annotations

import numpy as np
from typing import Any


def phase_pair_metrics(first: complex, second: complex, target_deg: float) -> dict[str, float]:
    """Return directed second/first phase separation and wrapped target residual.

    Separation is in [0, 360); target residual is in [-180, 180). This is a
    reference-phase comparison, not an independent impedance matching test.
    """
    if not all(np.isfinite(value) and abs(value) > 0 for value in (first, second)):
        raise ValueError("Phase inputs must be finite and nonzero")
    delta = float(np.rad2deg(np.angle(second / first)) % 360.0)
    # Floating point modulo can round an infinitesimal negative value to 360.
    if delta >= 360.0:
        delta = 0.0
    return {
        "first_real": float(first.real), "first_imag": float(first.imag),
        "second_real": float(second.real), "second_imag": float(second.imag),
        "first_phase_deg": float(np.rad2deg(np.angle(first))),
        "second_phase_deg": float(np.rad2deg(np.angle(second))),
        "first_modulus": float(abs(first)), "second_modulus": float(abs(second)),
        "delta_deg": delta,
        "target_delta_deg": float(target_deg),
        "signed_target_residual_deg": float((delta - target_deg + 180.0) % 360.0 - 180.0),
        "relative_phase_psi_deg": float((delta - 180.0 + 180.0) % 360.0 - 180.0),
    }

def radius_phase_response(
    radii: np.ndarray, raw_phases_deg: np.ndarray, nominal_radius: float
) -> dict[str, Any]:
    """Unwrap along radius and calculate the centered local phase derivative."""
    radii, phases = np.asarray(radii, dtype=float), np.asarray(raw_phases_deg, dtype=float)
    if (radii.ndim != 1 or radii.shape != phases.shape or len(radii) < 3
            or not np.isfinite(radii).all() or not np.isfinite(phases).all()
            or np.any(np.diff(radii) <= 0)):
        raise ValueError("Radius and phase must be finite equal-length arrays on a sorted grid")
    matches = np.flatnonzero(np.isclose(radii, nominal_radius, atol=1e-10, rtol=0))
    if len(matches) != 1 or matches[0] in (0, len(radii) - 1):
        raise ValueError("Nominal radius must identify one interior grid point")
    index = int(matches[0])
    if not np.isclose(radii[index] - radii[index - 1], radii[index + 1] - radii[index],
                      atol=1e-10, rtol=0):
        raise ValueError("Nominal centered derivative requires equal adjacent radius steps")
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phases)))
    unwrapped -= 360.0 * round((unwrapped[index] - phases[index]) / 360.0)
    slope = (unwrapped[index + 1] - unwrapped[index - 1]) / (radii[index + 1] - radii[index - 1])
    return {"unwrapped_phase_deg": unwrapped.tolist(),
            "branch_turns": np.rint((unwrapped - phases) / 360.0).astype(int).tolist(),
            "nominal_index": index, "nominal_central_slope_deg_per_mm": float(slope),
            "slope_lower_radius_mm": float(radii[index - 1]),
            "slope_upper_radius_mm": float(radii[index + 1])}
