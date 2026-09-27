"""Two-chain equivalent-circuit dispersion model for dipole cavities.

The formulation follows Eqs. (21)--(23) in G. Burt, *Transverse
deflecting cavities*, CERN-2011-007, pp. 395--405.  The four circuit
parameters are reconstructed from the 0- and pi-mode frequencies of the
two hybrid passbands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DoubleChainParameters:
    """Band-edge reconstruction parameters in inverse GHz squared."""

    x_inv_GHz2: float
    xbar_inv_GHz2: float
    kappa_inv_GHz2: float
    kappabar_inv_GHz2: float

    @classmethod
    def from_band_edges(
        cls,
        *,
        lower_f0_ghz: float,
        lower_fpi_ghz: float,
        upper_f0_ghz: float,
        upper_fpi_ghz: float,
    ) -> "DoubleChainParameters":
        """Construct Burt's two-chain parameters from four band edges."""

        band_edges = np.asarray(
            [lower_f0_ghz, lower_fpi_ghz, upper_f0_ghz, upper_fpi_ghz],
            dtype=float,
        )
        if not np.all(np.isfinite(band_edges) & (band_edges > 0.0)):
            raise ValueError("Band edges must be positive finite frequencies")

        lambda_lower_0 = 1.0 / lower_f0_ghz**2
        lambda_lower_pi = 1.0 / lower_fpi_ghz**2
        lambda_upper_0 = 1.0 / upper_f0_ghz**2
        lambda_upper_pi = 1.0 / upper_fpi_ghz**2
        return cls(
            x_inv_GHz2=0.5 * (lambda_lower_pi + lambda_lower_0),
            xbar_inv_GHz2=0.5 * (lambda_upper_0 + lambda_upper_pi),
            kappa_inv_GHz2=0.5 * (lambda_lower_pi - lambda_lower_0),
            kappabar_inv_GHz2=0.5 * (lambda_upper_0 - lambda_upper_pi),
        )


def predict_double_chain_frequencies(
    phase_deg: np.ndarray,
    parameters: DoubleChainParameters,
) -> pd.DataFrame:
    """Predict the lower and upper hybrid passbands versus phase advance."""

    phase = np.asarray(phase_deg, dtype=float)
    cosine = np.cos(np.deg2rad(phase))
    x = parameters.x_inv_GHz2
    xbar = parameters.xbar_inv_GHz2
    kappa = parameters.kappa_inv_GHz2
    kappabar = parameters.kappabar_inv_GHz2

    b = 0.5 * (x + xbar - (kappa - kappabar) * cosine)
    c = x * xbar - kappa * kappabar + (x * kappabar - kappa * xbar) * cosine
    discriminant = np.clip(np.square(b) - c, 0.0, None)
    root = np.sqrt(discriminant)
    lambda_lower = b + root
    lambda_upper = b - root
    return pd.DataFrame(
        {
            "phase_deg": phase,
            "lower_freq_GHz": 1.0 / np.sqrt(lambda_lower),
            "upper_freq_GHz": 1.0 / np.sqrt(lambda_upper),
        }
    )


def track_nearest_mode_branch(
    dispersion_table: pd.DataFrame,
    *,
    phase_deg: np.ndarray,
    predicted_freq_ghz: np.ndarray,
    candidate_mode_indices: tuple[int, ...],
) -> pd.DataFrame:
    """Track a physical branch when frequency-sorted CST mode indices switch."""

    phases = np.asarray(phase_deg, dtype=float)
    predicted = np.asarray(predicted_freq_ghz, dtype=float)
    rows: list[dict[str, float | int]] = []
    candidates = dispersion_table[dispersion_table["mode_index"].isin(candidate_mode_indices)]
    for phase, predicted_frequency in zip(phases, predicted, strict=True):
        at_phase = candidates[np.isclose(candidates["phase_deg"], phase)]
        if at_phase.empty:
            raise ValueError(f"No candidate eigenmode at phase {phase:g} deg")
        distance = np.abs(at_phase["freq_GHz"].to_numpy(dtype=float) - predicted_frequency)
        observed = at_phase.iloc[int(np.argmin(distance))]
        observed_frequency = float(observed["freq_GHz"])
        rows.append(
            {
                "phase_deg": float(phase),
                "predicted_freq_GHz": float(predicted_frequency),
                "observed_mode_index": int(observed["mode_index"]),
                "observed_freq_GHz": observed_frequency,
                "residual_MHz": (observed_frequency - predicted_frequency) * 1000.0,
            }
        )
    return pd.DataFrame(rows)
