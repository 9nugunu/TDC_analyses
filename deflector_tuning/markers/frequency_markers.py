"""Marker-frequency extraction from CST dispersion summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")


@dataclass(frozen=True)
class TemperatureHumidityCorrection:
    """Frequency scaling used for measurement-condition marker frequencies."""

    temp_op_C: float = 20.0
    temp_meas_C: float = 24.4
    humidity_fraction: float = 0.65
    thermal_alpha_per_C: float = 1.68e-5
    eps_air_humid: float = 1.000712754221782

    @property
    def frequency_scale_factor(self) -> float:
        thermal_scale = 1.0 / (1.0 + self.thermal_alpha_per_C * (self.temp_meas_C - self.temp_op_C))
        dielectric_scale = 1.0 / (self.eps_air_humid**0.5)
        return thermal_scale * dielectric_scale


def extract_marker_frequencies(
    path: str | Path,
    *,
    marker_role: str = "sim",
    mode_index: int = 1,
    correction: TemperatureHumidityCorrection | None = None,
) -> pd.DataFrame:
    """Extract f_2pi3/f_mean/f_pi2 marker frequencies from a dispersion summary.

    Simulation markers use the dispersion frequencies directly. Experimental
    markers apply the measurement-condition temperature/humidity correction to
    the same dispersion basis.
    """

    folder = Path(path)
    summary_path = _find_dispersion_summary(folder)
    summary = pd.read_csv(summary_path)
    mode_rows = summary.loc[summary["mode_index"] == mode_index]
    if mode_rows.empty:
        raise ValueError(f"No mode_index={mode_index} row in {summary_path}")
    row = mode_rows.iloc[0]

    uncorrected = {
        "f_2pi3": float(row["freq_120_GHz"]),
        "f_pi2": float(row["freq_90_GHz"]),
    }
    uncorrected["f_mean"] = (uncorrected["f_2pi3"] + uncorrected["f_pi2"]) / 2.0

    if marker_role == "sim":
        scale = 1.0
        marker_source = f"dispersion_mode_{mode_index}"
        correction_values = _empty_correction_values()
    elif marker_role == "exp":
        correction = correction or TemperatureHumidityCorrection()
        scale = correction.frequency_scale_factor
        marker_source = f"dispersion_mode_{mode_index}_temp_humidity_corrected"
        correction_values = {
            "temp_op_C": correction.temp_op_C,
            "temp_meas_C": correction.temp_meas_C,
            "humidity_fraction": correction.humidity_fraction,
            "thermal_alpha_per_C": correction.thermal_alpha_per_C,
            "eps_air_humid": correction.eps_air_humid,
        }
    else:
        raise ValueError("marker_role must be 'sim' or 'exp'")

    source_file = summary_path.relative_to(folder).as_posix()
    return pd.DataFrame(
        [
            {
                "marker_name": marker,
                "freq_ghz": uncorrected[marker] * scale,
                "marker_role": marker_role,
                "marker_source": marker_source,
                "source_file": source_file,
                "uncorrected_freq_ghz": uncorrected[marker],
                "frequency_scale_factor": scale,
                **correction_values,
            }
            for marker in MARKER_ORDER
        ]
    )


def _find_dispersion_summary(folder: Path) -> Path:
    candidates = sorted((folder / "processed").glob("*summary.csv"))
    if not candidates:
        candidates = sorted(folder.glob("*summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dispersion summary CSV found under {folder}")
    return candidates[0]


def _empty_correction_values() -> dict[str, None]:
    return {
        "temp_op_C": None,
        "temp_meas_C": None,
        "humidity_fraction": None,
        "thermal_alpha_per_C": None,
        "eps_air_humid": None,
    }
