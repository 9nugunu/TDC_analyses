"""Pure field-profile and phase-sweep metrics for Kyhl validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.field_profiles import (
    FieldPhaseExport,
    FieldPhaseTrace,
    FieldProfileExport,
    FieldProfileTrace,
)

TARGET_COUPLER_PHASE_DEG = 240.0
IDEAL_HALF_CELL_PHASE_DEG = 60.0
REGULAR_CELL_COUNT = 9


@dataclass(frozen=True)
class HalfCellStation:
    """One regular half-cell phase reference station."""

    station_index: int
    station_label: str
    station_kind: str
    z_mm: float


def build_field_energy_pairs(export: FieldProfileExport) -> pd.DataFrame:
    """Compute pairwise E_I^2/E_C^2 ratios from the no-plunger E-field profile."""

    trace = _single_profile_trace(export, expected_kind="e")
    params = export.parameters
    period = params["d"] + params["t"]
    iris_z = np.array([params["t"] / 2.0 + i * period for i in range(REGULAR_CELL_COUNT + 1)])
    cell_z = np.array([params["t"] + params["d"] / 2.0 + i * period for i in range(REGULAR_CELL_COUNT)])

    iris_e = interpolate_profile(trace.z_mm, trace.values, iris_z)
    cell_e = interpolate_profile(trace.z_mm, trace.values, cell_z)

    rows = []
    for idx in range(REGULAR_CELL_COUNT):
        upstream = iris_e[idx]
        downstream = iris_e[idx + 1]
        adjacent_average_energy = 0.5 * (upstream**2 + downstream**2)
        adjacent_max_energy = max(upstream**2, downstream**2)
        cell_energy = cell_e[idx] ** 2
        rows.append(
            {
                "pair_index": idx + 1,
                "cell_label": f"R{idx + 1}",
                "upstream_iris_label": "IN" if idx == 0 else f"I{idx}",
                "downstream_iris_label": "OUT" if idx == REGULAR_CELL_COUNT - 1 else f"I{idx + 1}",
                "cell_center_z_mm": cell_z[idx],
                "upstream_iris_z_mm": iris_z[idx],
                "downstream_iris_z_mm": iris_z[idx + 1],
                "E_cell": cell_e[idx],
                "E_upstream_iris": upstream,
                "E_downstream_iris": downstream,
                "E_cell_squared": cell_energy,
                "E_upstream_iris_squared": upstream**2,
                "E_downstream_iris_squared": downstream**2,
                "ratio_upstream_iris_to_cell": upstream**2 / cell_energy,
                "ratio_downstream_iris_to_cell": downstream**2 / cell_energy,
                "ratio_adjacent_average_iris_to_cell": adjacent_average_energy / cell_energy,
                "ratio_adjacent_max_iris_to_cell": adjacent_max_energy / cell_energy,
            }
        )
    return pd.DataFrame(rows)


def summarize_field_energy_pairs(field_pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate field-energy sensitivity ratios."""

    rows = []
    for column, label in [
        ("ratio_upstream_iris_to_cell", "upstream_iris_pair"),
        ("ratio_downstream_iris_to_cell", "downstream_iris_pair"),
        ("ratio_adjacent_average_iris_to_cell", "adjacent_iris_average"),
        ("ratio_adjacent_max_iris_to_cell", "adjacent_iris_max"),
    ]:
        values = field_pairs[column].astype(float)
        rows.append(
            {
                "ratio_definition": label,
                "pair_count": int(values.count()),
                "mean_ratio": float(values.mean()),
                "std_ratio": float(values.std(ddof=1)),
                "min_ratio": float(values.min()),
                "max_ratio": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_no_plunger_phase_points(export: FieldPhaseExport) -> pd.DataFrame:
    """Sample unwrapped no-plunger phase at alternating iris/cell half-cell stations."""

    trace = _single_phase_trace(export, expected_kind="e")
    params = export.parameters
    stations = build_half_cell_stations(params)
    z_sorted, phase_sorted = _sorted_xy(trace.z_mm, trace.phase_deg)
    phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phase_sorted)))
    sampled_z = np.array([station.z_mm for station in stations], dtype=float)
    sampled_phase = interpolate_profile(z_sorted, phase_unwrapped, sampled_z)
    sampled_phase_wrapped = wrap_deg(sampled_phase)

    rows = []
    for station, phase_deg, wrapped_deg in zip(stations, sampled_phase, sampled_phase_wrapped):
        rows.append(
            {
                "dataset_id": "sim_profile_260618_PhaseDistribution",
                "source": "EM_fieldPhase.txt",
                "field_kind": trace.field_kind,
                "component": trace.component,
                "station_index": station.station_index,
                "station_label": station.station_label,
                "station_kind": station.station_kind,
                "z_mm": station.z_mm,
                "phase_unwrapped_deg": float(phase_deg),
                "phase_wrapped_deg": float(wrapped_deg),
            }
        )
    return pd.DataFrame(rows)


def build_no_plunger_phase_residuals(phase_points: pd.DataFrame) -> pd.DataFrame:
    """Compute residuals from the ideal 60 degree half-cell phase map."""

    rows = []
    for slope in (IDEAL_HALF_CELL_PHASE_DEG, -IDEAL_HALF_CELL_PHASE_DEG):
        phase = phase_points["phase_unwrapped_deg"].astype(float).to_numpy()
        station_index = phase_points["station_index"].astype(float).to_numpy()
        offset = float(np.mean(phase - slope * station_index))
        ideal = offset + slope * station_index
        residual = wrap_deg(phase - ideal)
        table = phase_points.copy()
        table["ideal_slope_deg_per_half_cell"] = slope
        table["fitted_phase_offset_deg"] = offset
        table["ideal_phase_deg"] = ideal
        table["phase_residual_deg"] = residual
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def summarize_no_plunger_phase_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """Summarize no-plunger phase residuals by ideal slope convention."""

    rows = []
    for slope, group in residuals.groupby("ideal_slope_deg_per_half_cell", sort=True):
        errors = group["phase_residual_deg"].astype(float)
        rows.append(
            {
                "dataset_id": "sim_profile_260618_PhaseDistribution",
                "source": "EM_fieldPhase.txt",
                "ideal_slope_deg_per_half_cell": float(slope),
                "point_count": int(errors.count()),
                "rms_phase_error_deg": rms(errors),
                "mean_phase_error_deg": float(errors.mean()),
                "mean_abs_phase_error_deg": float(errors.abs().mean()),
                "max_abs_phase_error_deg": float(errors.abs().max()),
                "integrated_phase_error_deg_half_cell": float(errors.sum()),
                "integrated_abs_phase_error_deg_half_cell": float(errors.abs().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_phase_correction_proxy(path: Path) -> pd.DataFrame:
    """Build a ratio table from existing cell/iris sweep summaries.

    This is explicitly a proxy because the table contains phase-advance errors
    for separate cell/iris perturbation positions, not a no-plunger before/after
    phase-delta pair.
    """

    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "dataset_id": "sim_sweep_260618_Nodalshift",
                    "marker_name": "not_available",
                    "metric_definition": "no usable nodal sweep summary table found",
                    "R_phase_proxy": np.nan,
                    "supports_iris_gt_cell": False,
                    "caveat": "direct DeltaPhi_I/DeltaPhi_C requires matched before/after plunger data",
                }
            ]
        )

    summary = pd.read_csv(path)
    rows = []
    for marker_name, group in summary.groupby("marker_name", sort=False):
        by_family = group.set_index("position_family")
        if not {"iris", "cell"}.issubset(set(by_family.index)):
            continue
        iris_error = float(by_family.loc["iris", "phase_err_rms_deg"])
        cell_error = float(by_family.loc["cell", "phase_err_rms_deg"])
        rows.append(
            {
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "marker_name": marker_name,
                "metric_definition": "abs phase-advance error ratio from existing nodal sweep; lower is better, not direct before/after sensitivity",
                "DeltaPhi_I_proxy_deg": iris_error,
                "DeltaPhi_C_proxy_deg": cell_error,
                "R_phase_proxy_abs_I_over_abs_C": iris_error / cell_error if cell_error else np.nan,
                "supports_iris_gt_cell": bool(iris_error > cell_error),
                "caveat": "not a direct DeltaPhi_I/DeltaPhi_C perturbation ratio; use only as consistency/proxy evidence",
            }
        )
    return pd.DataFrame(rows)


def build_full_sweep_rms_comparison(phase_advance: pd.DataFrame) -> pd.DataFrame:
    """Compare cell and iris phase-map residuals from a full-structure sweep table."""

    table = phase_advance.copy()
    table["wrapped_phase_error_from_240_deg"] = wrap_deg(
        table["phase_adv_deg"].astype(float).to_numpy() - TARGET_COUPLER_PHASE_DEG
    )
    rows = []
    for marker_name, marker_group in table.groupby("marker_name", sort=False):
        metrics = {}
        for family, group in marker_group.groupby("position_family"):
            errors = group["wrapped_phase_error_from_240_deg"].astype(float)
            metrics[family] = {
                "count": int(errors.count()),
                "rms": rms(errors),
                "mean_abs": float(errors.abs().mean()),
                "max_abs": float(errors.abs().max()),
            }
        if not {"cell", "iris"}.issubset(metrics):
            continue
        cell = metrics["cell"]
        iris = metrics["iris"]
        rows.append(
            {
                "dataset_id": str(marker_group["dataset_id"].iloc[0]),
                "marker_name": marker_name,
                "sigma_phi_cell_deg": cell["rms"],
                "sigma_phi_iris_deg": iris["rms"],
                "mean_abs_phi_cell_deg": cell["mean_abs"],
                "mean_abs_phi_iris_deg": iris["mean_abs"],
                "max_abs_phi_cell_deg": cell["max_abs"],
                "max_abs_phi_iris_deg": iris["max_abs"],
                "cell_transition_count": cell["count"],
                "iris_transition_count": iris["count"],
                "rms_improvement_factor_cell_over_iris": cell["rms"] / iris["rms"] if iris["rms"] else np.nan,
                "supports_iris_lower_rms": bool(iris["rms"] < cell["rms"]),
                "metric_definition": "full-structure sweep transition residuals wrapped against 240 deg target",
            }
        )
    return pd.DataFrame(rows)


def build_full_sweep_phase_response_ratio(phase_advance: pd.DataFrame) -> pd.DataFrame:
    """Compute iris/cell phase response ratios from full-structure transitions."""

    table = phase_advance.copy()
    table["signed_response_abs_deg"] = table["phase_step_deg"].astype(float).abs()
    table["advance_response_abs_deg"] = table["phase_adv_deg"].astype(float).abs()
    table["target_residual_abs_deg"] = wrap_deg(
        table["phase_adv_deg"].astype(float).to_numpy() - TARGET_COUPLER_PHASE_DEG
    )
    table["target_residual_abs_deg"] = table["target_residual_abs_deg"].abs()

    rows = []
    for marker_name, marker_group in table.groupby("marker_name", sort=False):
        metrics = {}
        for family, group in marker_group.groupby("position_family"):
            metrics[family] = {
                "transition_count": int(len(group)),
                "mean_abs_signed_phase_step_deg": float(group["signed_response_abs_deg"].mean()),
                "rms_signed_phase_step_deg": rms(group["phase_step_deg"]),
                "mean_phase_advance_0to360_deg": float(group["phase_adv_deg"].mean()),
                "mean_abs_target_residual_deg": float(group["target_residual_abs_deg"].mean()),
                "rms_target_residual_deg": rms(group["target_residual_abs_deg"]),
            }
        if not {"cell", "iris"}.issubset(metrics):
            continue
        cell = metrics["cell"]
        iris = metrics["iris"]
        rows.append(
            {
                "dataset_id": str(marker_group["dataset_id"].iloc[0]),
                "marker_name": marker_name,
                "DeltaPhi_I_response_mean_abs_signed_step_deg": iris["mean_abs_signed_phase_step_deg"],
                "DeltaPhi_C_response_mean_abs_signed_step_deg": cell["mean_abs_signed_phase_step_deg"],
                "R_phase_response_I_over_C": safe_ratio(
                    iris["mean_abs_signed_phase_step_deg"],
                    cell["mean_abs_signed_phase_step_deg"],
                ),
                "rms_signed_step_iris_deg": iris["rms_signed_phase_step_deg"],
                "rms_signed_step_cell_deg": cell["rms_signed_phase_step_deg"],
                "R_phase_response_rms_I_over_C": safe_ratio(
                    iris["rms_signed_phase_step_deg"],
                    cell["rms_signed_phase_step_deg"],
                ),
                "mean_phase_advance_iris_deg": iris["mean_phase_advance_0to360_deg"],
                "mean_phase_advance_cell_deg": cell["mean_phase_advance_0to360_deg"],
                "target_residual_mean_abs_iris_deg": iris["mean_abs_target_residual_deg"],
                "target_residual_mean_abs_cell_deg": cell["mean_abs_target_residual_deg"],
                "target_residual_ratio_iris_over_cell": safe_ratio(
                    iris["mean_abs_target_residual_deg"],
                    cell["mean_abs_target_residual_deg"],
                ),
                "iris_transition_count": iris["transition_count"],
                "cell_transition_count": cell["transition_count"],
                "supports_iris_larger_phase_response": bool(
                    iris["mean_abs_signed_phase_step_deg"] > cell["mean_abs_signed_phase_step_deg"]
                ),
                "supports_iris_lower_target_residual": bool(
                    iris["mean_abs_target_residual_deg"] < cell["mean_abs_target_residual_deg"]
                ),
                "metric_definition": (
                    "FullstructureSweep_ports_swapped cell/iris transition comparison. "
                    "R_phase_response uses mean abs signed phase step; residual ratio uses abs error from 240 deg."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_coupler_to_first_error(phase_advance: pd.DataFrame) -> pd.DataFrame:
    """Extract the first transition in each family as the coupler-to-first check."""

    table = phase_advance.copy()
    table["wrapped_error_from_240_deg"] = wrap_deg(
        table["phase_adv_deg"].astype(float).to_numpy() - TARGET_COUPLER_PHASE_DEG
    )
    rows = []
    for (marker_name, family), group in table.groupby(["marker_name", "position_family"], sort=False):
        first = group.sort_values(["pos_from", "pos_to"]).iloc[0]
        rows.append(
            {
                "dataset_id": first["dataset_id"],
                "marker_name": marker_name,
                "position_family": family,
                "from_tune_position": float(first["pos_from"]),
                "to_tune_position": float(first["pos_to"]),
                "phase_advance_0to360_deg": float(first["phase_adv_deg"]),
                "target_phase_advance_deg": TARGET_COUPLER_PHASE_DEG,
                "wrapped_error_from_240_deg": float(first["wrapped_error_from_240_deg"]),
                "abs_error_from_240_deg": abs(float(first["wrapped_error_from_240_deg"])),
                "convention_note": "240 deg is equivalent to -120 deg modulo 360",
            }
        )
    return pd.DataFrame(rows)


def build_half_cell_stations(params: dict[str, float]) -> list[HalfCellStation]:
    """Return alternating iris and regular-cell-center phase stations."""

    period = params["d"] + params["t"]
    stations: list[HalfCellStation] = []
    for idx in range(REGULAR_CELL_COUNT):
        iris_z = params["t"] / 2.0 + idx * period
        cell_z = params["t"] + params["d"] / 2.0 + idx * period
        stations.append(
            HalfCellStation(
                station_index=2 * idx,
                station_label="IN" if idx == 0 else f"I{idx}",
                station_kind="iris",
                z_mm=iris_z,
            )
        )
        stations.append(
            HalfCellStation(
                station_index=2 * idx + 1,
                station_label=f"R{idx + 1}",
                station_kind="cell_center",
                z_mm=cell_z,
            )
        )
    stations.append(
        HalfCellStation(
            station_index=2 * REGULAR_CELL_COUNT,
            station_label="OUT",
            station_kind="iris",
            z_mm=params["t"] / 2.0 + REGULAR_CELL_COUNT * period,
        )
    )
    return stations


def interpolate_profile(source_z: np.ndarray, source_y: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Interpolate a profile after sorting by z."""

    z_sorted, y_sorted = _sorted_xy(source_z, source_y)
    return np.interp(target_z, z_sorted, y_sorted)


def wrap_deg(values: np.ndarray | pd.Series) -> np.ndarray:
    """Wrap angle differences to [-180, 180) degrees."""

    array = np.asarray(values, dtype=float)
    return (array + 180.0) % 360.0 - 180.0


def rms(values: pd.Series | np.ndarray) -> float:
    """Return root-mean-square of finite values."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(array**2))) if len(array) else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, guarding zero denominators."""

    return float(numerator / denominator) if denominator else float("nan")


def _single_profile_trace(
    export: FieldProfileExport,
    *,
    expected_kind: str,
) -> FieldProfileTrace:
    traces = [trace for trace in export.traces if _normalized_field_kind(trace.field_kind) == expected_kind]
    if not traces:
        raise ValueError(f"No {expected_kind} profile trace found")
    return traces[0]


def _single_phase_trace(
    export: FieldPhaseExport,
    *,
    expected_kind: str,
) -> FieldPhaseTrace:
    traces = [trace for trace in export.traces if _normalized_field_kind(trace.field_kind) == expected_kind]
    if not traces:
        raise ValueError(f"No {expected_kind} phase trace found")
    return traces[0]


def _normalized_field_kind(field_kind: str) -> str:
    text = field_kind.strip().lower().replace("_", "-")
    if text in {"e", "e-field", "electric", "electric-field"}:
        return "e"
    if text in {"h", "h-field", "magnetic", "magnetic-field"}:
        return "h"
    return text


def _sorted_xy(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_values)
    return np.asarray(x_values, dtype=float)[order], np.asarray(y_values, dtype=float)[order]
