"""KYHL operation-mode admittance transition metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from deflector_tuning.analysis.coupler_position_gate import is_allowed_coupler_cavity_transition

GROUP_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
]
TUNE_SWEEP_METADATA_COLUMNS: frozenset[str] = frozenset({"sim_NumTune", "sim_NumDepth"})
OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
    "position_family",
    "from_source_file",
    "to_source_file",
    "from_tune_position",
    "to_tune_position",
    "target_freq_ghz",
    "from_freq_ghz",
    "to_freq_ghz",
    "operation_mode_deg",
    "operation_mode_scale",
    "axis_sign",
    "operation_axes_deg",
    "from_gamma_real",
    "from_gamma_imag",
    "to_gamma_real",
    "to_gamma_imag",
    "from_raw_pseudo_admittance_real",
    "from_raw_pseudo_admittance_imag",
    "to_raw_pseudo_admittance_real",
    "to_raw_pseudo_admittance_imag",
    "from_admittance_real",
    "from_admittance_imag",
    "to_admittance_real",
    "to_admittance_imag",
    "delta_raw_pseudo_admittance_real",
    "delta_raw_pseudo_admittance_imag",
    "delta_admittance_real",
    "delta_admittance_imag",
    "delta_operation_scaled_admittance_real",
    "delta_operation_scaled_admittance_imag",
    "operation_scaled_admittance_delta_abs",
    "operation_scaled_admittance_angle_deg",
    "kyhl_operation_real",
    "kyhl_operation_imag",
    "kyhl_operation_angle_deg",
    "nearest_operation_axis_deg",
    "operation_axis_error_deg",
    "abs_operation_axis_error_deg",
    "raw_signed_phase_step_deg",
    "raw_phase_advance_0to360_deg",
    "convention_note",
]
POINT_OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
    "source_file",
    "tune_position",
    "target_freq_ghz",
    "freq_ghz",
    "operation_mode_deg",
    "operation_mode_scale",
    "axis_sign",
    "operation_axes_deg",
    "gamma_real",
    "gamma_imag",
    "raw_pseudo_admittance_real",
    "raw_pseudo_admittance_imag",
    "admittance_real",
    "admittance_imag",
    "operation_scaled_admittance_real",
    "operation_scaled_admittance_imag",
    "operation_scaled_admittance_abs",
    "operation_scaled_admittance_angle_deg",
    "kyhl_operation_real",
    "kyhl_operation_imag",
    "kyhl_operation_angle_deg",
    "nearest_operation_axis_deg",
    "operation_axis_error_deg",
    "abs_operation_axis_error_deg",
    "raw_s_phase_deg",
    "convention_note",
]


def compute_kyhl_admittance_points(
    marker_points: pd.DataFrame,
    *,
    operation_mode_deg: float = 120.0,
    axis_sign: int = 1,
) -> pd.DataFrame:
    """Return one KYHL operation-mode admittance row per sampled marker point."""

    axes = operation_mode_axes(operation_mode_deg)
    scale = operation_mode_scale(operation_mode_deg)
    axis_sign = _validate_axis_sign(axis_sign)
    table = marker_points.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    if "tune_position" not in table:
        table["tune_position"] = pd.NA
    output_columns = _point_output_columns(table)

    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table[table["_tune_sort"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=output_columns)
    table = _add_admittance_columns(table)

    axes_text = _format_axes(axes)
    rows: list[dict[str, object]] = []
    for _, point in table.sort_values(_point_sort_columns(table), kind="mergesort").iterrows():
        kyhl_value = complex(float(point["_admittance_real"]), axis_sign * scale * float(point["_admittance_imag"]))
        kyhl_angle = _wrap360(math.degrees(math.atan2(kyhl_value.imag, kyhl_value.real)))
        nearest_axis = nearest_operation_axis(kyhl_angle, axes)
        axis_error = wrap180(kyhl_angle - nearest_axis)
        rows.append(
            {
                **{column: point[column] for column in GROUP_COLUMNS},
                **{column: point[column] for column in _metadata_columns(table)},
                "source_file": point.get("source_file", pd.NA),
                "tune_position": point["tune_position"],
                "target_freq_ghz": point.get("target_freq_ghz", pd.NA),
                "freq_ghz": point.get("freq_ghz", pd.NA),
                "operation_mode_deg": float(operation_mode_deg),
                "operation_mode_scale": scale,
                "axis_sign": axis_sign,
                "operation_axes_deg": axes_text,
                "gamma_real": float(point["_gamma_real"]),
                "gamma_imag": float(point["_gamma_imag"]),
                "raw_pseudo_admittance_real": float(point["_admittance_real"]),
                "raw_pseudo_admittance_imag": float(point["_admittance_imag"]),
                "admittance_real": float(point["_admittance_real"]),
                "admittance_imag": float(point["_admittance_imag"]),
                "operation_scaled_admittance_real": float(kyhl_value.real),
                "operation_scaled_admittance_imag": float(kyhl_value.imag),
                "operation_scaled_admittance_abs": float(abs(kyhl_value)),
                "operation_scaled_admittance_angle_deg": kyhl_angle,
                "kyhl_operation_real": float(kyhl_value.real),
                "kyhl_operation_imag": float(kyhl_value.imag),
                "kyhl_operation_angle_deg": kyhl_angle,
                "nearest_operation_axis_deg": nearest_axis,
                "operation_axis_error_deg": axis_error,
                "abs_operation_axis_error_deg": abs(axis_error),
                "raw_s_phase_deg": point.get("s_phase_deg", pd.NA),
                "convention_note": (
                    "KYHL point branch target comes from operation_mode_deg/2 + n*operation_mode_deg; "
                    "raw S11 is not de-embedded or CST-renormalized here; operation_scaled_admittance "
                    "is a derived comparison coordinate"
                ),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def compute_kyhl_admittance_transitions(
    marker_points: pd.DataFrame,
    *,
    operation_mode_deg: float = 120.0,
    axis_sign: int = 1,
) -> pd.DataFrame:
    """Compute KYHL transition angles in an operation-mode admittance plane.

    The transform keeps raw Touchstone samples untouched and creates a derived
    pseudo-admittance coordinate for KYHL branch matching:

    ``y = (1 - gamma) / (1 + gamma)``

    The operation mode scales the admittance-plane imaginary component by
    ``tan(operation_mode / 2)``. For 2pi/3 operation this is ``sqrt(3)`` and the
    branch axes are 60, 180, and 300 degrees.
    """

    axes = operation_mode_axes(operation_mode_deg)
    scale = operation_mode_scale(operation_mode_deg)
    axis_sign = _validate_axis_sign(axis_sign)
    table = marker_points.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    if "tune_position" not in table:
        table["tune_position"] = pd.NA
    output_columns = _output_columns(table)

    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table[table["_tune_sort"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=output_columns)
    table["position_family"] = table["_tune_sort"].map(_position_family)
    table = _add_admittance_columns(table)

    rows: list[dict[str, object]] = []
    grouping_columns = [*GROUP_COLUMNS, *_metadata_columns(table), "position_family"]
    axes_text = _format_axes(axes)
    for group_values, group in table.groupby(grouping_columns, dropna=False, sort=False):
        group = group.sort_values("_tune_sort", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue
        group_metadata = dict(zip(grouping_columns, group_values, strict=True))
        for index in range(len(group) - 1):
            start = group.iloc[index]
            end = group.iloc[index + 1]
            if not is_allowed_coupler_cavity_transition(start["tune_position"], end["tune_position"]):
                continue
            delta_y = complex(
                float(end["_admittance_real"]) - float(start["_admittance_real"]),
                float(end["_admittance_imag"]) - float(start["_admittance_imag"]),
            )
            kyhl_value = complex(delta_y.real, axis_sign * scale * delta_y.imag)
            kyhl_angle = _wrap360(math.degrees(math.atan2(kyhl_value.imag, kyhl_value.real)))
            nearest_axis = nearest_operation_axis(kyhl_angle, axes)
            axis_error = wrap180(kyhl_angle - nearest_axis)
            raw_signed_phase_step = wrap180(float(end["s_phase_deg"]) - float(start["s_phase_deg"]))
            rows.append(
                {
                    **group_metadata,
                    "from_source_file": start.get("source_file", pd.NA),
                    "to_source_file": end.get("source_file", pd.NA),
                    "from_tune_position": start["tune_position"],
                    "to_tune_position": end["tune_position"],
                    "target_freq_ghz": start.get("target_freq_ghz", pd.NA),
                    "from_freq_ghz": start.get("freq_ghz", pd.NA),
                    "to_freq_ghz": end.get("freq_ghz", pd.NA),
                    "operation_mode_deg": float(operation_mode_deg),
                    "operation_mode_scale": scale,
                    "axis_sign": axis_sign,
                    "operation_axes_deg": axes_text,
                    "from_gamma_real": float(start["_gamma_real"]),
                    "from_gamma_imag": float(start["_gamma_imag"]),
                    "to_gamma_real": float(end["_gamma_real"]),
                    "to_gamma_imag": float(end["_gamma_imag"]),
                    "from_raw_pseudo_admittance_real": float(start["_admittance_real"]),
                    "from_raw_pseudo_admittance_imag": float(start["_admittance_imag"]),
                    "to_raw_pseudo_admittance_real": float(end["_admittance_real"]),
                    "to_raw_pseudo_admittance_imag": float(end["_admittance_imag"]),
                    "from_admittance_real": float(start["_admittance_real"]),
                    "from_admittance_imag": float(start["_admittance_imag"]),
                    "to_admittance_real": float(end["_admittance_real"]),
                    "to_admittance_imag": float(end["_admittance_imag"]),
                    "delta_raw_pseudo_admittance_real": float(delta_y.real),
                    "delta_raw_pseudo_admittance_imag": float(delta_y.imag),
                    "delta_admittance_real": float(delta_y.real),
                    "delta_admittance_imag": float(delta_y.imag),
                    "delta_operation_scaled_admittance_real": float(kyhl_value.real),
                    "delta_operation_scaled_admittance_imag": float(kyhl_value.imag),
                    "operation_scaled_admittance_delta_abs": float(abs(kyhl_value)),
                    "operation_scaled_admittance_angle_deg": kyhl_angle,
                    "kyhl_operation_real": float(kyhl_value.real),
                    "kyhl_operation_imag": float(kyhl_value.imag),
                    "kyhl_operation_angle_deg": kyhl_angle,
                    "nearest_operation_axis_deg": nearest_axis,
                    "operation_axis_error_deg": axis_error,
                    "abs_operation_axis_error_deg": abs(axis_error),
                    "raw_signed_phase_step_deg": raw_signed_phase_step,
                    "raw_phase_advance_0to360_deg": raw_signed_phase_step % 360.0,
                    "convention_note": (
                        "KYHL branch target comes from operation_mode_deg/2 + n*operation_mode_deg; "
                        "raw S11 is not de-embedded or CST-renormalized here; operation_scaled_admittance "
                        "is a derived comparison coordinate"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def operation_mode_axes(operation_mode_deg: float) -> tuple[float, ...]:
    """Return branch axes for an operation phase advance."""

    operation_mode_deg = _validate_operation_mode(operation_mode_deg)
    offset = operation_mode_deg / 2.0
    count = max(1, math.ceil(360.0 / operation_mode_deg))
    axes: list[float] = []
    for index in range(count):
        axis = _wrap360(offset + index * operation_mode_deg)
        if not any(abs(wrap180(axis - existing)) < 1e-9 for existing in axes):
            axes.append(axis)
    return tuple(sorted(axes))


def operation_mode_scale(operation_mode_deg: float) -> float:
    """Return the KYHL admittance-plane scale for an operation mode."""

    operation_mode_deg = _validate_operation_mode(operation_mode_deg)
    scale = math.tan(math.radians(operation_mode_deg) / 2.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"operation_mode_deg produces an invalid KYHL scale: {operation_mode_deg:g}")
    return float(scale)


def nearest_operation_axis(angle_deg: float, axes_deg: tuple[float, ...]) -> float:
    """Return the operation axis with the smallest wrapped angular error."""

    if not axes_deg:
        raise ValueError("axes_deg must not be empty")
    angle_deg = _wrap360(float(angle_deg))
    return min(axes_deg, key=lambda axis: abs(wrap180(angle_deg - axis)))


def wrap180(angle_deg: float) -> float:
    """Wrap an angle difference to [-180, 180) degrees."""

    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _complex_gamma(table: pd.DataFrame) -> np.ndarray:
    if {"s_real", "s_imag"}.issubset(table.columns):
        real = pd.to_numeric(table["s_real"], errors="coerce")
        imag = pd.to_numeric(table["s_imag"], errors="coerce")
        has_complex = real.notna() & imag.notna()
    else:
        real = pd.Series(np.nan, index=table.index)
        imag = pd.Series(np.nan, index=table.index)
        has_complex = pd.Series(False, index=table.index)

    if {"s_db", "s_phase_deg"}.issubset(table.columns):
        magnitude = 10.0 ** (pd.to_numeric(table["s_db"], errors="coerce") / 20.0)
        phase_rad = np.deg2rad(pd.to_numeric(table["s_phase_deg"], errors="coerce"))
        fallback = magnitude.to_numpy(dtype=float) * np.exp(1j * phase_rad.to_numpy(dtype=float))
        has_fallback = magnitude.notna() & phase_rad.notna()
    else:
        fallback = np.full(len(table), np.nan + 1j * np.nan, dtype=complex)
        has_fallback = pd.Series(False, index=table.index)

    usable = has_complex | has_fallback
    if not bool(usable.all()):
        raise ValueError("marker_points requires s_real/s_imag or s_db/s_phase_deg for KYHL admittance metrics")

    gamma = fallback
    gamma[has_complex.to_numpy()] = real[has_complex].to_numpy(dtype=float) + 1j * imag[has_complex].to_numpy(dtype=float)
    return gamma


def _add_admittance_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    gamma = _complex_gamma(table)
    admittance = _pseudo_admittance(gamma)
    table["_gamma_real"] = gamma.real
    table["_gamma_imag"] = gamma.imag
    table["_admittance_real"] = admittance.real
    table["_admittance_imag"] = admittance.imag
    return table


def _pseudo_admittance(gamma: np.ndarray) -> np.ndarray:
    denominator = 1.0 + gamma
    output = np.full(gamma.shape, np.nan + 1j * np.nan, dtype=complex)
    np.divide(1.0 - gamma, denominator, out=output, where=np.abs(denominator) > 1e-15)
    return output


def _validate_operation_mode(operation_mode_deg: float) -> float:
    operation_mode_deg = float(operation_mode_deg)
    if not 0.0 < operation_mode_deg < 360.0:
        raise ValueError("operation_mode_deg must be between 0 and 360 degrees")
    return operation_mode_deg


def _validate_axis_sign(axis_sign: int) -> int:
    axis_sign = int(axis_sign)
    if axis_sign not in {-1, 1}:
        raise ValueError("axis_sign must be -1 or 1")
    return axis_sign


def _metadata_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if column not in GROUP_COLUMNS
        and column not in TUNE_SWEEP_METADATA_COLUMNS
        and column not in OUTPUT_COLUMNS
        and column.startswith("sim_")
    ]


def _output_columns(table: pd.DataFrame) -> list[str]:
    metadata_columns = _metadata_columns(table)
    insert_index = OUTPUT_COLUMNS.index("position_family")
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _point_output_columns(table: pd.DataFrame) -> list[str]:
    metadata_columns = _metadata_columns(table)
    insert_index = POINT_OUTPUT_COLUMNS.index("source_file")
    return [*POINT_OUTPUT_COLUMNS[:insert_index], *metadata_columns, *POINT_OUTPUT_COLUMNS[insert_index:]]


def _point_sort_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "dataset_id",
            "marker_name",
            "s_name",
            "port_side",
            "_tune_sort",
            "source_file",
        )
        if column in table
    ]


def _position_family(tune_position: float) -> str:
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{fractional:g}"


def _wrap360(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


def _format_axes(axes_deg: tuple[float, ...]) -> str:
    return ";".join(f"{axis:g}" for axis in axes_deg)
