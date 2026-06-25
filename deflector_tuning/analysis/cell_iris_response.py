"""Compare matched cell-center and iris-center response transitions."""

from __future__ import annotations

import math

import pandas as pd

GROUP_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
    "target_freq_ghz",
    "operation_mode_deg",
    "operation_axes_deg",
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
    "target_freq_ghz",
    "operation_mode_deg",
    "operation_axes_deg",
    "transition_pair_index",
    "cell_from_source_file",
    "cell_to_source_file",
    "iris_from_source_file",
    "iris_to_source_file",
    "cell_from_tune_position",
    "cell_to_tune_position",
    "iris_from_tune_position",
    "iris_to_tune_position",
    "cell_operation_scaled_admittance_delta_abs",
    "iris_operation_scaled_admittance_delta_abs",
    "operation_scaled_admittance_response_ratio_iris_over_cell",
    "cell_signed_phase_step_deg",
    "iris_signed_phase_step_deg",
    "phase_step_response_ratio_iris_over_cell",
    "target_phase_advance_deg",
    "cell_phase_advance_0to360_deg",
    "iris_phase_advance_0to360_deg",
    "cell_phase_residual_from_target_deg",
    "iris_phase_residual_from_target_deg",
    "cell_abs_operation_axis_error_deg",
    "iris_abs_operation_axis_error_deg",
    "supports_iris_larger_admittance_response",
    "supports_iris_better_branch_alignment",
    "supports_iris_lower_phase_residual",
]


def compare_cell_and_iris_responses(
    kyhl_admittance_transitions: pd.DataFrame,
    *,
    target_phase_advance_deg: float = 240.0,
) -> pd.DataFrame:
    """Pair cell-center and iris-center transitions and compare responses.

    The input is the KYHL transition table produced from raw S11 marker points.
    The tan(theta0/2) scaling is treated as a derived operation-mode coordinate;
    this function compares delta magnitudes in that coordinate without changing
    the original measured/simulated S-parameter samples.
    """

    output_columns = _output_columns(kyhl_admittance_transitions)
    if kyhl_admittance_transitions.empty:
        return pd.DataFrame(columns=output_columns)

    table = kyhl_admittance_transitions.copy()
    required_columns = {
        "position_family",
        "from_tune_position",
        "to_tune_position",
        "delta_operation_scaled_admittance_real",
        "delta_operation_scaled_admittance_imag",
        "raw_signed_phase_step_deg",
        "raw_phase_advance_0to360_deg",
    }
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"kyhl_admittance_transitions is missing required columns: {missing}")

    table["_from_tune_sort"] = pd.to_numeric(table["from_tune_position"], errors="coerce")
    table["_to_tune_sort"] = pd.to_numeric(table["to_tune_position"], errors="coerce")
    table = table[table["_from_tune_sort"].notna() & table["_to_tune_sort"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, object]] = []
    grouping_columns = [*GROUP_COLUMNS, *_metadata_columns(table)]
    grouping_columns = [column for column in grouping_columns if column in table]
    for group_values, group in table.groupby(grouping_columns, dropna=False, sort=False):
        group_metadata = dict(zip(grouping_columns, _as_tuple(group_values), strict=True))
        cell_rows = _family_rows(group, "cell")
        iris_rows = _family_rows(group, "iris")
        pair_count = min(len(cell_rows), len(iris_rows))
        for pair_index in range(pair_count):
            cell = cell_rows.iloc[pair_index]
            iris = iris_rows.iloc[pair_index]
            cell_delta_abs = _operation_delta_abs(cell)
            iris_delta_abs = _operation_delta_abs(iris)
            cell_phase_abs = abs(float(cell["raw_signed_phase_step_deg"]))
            iris_phase_abs = abs(float(iris["raw_signed_phase_step_deg"]))
            cell_phase_residual = abs(wrap180(float(cell["raw_phase_advance_0to360_deg"]) - target_phase_advance_deg))
            iris_phase_residual = abs(wrap180(float(iris["raw_phase_advance_0to360_deg"]) - target_phase_advance_deg))
            cell_axis_error = _abs_axis_error(cell)
            iris_axis_error = _abs_axis_error(iris)
            rows.append(
                {
                    **group_metadata,
                    "transition_pair_index": pair_index + 1,
                    "cell_from_source_file": cell.get("from_source_file", pd.NA),
                    "cell_to_source_file": cell.get("to_source_file", pd.NA),
                    "iris_from_source_file": iris.get("from_source_file", pd.NA),
                    "iris_to_source_file": iris.get("to_source_file", pd.NA),
                    "cell_from_tune_position": cell["from_tune_position"],
                    "cell_to_tune_position": cell["to_tune_position"],
                    "iris_from_tune_position": iris["from_tune_position"],
                    "iris_to_tune_position": iris["to_tune_position"],
                    "cell_operation_scaled_admittance_delta_abs": cell_delta_abs,
                    "iris_operation_scaled_admittance_delta_abs": iris_delta_abs,
                    "operation_scaled_admittance_response_ratio_iris_over_cell": safe_ratio(
                        iris_delta_abs,
                        cell_delta_abs,
                    ),
                    "cell_signed_phase_step_deg": float(cell["raw_signed_phase_step_deg"]),
                    "iris_signed_phase_step_deg": float(iris["raw_signed_phase_step_deg"]),
                    "phase_step_response_ratio_iris_over_cell": safe_ratio(iris_phase_abs, cell_phase_abs),
                    "target_phase_advance_deg": float(target_phase_advance_deg),
                    "cell_phase_advance_0to360_deg": float(cell["raw_phase_advance_0to360_deg"]),
                    "iris_phase_advance_0to360_deg": float(iris["raw_phase_advance_0to360_deg"]),
                    "cell_phase_residual_from_target_deg": cell_phase_residual,
                    "iris_phase_residual_from_target_deg": iris_phase_residual,
                    "cell_abs_operation_axis_error_deg": cell_axis_error,
                    "iris_abs_operation_axis_error_deg": iris_axis_error,
                    "supports_iris_larger_admittance_response": bool(iris_delta_abs > cell_delta_abs),
                    "supports_iris_better_branch_alignment": bool(iris_axis_error < cell_axis_error),
                    "supports_iris_lower_phase_residual": bool(iris_phase_residual < cell_phase_residual),
                }
            )

    output = pd.DataFrame(rows, columns=output_columns)
    for column in (
        "supports_iris_larger_admittance_response",
        "supports_iris_better_branch_alignment",
        "supports_iris_lower_phase_residual",
    ):
        if column in output:
            output[column] = output[column].astype(object)
    return output


def wrap180(angle_deg: float) -> float:
    """Wrap an angle difference to [-180, 180) degrees."""

    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator and guard zero denominators."""

    return float(numerator / denominator) if denominator else float("nan")


def _family_rows(table: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = table[table["position_family"] == family].copy()
    return rows.sort_values(["_from_tune_sort", "_to_tune_sort"], kind="mergesort").reset_index(drop=True)


def _operation_delta_abs(row: pd.Series) -> float:
    if "operation_scaled_admittance_delta_abs" in row and pd.notna(row["operation_scaled_admittance_delta_abs"]):
        return float(row["operation_scaled_admittance_delta_abs"])
    real = float(row["delta_operation_scaled_admittance_real"])
    imag = float(row["delta_operation_scaled_admittance_imag"])
    return float(math.hypot(real, imag))


def _abs_axis_error(row: pd.Series) -> float:
    if "abs_operation_axis_error_deg" in row and pd.notna(row["abs_operation_axis_error_deg"]):
        return float(row["abs_operation_axis_error_deg"])
    return abs(float(row.get("operation_axis_error_deg", float("nan"))))


def _metadata_columns(table: pd.DataFrame) -> list[str]:
    excluded = set(OUTPUT_COLUMNS) | TUNE_SWEEP_METADATA_COLUMNS
    return [column for column in table.columns if column.startswith("sim_") and column not in excluded]


def _output_columns(table: pd.DataFrame) -> list[str]:
    insert_index = OUTPUT_COLUMNS.index("transition_pair_index")
    metadata_columns = _metadata_columns(table)
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
