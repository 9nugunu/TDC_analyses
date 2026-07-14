"""Compare matched cell-center and iris-center response transitions."""

from __future__ import annotations

import math

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
    "freq_target_ghz",
    "op_mode_deg",
    "op_admit_axes_deg",
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
    "freq_target_ghz",
    "op_mode_deg",
    "op_admit_axes_deg",
    "pair_index",
    "cell_file_from",
    "cell_file_to",
    "iris_file_from",
    "iris_file_to",
    "cell_pos_from",
    "cell_pos_to",
    "iris_pos_from",
    "iris_pos_to",
    "cell_admit_delta_mag",
    "iris_admit_delta_mag",
    "admit_ratio_iris_cell",
    "cell_phase_step_deg",
    "iris_phase_step_deg",
    "phase_ratio_iris_cell",
    "phase_target_deg",
    "cell_phase_adv_deg",
    "iris_phase_adv_deg",
    "cell_phase_err_deg",
    "iris_phase_err_deg",
    "cell_admit_axis_err_abs_deg",
    "iris_admit_axis_err_abs_deg",
    "supports_iris_admit",
    "supports_iris_axis",
    "supports_iris_phase",
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
        "pos_from",
        "pos_to",
        "op_admit_delta_re",
        "op_admit_delta_im",
        "phase_step_deg",
        "phase_adv_deg",
    }
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"kyhl_admittance_transitions is missing required columns: {missing}")

    table["_from_tune_sort"] = pd.to_numeric(table["pos_from"], errors="coerce")
    table["_to_tune_sort"] = pd.to_numeric(table["pos_to"], errors="coerce")
    table = table[table["_from_tune_sort"].notna() & table["_to_tune_sort"].notna()].copy()
    allowed_transition = table.apply(
        lambda row: is_allowed_coupler_cavity_transition(row["pos_from"], row["pos_to"]),
        axis=1,
    )
    table = table[allowed_transition].copy()
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
            cell_phase_abs = abs(float(cell["phase_step_deg"]))
            iris_phase_abs = abs(float(iris["phase_step_deg"]))
            cell_phase_residual = abs(wrap180(float(cell["phase_adv_deg"]) - target_phase_advance_deg))
            iris_phase_residual = abs(wrap180(float(iris["phase_adv_deg"]) - target_phase_advance_deg))
            cell_axis_error = _abs_axis_error(cell)
            iris_axis_error = _abs_axis_error(iris)
            rows.append(
                {
                    **group_metadata,
                    "pair_index": pair_index + 1,
                    "cell_file_from": cell.get("file_from", pd.NA),
                    "cell_file_to": cell.get("file_to", pd.NA),
                    "iris_file_from": iris.get("file_from", pd.NA),
                    "iris_file_to": iris.get("file_to", pd.NA),
                    "cell_pos_from": cell["pos_from"],
                    "cell_pos_to": cell["pos_to"],
                    "iris_pos_from": iris["pos_from"],
                    "iris_pos_to": iris["pos_to"],
                    "cell_admit_delta_mag": cell_delta_abs,
                    "iris_admit_delta_mag": iris_delta_abs,
                    "admit_ratio_iris_cell": safe_ratio(
                        iris_delta_abs,
                        cell_delta_abs,
                    ),
                    "cell_phase_step_deg": float(cell["phase_step_deg"]),
                    "iris_phase_step_deg": float(iris["phase_step_deg"]),
                    "phase_ratio_iris_cell": safe_ratio(iris_phase_abs, cell_phase_abs),
                    "phase_target_deg": float(target_phase_advance_deg),
                    "cell_phase_adv_deg": float(cell["phase_adv_deg"]),
                    "iris_phase_adv_deg": float(iris["phase_adv_deg"]),
                    "cell_phase_err_deg": cell_phase_residual,
                    "iris_phase_err_deg": iris_phase_residual,
                    "cell_admit_axis_err_abs_deg": cell_axis_error,
                    "iris_admit_axis_err_abs_deg": iris_axis_error,
                    "supports_iris_admit": bool(iris_delta_abs > cell_delta_abs),
                    "supports_iris_axis": bool(iris_axis_error < cell_axis_error),
                    "supports_iris_phase": bool(iris_phase_residual < cell_phase_residual),
                }
            )

    output = pd.DataFrame(rows, columns=output_columns)
    for column in (
        "supports_iris_admit",
        "supports_iris_axis",
        "supports_iris_phase",
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
    if "op_admit_delta_mag" in row and pd.notna(row["op_admit_delta_mag"]):
        return float(row["op_admit_delta_mag"])
    real = float(row["op_admit_delta_re"])
    imag = float(row["op_admit_delta_im"])
    return float(math.hypot(real, imag))


def _abs_axis_error(row: pd.Series) -> float:
    if "op_admit_axis_err_abs_deg" in row and pd.notna(row["op_admit_axis_err_abs_deg"]):
        return float(row["op_admit_axis_err_abs_deg"])
    return abs(float(row.get("op_admit_axis_err_deg", float("nan"))))


def _metadata_columns(table: pd.DataFrame) -> list[str]:
    excluded = set(OUTPUT_COLUMNS) | TUNE_SWEEP_METADATA_COLUMNS
    return [column for column in table.columns if column.startswith("sim_") and column not in excluded]


def _output_columns(table: pd.DataFrame) -> list[str]:
    insert_index = OUTPUT_COLUMNS.index("pair_index")
    metadata_columns = _metadata_columns(table)
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
