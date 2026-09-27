"""Geometry-sweep phase response derived from marker-sampled S11 points."""

from __future__ import annotations

import numpy as np
import pandas as pd

TUNE_SWEEP_METADATA_COLUMNS: frozenset[str] = frozenset({"sim_NumDepth", "sim_NumTune"})
GROUP_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
)
OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
    "sweep_axis",
    "sweep_value",
    "sweep_base",
    "cell_file",
    "iris_file",
    "cell_pos",
    "iris_pos",
    "cell_phase_deg",
    "iris_phase_deg",
    "cell_phase_shift_deg",
    "iris_phase_shift_deg",
    "cell_iris_phase_delta_deg",
    "cell_iris_delta_shift_deg",
]


def compute_geometry_phase_response(
    marker_points: pd.DataFrame,
    *,
    sweep_axis: str | None = None,
    sweep_base: float | None = None,
) -> pd.DataFrame:
    """Return marker phase pickup across a one-dimensional geometry sweep.

    The response is enabled for simulation-style sweeps with one varying
    numeric ``sim_*`` geometry axis and paired cell/iris marker points at each
    axis value. Dataset names are intentionally ignored.
    """

    if marker_points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = [column for column in ("marker_name", "s_phase_deg", "tune_position") if column not in marker_points]
    if missing:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if sweep_base is not None and sweep_axis is None:
        raise ValueError("sweep_base requires an explicit sweep_axis")

    table = marker_points.copy()
    sweep_axes = (
        [_validate_explicit_sweep_axis(table, sweep_axis)]
        if sweep_axis is not None
        else _varying_geometry_axes(table)
    )
    if len(sweep_axes) != 1:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    sweep_axis = sweep_axes[0]

    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table["_sweep_value"] = pd.to_numeric(table[sweep_axis], errors="coerce")
    table = table[table["_tune_sort"].notna() & table["_sweep_value"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    table["position_family"] = table["_tune_sort"].map(_position_family)
    table = table[table["position_family"].isin({"cell", "iris"})].copy()
    if table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    metadata_columns = _constant_geometry_metadata_columns(table, sweep_axis)
    rows: list[dict[str, object]] = []
    grouping_columns = [*GROUP_COLUMNS, *metadata_columns]
    for group_values, group in table.groupby(grouping_columns, dropna=False, sort=False):
        response_rows = _response_rows_for_group(
            group,
            sweep_axis=sweep_axis,
            sweep_base=sweep_base,
            group_metadata=dict(zip(grouping_columns, group_values, strict=True)),
        )
        rows.extend(response_rows)

    if not rows:
        return pd.DataFrame(columns=[*_output_columns(metadata_columns)])
    return (
        pd.DataFrame(rows, columns=_output_columns(metadata_columns))
        .sort_values(["marker_name", "sweep_axis", "sweep_value"], kind="mergesort")
        .reset_index(drop=True)
    )


def _response_rows_for_group(
    group: pd.DataFrame,
    *,
    sweep_axis: str,
    sweep_base: float | None,
    group_metadata: dict[str, object],
) -> list[dict[str, object]]:
    pair_rows = []
    for sweep_value, sweep_group in group.groupby("_sweep_value", dropna=False, sort=True):
        cell = _first_family_row(sweep_group, "cell")
        iris = _first_family_row(sweep_group, "iris")
        if cell is None or iris is None:
            continue
        pair_rows.append(
            {
                **group_metadata,
                "sweep_axis": sweep_axis,
                "sweep_value": float(sweep_value),
                "cell_file": cell.get("source_file", pd.NA),
                "iris_file": iris.get("source_file", pd.NA),
                "cell_pos": cell["tune_position"],
                "iris_pos": iris["tune_position"],
                "cell_phase_deg": float(cell["s_phase_deg"]),
                "iris_phase_deg": float(iris["s_phase_deg"]),
                "cell_iris_phase_delta_deg": _wrap180(float(iris["s_phase_deg"]) - float(cell["s_phase_deg"])),
            }
        )
    if not pair_rows:
        return []

    if sweep_base is None:
        baseline = min(
            pair_rows,
            key=lambda row: (
                abs(float(row["sweep_value"])),
                float(row["sweep_value"]),
            ),
        )
    else:
        matches = [
            row
            for row in pair_rows
            if np.isclose(
                float(row["sweep_value"]),
                float(sweep_base),
                rtol=0.0,
                atol=1e-9,
            )
        ]
        if not matches:
            raise ValueError(
                f"sweep_base {sweep_base:g} is not a sampled value of {sweep_axis}"
            )
        baseline = matches[0]
    baseline_sweep = float(baseline["sweep_value"])
    baseline_cell_phase = float(baseline["cell_phase_deg"])
    baseline_iris_phase = float(baseline["iris_phase_deg"])
    baseline_delta = float(baseline["cell_iris_phase_delta_deg"])
    for row in pair_rows:
        row["sweep_base"] = baseline_sweep
        row["cell_phase_shift_deg"] = _wrap180(float(row["cell_phase_deg"]) - baseline_cell_phase)
        row["iris_phase_shift_deg"] = _wrap180(float(row["iris_phase_deg"]) - baseline_iris_phase)
        row["cell_iris_delta_shift_deg"] = _wrap180(
            float(row["cell_iris_phase_delta_deg"]) - baseline_delta
        )
    return pair_rows


def _first_family_row(group: pd.DataFrame, family: str) -> pd.Series | None:
    rows = group[group["position_family"] == family].sort_values(["_tune_sort", "source_file"], kind="mergesort")
    if rows.empty:
        return None
    return rows.iloc[0]


def _varying_geometry_axes(table: pd.DataFrame) -> list[str]:
    columns = []
    for column in table.columns:
        if not column.startswith("sim_") or column in TUNE_SWEEP_METADATA_COLUMNS:
            continue
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        if values.nunique() > 1:
            columns.append(column)
    return columns


def _validate_explicit_sweep_axis(table: pd.DataFrame, sweep_axis: str) -> str:
    if sweep_axis not in table:
        raise ValueError(f"explicit sweep axis is missing: {sweep_axis}")
    values = pd.to_numeric(table[sweep_axis], errors="coerce").dropna()
    if values.nunique() <= 1:
        raise ValueError(f"explicit sweep axis must vary: {sweep_axis}")
    return sweep_axis


def _constant_geometry_metadata_columns(table: pd.DataFrame, sweep_axis: str) -> list[str]:
    columns = []
    for column in table.columns:
        if not column.startswith("sim_") or column in TUNE_SWEEP_METADATA_COLUMNS or column == sweep_axis:
            continue
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        if values.nunique() <= 1:
            columns.append(column)
    return columns


def _output_columns(metadata_columns: list[str]) -> list[str]:
    insert_index = OUTPUT_COLUMNS.index("sweep_axis")
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _position_family(tune_position: float) -> str:
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{fractional:g}"


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0
