"""Compute local phase advance from marker-sampled S-parameter points."""

from __future__ import annotations

import pandas as pd

GROUP_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
]
GEOMETRY_METADATA_PREFIXES: tuple[str, ...] = ("sim_",)
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
    "from_s_db",
    "to_s_db",
    "from_phase_deg",
    "to_phase_deg",
    "signed_phase_step_deg",
    "phase_advance_0to360_deg",
    "phase_error_from_240_deg",
]


def compute_phase_advance(marker_points: pd.DataFrame) -> pd.DataFrame:
    """Compute local phase advance between same-family periodic positions.

    Uses the project convention that a signed -120 degree phase step is a
    240-degree phase advance. Phase advance is computed within periodic
    position families, not by mixing adjacent cell/iris positions: half-integer
    positions are treated as cell-to-cell and integer positions as iris-to-iris.
    """

    table = marker_points.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    if "tune_position" not in table:
        table["tune_position"] = pd.NA
    metadata_columns = _geometry_metadata_columns(table)
    output_columns = _output_columns(metadata_columns)

    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table[table["_tune_sort"].notna()].copy()
    if table.empty:
        return pd.DataFrame(columns=output_columns)
    table["position_family"] = table["_tune_sort"].map(_position_family)

    rows: list[dict[str, object]] = []
    grouping_columns = [*GROUP_COLUMNS, *metadata_columns, "position_family"]
    for group_values, group in table.groupby(grouping_columns, dropna=False, sort=False):
        group = group.sort_values("_tune_sort", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue
        group_metadata = dict(zip(grouping_columns, group_values, strict=True))
        for index in range(len(group) - 1):
            start = group.iloc[index]
            end = group.iloc[index + 1]
            signed_step = _wrap180(float(end["s_phase_deg"]) - float(start["s_phase_deg"]))
            phase_advance = signed_step % 360.0
            rows.append(
                {
                    **group_metadata,
                    "from_source_file": start["source_file"],
                    "to_source_file": end["source_file"],
                    "from_tune_position": start["tune_position"],
                    "to_tune_position": end["tune_position"],
                    "target_freq_ghz": start["target_freq_ghz"],
                    "from_freq_ghz": start["freq_ghz"],
                    "to_freq_ghz": end["freq_ghz"],
                    "from_s_db": start["s_db"],
                    "to_s_db": end["s_db"],
                    "from_phase_deg": start["s_phase_deg"],
                    "to_phase_deg": end["s_phase_deg"],
                    "signed_phase_step_deg": signed_step,
                    "phase_advance_0to360_deg": phase_advance,
                    "phase_error_from_240_deg": phase_advance - 240.0,
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def _geometry_metadata_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if column not in GROUP_COLUMNS
        and column not in TUNE_SWEEP_METADATA_COLUMNS
        and any(column.startswith(prefix) for prefix in GEOMETRY_METADATA_PREFIXES)
    ]


def _output_columns(metadata_columns: list[str]) -> list[str]:
    insert_index = OUTPUT_COLUMNS.index("position_family")
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _position_family(tune_position: float) -> str:
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{fractional:g}"
