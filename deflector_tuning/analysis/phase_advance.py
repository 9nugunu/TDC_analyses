"""Compute local phase advance from marker-sampled S-parameter points."""

from __future__ import annotations

import pandas as pd
import math
import warnings

from deflector_tuning.data_contracts import simulation_geometry_columns

GROUP_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
]
OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_name",
    "marker_role",
    "port_side",
    "s_name",
    "position_family",
    "file_from",
    "file_to",
    "pos_from",
    "pos_to",
    "freq_target_ghz",
    "freq_ghz",
    "s_db_from",
    "s_db_to",
    "phase_from_deg",
    "phase_to_deg",
    "phase_step_deg",
    "phase_adv_deg",
    "phase_err_240_deg",
]


def compute_phase_advance(marker_points: pd.DataFrame, *, expected_position_step: float = 1.0) -> pd.DataFrame:
    """Compute local phase advance between same-family periodic positions.

    Uses the project convention that a signed -120 degree phase step is a
    240-degree phase advance. Phase advance is computed within periodic
    position families, not by mixing adjacent cell/iris positions: half-integer
    positions are treated as cell-to-cell and integer positions as iris-to-iris.
    Nonlocal gaps are warned about and excluded; expected_position_step is one
    same-family physical period, never an inferred distance between samples.
    """

    if not math.isfinite(expected_position_step) or expected_position_step <= 0:
        raise ValueError("expected_position_step must be finite and positive")
    table = marker_points.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    if "tune_position" not in table:
        table["tune_position"] = pd.NA
    metadata_columns = simulation_geometry_columns(table)
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
        duplicated = group["_tune_sort"].duplicated(keep=False)
        if duplicated.any():
            positions = sorted(group.loc[duplicated, "_tune_sort"].unique())
            raise ValueError(
                f"Ambiguous duplicate tune positions {positions} within one condition; "
                "select one trace per position or supply distinct condition metadata"
            )
        if len(group) < 2:
            continue
        group_metadata = dict(zip(grouping_columns, group_values, strict=True))
        for index in range(len(group) - 1):
            start = group.iloc[index]
            end = group.iloc[index + 1]
            step = float(end["_tune_sort"]) - float(start["_tune_sort"])
            if not math.isclose(step, expected_position_step, rel_tol=0.0, abs_tol=1e-9):
                warnings.warn(
                    f"Skipping nonlocal phase transition {start['tune_position']} -> "
                    f"{end['tune_position']}; expected position step {expected_position_step:g}",
                    UserWarning, stacklevel=2,
                )
                continue
            signed_step = _wrap180(float(end["s_phase_deg"]) - float(start["s_phase_deg"]))
            phase_advance = signed_step % 360.0
            rows.append(
                {
                    **group_metadata,
                    "file_from": start["source_file"],
                    "file_to": end["source_file"],
                    "pos_from": start["tune_position"],
                    "pos_to": end["tune_position"],
                    "freq_target_ghz": start["freq_target_ghz"],
                    "freq_ghz": start["freq_ghz"],
                    "s_db_from": start["s_db"],
                    "s_db_to": end["s_db"],
                    "phase_from_deg": start["s_phase_deg"],
                    "phase_to_deg": end["s_phase_deg"],
                    "phase_step_deg": signed_step,
                    "phase_adv_deg": phase_advance,
                    "phase_err_240_deg": phase_advance - 240.0,
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


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
