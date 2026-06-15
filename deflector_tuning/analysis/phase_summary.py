"""Summarize local phase-advance errors."""

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
OUTPUT_COLUMNS: list[str] = [
    *GROUP_COLUMNS,
    "transition_count",
    "mean_phase_advance_deg",
    "mean_phase_error_deg",
    "mean_abs_phase_error_deg",
    "rms_phase_error_deg",
    "max_abs_phase_error_deg",
    "worst_from_tune_position",
    "worst_to_tune_position",
    "worst_phase_error_deg",
]


def summarize_phase_advance(phase_table: pd.DataFrame) -> pd.DataFrame:
    """Return per-marker/port summary metrics for phase-advance errors."""

    table = phase_table.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA

    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        errors = group["phase_error_from_240_deg"].astype(float)
        advances = group["phase_advance_0to360_deg"].astype(float)
        worst_index = errors.abs().idxmax()
        worst = group.loc[worst_index]
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, group_values, strict=True)),
                "transition_count": int(len(group)),
                "mean_phase_advance_deg": float(advances.mean()),
                "mean_phase_error_deg": float(errors.mean()),
                "mean_abs_phase_error_deg": float(errors.abs().mean()),
                "rms_phase_error_deg": float((errors.pow(2).mean()) ** 0.5),
                "max_abs_phase_error_deg": float(errors.abs().max()),
                "worst_from_tune_position": worst["from_tune_position"],
                "worst_to_tune_position": worst["to_tune_position"],
                "worst_phase_error_deg": worst["phase_error_from_240_deg"],
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
