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
    "position_family",
]
OUTPUT_COLUMNS: list[str] = [
    *GROUP_COLUMNS,
    "n_steps",
    "phase_adv_mean_deg",
    "phase_err_mean_deg",
    "phase_err_abs_mean_deg",
    "phase_err_rms_deg",
    "phase_err_abs_max_deg",
    "worst_pos_from",
    "worst_pos_to",
    "worst_phase_err_deg",
]


def summarize_phase_advance(phase_table: pd.DataFrame) -> pd.DataFrame:
    """Return per-marker/port summary metrics for phase-advance errors."""

    table = phase_table.copy()
    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA

    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        errors = group["phase_err_240_deg"].astype(float)
        advances = group["phase_adv_deg"].astype(float)
        worst_index = errors.abs().idxmax()
        worst = group.loc[worst_index]
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, group_values, strict=True)),
                "n_steps": int(len(group)),
                "phase_adv_mean_deg": float(advances.mean()),
                "phase_err_mean_deg": float(errors.mean()),
                "phase_err_abs_mean_deg": float(errors.abs().mean()),
                "phase_err_rms_deg": float((errors.pow(2).mean()) ** 0.5),
                "phase_err_abs_max_deg": float(errors.abs().max()),
                "worst_pos_from": worst["pos_from"],
                "worst_pos_to": worst["pos_to"],
                "worst_phase_err_deg": worst["phase_err_240_deg"],
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
