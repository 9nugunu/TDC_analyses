"""Nodal-shift target errors derived from local phase-advance rows."""

from __future__ import annotations

import pandas as pd

TARGET_PHASE_ADVANCE_DEG: dict[str, float] = {
    "f_2pi3": 240.0,
    "f_pi2": 180.0,
}
MARKER_ORDER: tuple[str, ...] = tuple(TARGET_PHASE_ADVANCE_DEG)
REQUIRED_COLUMNS: tuple[str, ...] = ("marker_name", "phase_adv_deg")
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
    "phase_adv_deg",
    "phase_target_deg",
    "phase_err_deg",
    "phase_err_abs_deg",
]


def compute_nodal_shift_errors(phase_advance: pd.DataFrame) -> pd.DataFrame:
    """Return marker-specific nodal-shift phase errors.

    ``f_2pi3`` is evaluated against 240 degrees, ``f_pi2`` against 180 degrees,
    and ``f_mean`` is intentionally excluded until it has a clear nodal target.
    """

    if phase_advance.empty:
        return pd.DataFrame(columns=_output_columns(phase_advance))
    missing = [column for column in REQUIRED_COLUMNS if column not in phase_advance]
    if missing:
        raise ValueError(f"phase_advance is missing required columns: {missing}")

    table = phase_advance[phase_advance["marker_name"].isin(MARKER_ORDER)].copy()
    if table.empty:
        return pd.DataFrame(columns=_output_columns(phase_advance))
    table["phase_target_deg"] = table["marker_name"].map(TARGET_PHASE_ADVANCE_DEG).astype(float)
    table["phase_err_deg"] = (
        table["phase_adv_deg"].astype(float) - table["phase_target_deg"]
    )
    table["phase_err_abs_deg"] = table["phase_err_deg"].abs()
    columns = _output_columns(table)
    return table.reindex(columns=columns).sort_values(_sort_columns(table), kind="mergesort").reset_index(drop=True)


def _output_columns(table: pd.DataFrame) -> list[str]:
    metadata_columns = [
        column
        for column in table.columns
        if column.startswith("sim_") and column not in OUTPUT_COLUMNS
    ]
    insert_index = OUTPUT_COLUMNS.index("position_family")
    base_columns = [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]
    return [column for column in base_columns if column in table.columns or column in OUTPUT_COLUMNS[-3:]]


def _sort_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("position_family", "pos_from", "pos_to", "marker_name", "sim_r_c", "sim_w_c")
        if column in table
    ]
