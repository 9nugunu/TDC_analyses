"""Sample S-parameter tables at marker frequencies."""

from __future__ import annotations

import pandas as pd

from deflector_tuning.table_schema import SOURCE_COLUMN_ALIASES

GROUP_COLUMNS: list[str] = ["source_file", "s_name", "tune_position", "port_side"]
OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "tune_position",
    "port_side",
    "s_name",
    "marker_name",
    "marker_role",
    "marker_source",
    "freq_target_ghz",
    "freq_ghz",
    "freq_error_ghz",
    "s_db",
    "s_phase_deg",
    "source_format",
    "uncorrected_freq_ghz",
    "frequency_scale_factor",
    "temp_op_C",
    "temp_meas_C",
    "humidity_fraction",
]
PASSTHROUGH_PREFIXES: tuple[str, ...] = ("sim_",)
PASSTHROUGH_COLUMNS: tuple[str, ...] = ("run_id", "scan_type", "reference_ohm", "is_normalized")
MARKER_METADATA_COLUMNS: list[str] = [
    "uncorrected_freq_ghz",
    "frequency_scale_factor",
    "temp_op_C",
    "temp_meas_C",
    "humidity_fraction",
]


def sample_nearest_markers(sparameter_table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Return long-form nearest S-parameter samples for each marker frequency."""

    source_table = _canonicalize_source_columns(sparameter_table)
    frames = [_sample_one_marker(source_table, marker) for _, marker in markers.iterrows()]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    sampled = pd.concat(frames, ignore_index=True)
    passthrough_columns = _passthrough_columns(source_table)
    return sampled[[*OUTPUT_COLUMNS, *passthrough_columns]]


def _sample_one_marker(sparameter_table: pd.DataFrame, marker: pd.Series) -> pd.DataFrame:
    target_freq_ghz = float(marker["freq_ghz"])
    table = sparameter_table.copy()
    table["freq_target_ghz"] = target_freq_ghz
    table["freq_error_ghz"] = table["freq_ghz"] - target_freq_ghz
    table["_abs_freq_error_ghz"] = table["freq_error_ghz"].abs()

    for column in GROUP_COLUMNS:
        if column not in table:
            table[column] = pd.NA

    nearest_index = table.groupby(GROUP_COLUMNS, dropna=False)["_abs_freq_error_ghz"].idxmin()
    nearest = table.loc[nearest_index].copy()
    nearest["marker_name"] = marker["marker_name"]
    nearest["marker_role"] = marker["marker_role"]
    nearest["marker_source"] = marker["marker_source"]
    for column in MARKER_METADATA_COLUMNS:
        nearest[column] = marker[column] if column in marker else pd.NA
    return nearest


def _passthrough_columns(sparameter_table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in sparameter_table.columns:
        if column in OUTPUT_COLUMNS or column in columns:
            continue
        if column in PASSTHROUGH_COLUMNS or any(column.startswith(prefix) for prefix in PASSTHROUGH_PREFIXES):
            columns.append(column)
    return columns


def _canonicalize_source_columns(table: pd.DataFrame) -> pd.DataFrame:
    collisions = [
        (legacy, canonical)
        for legacy, canonical in SOURCE_COLUMN_ALIASES.items()
        if legacy in table.columns and canonical in table.columns
    ]
    if collisions:
        details = ", ".join(f"{legacy}/{canonical}" for legacy, canonical in collisions)
        raise ValueError(f"source metadata alias collision: {details}")
    renames = {
        legacy: canonical
        for legacy, canonical in SOURCE_COLUMN_ALIASES.items()
        if legacy in table.columns
    }
    return table.rename(columns=renames).copy()
