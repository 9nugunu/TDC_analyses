from __future__ import annotations

from typing import SupportsFloat, SupportsIndex

import pandas as pd


def grid_point_depth_group_columns(table: pd.DataFrame) -> list[str]:
    if "sim_NumDepth" in table and pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().nunique() > 1:
        return ["sim_NumDepth"]
    return []


def varying_sim_sweep_columns(table: pd.DataFrame) -> list[str]:
    columns = []
    for column in table.columns:
        if not column.startswith("sim_"):
            continue
        metadata_name = column.removeprefix("sim_")
        if metadata_name.lower().startswith("num"):
            continue
        if table[column].dropna().nunique() > 1:
            columns.append(column)
    return columns


def format_grid_value(value: object) -> str:
    if not isinstance(value, (str, SupportsFloat, SupportsIndex)):
        return str(value)
    try:
        numeric = float(value)
    except ValueError:
        return str(value)
    return f"{numeric:g}"
