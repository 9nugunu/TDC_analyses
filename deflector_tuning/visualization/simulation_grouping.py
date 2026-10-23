from __future__ import annotations

from typing import SupportsFloat, SupportsIndex

import pandas as pd

from deflector_tuning.data_contracts import simulation_geometry_columns

SIMULATION_PARAMETER_DECIMALS: dict[str, int] = {
    "sim_r_c": 2,
    "sim_w_c": 4,
}


def grid_point_depth_group_columns(table: pd.DataFrame) -> list[str]:
    if "sim_NumDepth" in table and pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().nunique() > 1:
        return ["sim_NumDepth"]
    return []


def varying_sim_sweep_columns(table: pd.DataFrame) -> list[str]:
    return [
        column for column in simulation_geometry_columns(table)
        if table[column].nunique(dropna=False) > 1
    ]



def format_grid_value(value: object) -> str:
    if not isinstance(value, (str, SupportsFloat, SupportsIndex)):
        return str(value)
    try:
        numeric = float(value)
    except ValueError:
        return str(value)
    return f"{numeric:g}"


def format_simulation_parameter_value(column: str, value: object) -> str:
    """Format a Navigator parameter value with stable filename precision."""

    decimals = SIMULATION_PARAMETER_DECIMALS.get(column)
    if decimals is None:
        return format_grid_value(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return format_grid_value(value)
    return f"{numeric:.{decimals}f}"


def format_simulation_parameter_token(column: str, value: object) -> str:
    """Return an ASCII-safe, fixed-width token for a Navigator value."""

    return format_simulation_parameter_value(column, value).replace("-", "m").replace(".", "p")
