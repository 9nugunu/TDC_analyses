"""Shared table identity rules for analysis and visualization."""

import pandas as pd


SERIES_IDENTITY_COLUMNS = ("dataset_id", "data_kind", "data_layer", "marker_role", "port_side", "s_name")
TUNE_STATE_COLUMNS = frozenset({"sim_NumTune", "sim_NumDepth", "sim_DepthPlunger"})


def simulation_geometry_columns(table: pd.DataFrame) -> list[str]:
    """Preserve arbitrary Navigator geometry, excluding the tuning state axis."""
    return [c for c in table if c.startswith("sim_") and c not in TUNE_STATE_COLUMNS]


def varying_condition_columns(table: pd.DataFrame) -> list[str]:
    """Identify independent conditions without treating per-position files as series."""
    candidates = [*SERIES_IDENTITY_COLUMNS, *simulation_geometry_columns(table)]
    return [column for column in candidates if column in table and table[column].nunique(dropna=False) > 1]
