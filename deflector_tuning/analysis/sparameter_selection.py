"""Shared S-parameter table selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_s11_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return only S11 rows when an S-parameter name column is available."""

    if "s_name" not in table:
        return table.copy()
    return table[table["s_name"].astype(str).str.upper() == "S11"].copy()


def select_phase_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Select measured finite phases, preserving magnitude-only rows elsewhere."""
    if "s_phase_deg" not in table:
        return table.iloc[:0].copy()
    phases = pd.to_numeric(table["s_phase_deg"], errors="coerce")
    return table.loc[np.isfinite(phases.to_numpy(dtype=float, na_value=np.nan))].copy()
