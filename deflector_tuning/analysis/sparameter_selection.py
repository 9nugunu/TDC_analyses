"""Shared S-parameter table selection helpers."""

from __future__ import annotations

import pandas as pd


def select_s11_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return only S11 rows when an S-parameter name column is available."""

    if "s_name" not in table:
        return table.copy()
    return table[table["s_name"].astype(str).str.upper() == "S11"].copy()
