from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_GRID_RC_LINE_SCAN_W_C: Final[float] = 19.3224
GRID_RC_LINE_SCAN_TOLERANCE: Final[float] = 1e-9
GRID_RC_LINE_SCAN_COORDINATES: Final[tuple[str, str]] = ("sim_r_c", "sim_w_c")
RC_LINE_SCAN_TIE_COLUMNS: Final[tuple[str, ...]] = ("sim_NumDepth", "run_id", "source_file", "marker_name")


def extract_fixed_width_rc_line_scan(
    table: pd.DataFrame,
    *,
    fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C,
    tolerance: float = GRID_RC_LINE_SCAN_TOLERANCE,
) -> pd.DataFrame:
    if table.empty:
        return table.copy().reset_index(drop=True)
    missing = [column for column in GRID_RC_LINE_SCAN_COORDINATES if column not in table]
    if missing:
        return table.iloc[0:0].copy().reset_index(drop=True)

    widths = pd.to_numeric(table["sim_w_c"], errors="coerce")
    radii = pd.to_numeric(table["sim_r_c"], errors="coerce")
    mask = widths.sub(float(fixed_w_c)).abs().le(float(tolerance)) & radii.notna()
    line_scan = table.loc[mask].copy()
    if line_scan.empty:
        return line_scan.reset_index(drop=True)
    return _sort_rc_line_scan(line_scan)


def write_fixed_width_rc_line_scan_csv(
    table: pd.DataFrame,
    output_path: Path,
    *,
    fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C,
) -> Path | None:
    line_scan = extract_fixed_width_rc_line_scan(table, fixed_w_c=fixed_w_c)
    if line_scan.empty:
        return None
    line_scan.to_csv(output_path, index=False)
    return output_path


def rc_line_scan_filename(stem: str, fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C) -> str:
    return f"{stem}_r_c_line_scan_w_c_{_number_token(fixed_w_c)}.csv"


def _sort_rc_line_scan(table: pd.DataFrame) -> pd.DataFrame:
    sortable = table.copy()
    sortable["_sim_r_c_sort"] = pd.to_numeric(sortable["sim_r_c"], errors="coerce")
    sort_columns = [
        "_sim_r_c_sort",
        *(column for column in RC_LINE_SCAN_TIE_COLUMNS if column in sortable),
    ]
    sorted_table = sortable.sort_values(sort_columns, kind="mergesort")
    return sorted_table.drop(columns=["_sim_r_c_sort"]).reset_index(drop=True)


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")
