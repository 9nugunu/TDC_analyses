"""Helpers for rejecting non-finite plotting inputs with actionable context."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_ID_COLUMNS: tuple[str, ...] = ("dataset_id", "data_layer", "source_file", "marker_name", "s_name")


def require_finite_plot_columns(
    table: pd.DataFrame,
    *,
    columns: Iterable[str],
    context: str,
    id_columns: Iterable[str] = DEFAULT_ID_COLUMNS,
) -> None:
    """Raise ``ValueError`` when plotting columns contain NaN or infinite values."""

    column_list = [column for column in columns if column in table]
    if table.empty or not column_list:
        return

    numeric = table[column_list].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(numeric.to_numpy(dtype=float))
    if not invalid_mask.any():
        return

    invalid_rows = np.where(invalid_mask.any(axis=1))[0]
    examples: list[str] = []
    id_column_list = [column for column in id_columns if column in table]
    for row_position in invalid_rows[:3]:
        row = table.iloc[row_position]
        bad_columns = [column for column, is_invalid in zip(column_list, invalid_mask[row_position], strict=True) if is_invalid]
        location = ", ".join(f"{column}={row[column]!r}" for column in id_column_list) or f"row_index={table.index[row_position]!r}"
        values = ", ".join(f"{column}={row[column]!r}" for column in bad_columns)
        examples.append(f"{location} [{values}]")

    raise ValueError(
        f"Non-finite plotting values in {context}; first invalid rows: " + "; ".join(examples)
    )
