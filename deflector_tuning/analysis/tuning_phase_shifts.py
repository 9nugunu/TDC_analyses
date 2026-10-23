"""Pure marker selection and wrapped campaign phase differences."""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

MARKER_ORDER = ("f_2pi3", "f_mean", "f_pi2")

def build_phase_shift_table(
    state_points: Mapping[str, pd.DataFrame],
    *,
    state_order: Sequence[str],
    positions: Mapping[str, float],
) -> pd.DataFrame:
    """Return wrapped S11 phase shifts relative to each position's first state."""

    rows: list[dict[str, object]] = []
    for position_label, position in positions.items():
        for marker_name in MARKER_ORDER:
            observed: list[tuple[str, pd.Series]] = []
            for state_id in state_order:
                table = state_points.get(state_id)
                if table is None:
                    continue
                marker_row = _select_marker_row(table, position=position, marker_name=marker_name)
                if marker_row is not None:
                    observed.append((state_id, marker_row))
            if not observed:
                continue
            baseline_state, baseline = observed[0]
            baseline_phase = float(baseline["s_phase_deg"])
            for state_id, row in observed:
                phase = float(row["s_phase_deg"])
                rows.append(
                    {
                        "state_id": state_id,
                        "position_label": position_label,
                        "position": position,
                        "source_file": str(row["source_file"]),
                        "marker_name": marker_name,
                        "s_phase_deg": phase,
                        "baseline_state": baseline_state,
                        "baseline_phase_deg": baseline_phase,
                        "delta_phase_deg": _wrap_phase_deg(phase - baseline_phase),
                    }
                )
    return pd.DataFrame(rows)


def _select_marker_row(table: pd.DataFrame, *, position: float, marker_name: str) -> pd.Series | None:
    required = {"source_file", "tune_position", "marker_name", "s_phase_deg"}
    if not required.issubset(table.columns):
        raise ValueError(f"marker_pts is missing required columns: {sorted(required.difference(table.columns))}")
    rows = table.loc[
        ((pd.to_numeric(table["tune_position"], errors="coerce") - position).abs() < 1e-9)
        & table["marker_name"].astype(str).eq(marker_name)
    ].copy()
    rows["s_phase_deg"] = pd.to_numeric(rows["s_phase_deg"], errors="coerce")
    rows = rows[np.isfinite(rows["s_phase_deg"])].copy()
    if rows.empty:
        return None
    port_extended = rows.loc[
        rows["source_file"].astype(str).str.contains("portE", case=False, na=False)
        & ~rows["source_file"].astype(str).str.contains("noportE", case=False, na=False)
    ]
    if len(port_extended) == 1:
        return port_extended.iloc[0]
    if len(rows) == 1:
        return rows.iloc[0]
    raise ValueError(f"Ambiguous marker samples at position={position:g}, marker={marker_name}")


def _wrap_phase_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _position_label(family: str, position: float) -> str:
    return f"{family.title()} {float(position)}"
