"""Position gates for entrance coupler-cavity analyses."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

ALLOWED_COUPLER_CAVITY_TRANSITIONS: tuple[tuple[float, float, str], ...] = (
    (1.0, 2.0, "iris_center"),
    (0.5, 1.5, "cell_center"),
)


def is_allowed_coupler_cavity_transition(from_tune_position: object, to_tune_position: object) -> bool:
    """Return whether a transition is an ordered entrance coupler-cavity transition."""

    try:
        start = float(from_tune_position)
        end = float(to_tune_position)
    except (TypeError, ValueError):
        return False
    return any(
        abs(start - allowed_start) < 1e-9 and abs(end - allowed_end) < 1e-9
        for allowed_start, allowed_end, _basis in ALLOWED_COUPLER_CAVITY_TRANSITIONS
    )


def coupler_cavity_endpoint_metadata(tune_positions: Iterable[object]) -> dict[float, dict[str, object]]:
    """Return metadata for positions that form a complete allowed coupler pair."""

    numeric_positions = _numeric_positions(tune_positions)
    metadata: dict[float, dict[str, object]] = {}
    for start, end, basis in ALLOWED_COUPLER_CAVITY_TRANSITIONS:
        present_start = _matching_position(numeric_positions, start)
        present_end = _matching_position(numeric_positions, end)
        if present_start is None or present_end is None:
            continue
        pair_label = f"{start:g}_to_{end:g}"
        metadata[present_start] = {
            "cpl_pair": pair_label,
            "cpl_pos_basis": basis,
            "cpl_pos_from": start,
            "cpl_pos_to": end,
        }
        metadata[present_end] = {
            "cpl_pair": pair_label,
            "cpl_pos_basis": basis,
            "cpl_pos_from": start,
            "cpl_pos_to": end,
        }
    return metadata


def _numeric_positions(values: Iterable[object]) -> list[float]:
    positions: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        try:
            position = float(value)
        except (TypeError, ValueError):
            continue
        if not any(abs(position - existing) < 1e-9 for existing in positions):
            positions.append(position)
    return positions


def _matching_position(positions: list[float], target: float) -> float | None:
    for position in positions:
        if abs(position - target) < 1e-9:
            return position
    return None
