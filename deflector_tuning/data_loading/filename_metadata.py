"""Filename metadata helpers for experiment/simulation data files."""

from __future__ import annotations

import re
from pathlib import Path


def metadata_from_filename(file_path: str | Path, dataset_id: str) -> dict[str, object]:
    """Extract lightweight metadata from known tuning filenames.

    ``in``/``out`` side labels are only meaningful before full brazing. Once a
    dataset name contains ``fullbrazing``, keep the numeric tune position but
    ignore the side label.
    """

    stem = Path(file_path).stem.replace("_processed", "")
    side = _side_from_stem(stem)
    if "fullbrazing" in dataset_id.lower():
        side = None
    return {
        "tune_position": _first_number(stem),
        "port_side": side,
    }


def _first_number(text: str) -> float | None:
    cell_match = re.search(r"(?:^|[_-])cell[_-]?(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if cell_match is not None:
        return float(cell_match.group(1))
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def _side_from_stem(stem: str) -> str | None:
    lower = stem.lower()
    if lower.startswith("in_"):
        return "in"
    if lower.startswith("out_"):
        return "out"
    return None
