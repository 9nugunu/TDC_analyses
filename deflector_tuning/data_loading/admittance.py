"""Compatibility exports for direct CST Y11 admittance analysis."""

from __future__ import annotations

from pathlib import Path

from deflector_tuning.data_loading.one_port_matrix import (
    extract_one_port_marker_frequencies,
    load_y11_touchstone_folder,
    sample_y11_markers,
)


def is_y11_touchstone_folder(path: str | Path) -> bool:
    """Return whether a folder directly contains one-port Y Touchstone files."""

    folder = Path(path)
    if not folder.is_dir():
        return False
    return any(
        candidate.is_file() and candidate.suffix.lower() == ".y1p"
        for candidate in folder.iterdir()
    )


def extract_y11_marker_frequencies(
    path: str | Path,
    *,
    marker_role: str,
) -> pd.DataFrame:
    """Compatibility wrapper for direct one-port marker extraction."""

    return extract_one_port_marker_frequencies(path, marker_role=marker_role)


__all__ = [
    "extract_y11_marker_frequencies",
    "is_y11_touchstone_folder",
    "load_y11_touchstone_folder",
    "sample_y11_markers",
]
