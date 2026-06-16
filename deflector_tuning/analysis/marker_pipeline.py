"""One-folder marker analysis pipeline helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.phase_advance import compute_phase_advance
from deflector_tuning.analysis.phase_summary import summarize_phase_advance
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.markers.frequency_markers import extract_marker_frequencies
from deflector_tuning.markers.sampling import sample_nearest_markers

AnalysisTables = OrderedDict[str, pd.DataFrame]

TABLE_FILENAMES: dict[str, str] = {
    "markers": "markers.csv",
    "marker_points": "marker_points.csv",
    "phase_advance": "phase_advance.csv",
    "phase_summary": "phase_summary.csv",
}


def build_marker_analysis(
    *,
    sparameter_path: str | Path,
    dispersion_path: str | Path,
    marker_role: str,
    loader: DataLoader | None = None,
) -> AnalysisTables:
    """Build marker-frequency, marker-point, phase-advance, and summary tables.

    Processes exactly the one S-parameter folder passed as ``sparameter_path``;
    batch traversal belongs in a separate wrapper.
    """

    loader = loader or DataLoader()
    sparameter_table = select_s11_rows(loader.load(sparameter_path))
    markers = extract_marker_frequencies(dispersion_path, marker_role=marker_role)
    marker_points = sample_nearest_markers(sparameter_table, markers)
    phase_advance = compute_phase_advance(marker_points)
    phase_summary = summarize_phase_advance(phase_advance)
    return OrderedDict(
        [
            ("markers", markers),
            ("marker_points", marker_points),
            ("phase_advance", phase_advance),
            ("phase_summary", phase_summary),
        ]
    )


def save_marker_analysis(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> OrderedDict[str, Path]:
    """Write marker analysis tables as CSV files and return their paths."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: OrderedDict[str, Path] = OrderedDict()
    for name, filename in TABLE_FILENAMES.items():
        path = folder / filename
        _presentation_table(tables[name]).to_csv(path, index=False)
        paths[name] = path
    return paths


def _presentation_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if "data_kind" in output:
        output = output.drop(columns=["data_kind"])
    if "port_side" in output and output["port_side"].isna().all():
        output = output.drop(columns=["port_side"])
    return output
