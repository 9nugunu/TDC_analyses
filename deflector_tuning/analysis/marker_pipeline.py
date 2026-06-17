"""One-folder marker analysis pipeline helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.phase_advance import OUTPUT_COLUMNS as PHASE_ADVANCE_COLUMNS
from deflector_tuning.analysis.phase_advance import compute_phase_advance
from deflector_tuning.analysis.phase_summary import OUTPUT_COLUMNS as PHASE_SUMMARY_COLUMNS
from deflector_tuning.analysis.phase_summary import summarize_phase_advance
from deflector_tuning.analysis.nodal_shift import OUTPUT_COLUMNS as NODAL_SHIFT_COLUMNS
from deflector_tuning.analysis.nodal_shift import compute_nodal_shift_errors
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.markers.frequency_markers import extract_marker_frequencies
from deflector_tuning.markers.sampling import sample_nearest_markers

AnalysisTables = OrderedDict[str, pd.DataFrame]

TABLE_FILENAMES: dict[str, str] = {
    "markers": "markers.csv",
    "marker_points": "marker_points.csv",
    "marker_phase_polar": "marker_phase_polar.csv",
    "phase_advance": "phase_advance.csv",
    "phase_summary": "phase_summary.csv",
    "nodal_shift": "nodal_shift.csv",
}

POLAR_MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
POLAR_PHASE_COLUMNS: dict[str, str] = {
    "f_2pi3": "f_2pi3_phase_deg",
    "f_mean": "f_mean_phase_deg",
    "f_pi2": "f_pi2_phase_deg",
}
POLAR_ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "dataset_id",
    "source_file",
    "run_id",
    "tune_position",
    "port_side",
    "s_name",
    "marker_role",
    "scan_type",
    "sim_NumDepth",
    "sim_DepthPlunger_offset",
    "sim_NumTune",
    "sim_R2Taper2",
    "sim_L2Taper2",
    "sim_r_c",
    "sim_w_c",
)


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
    marker_phase_polar = build_marker_phase_polar_table(marker_points)
    phase_advance = _compute_phase_advance_when_supported(marker_points)
    phase_summary = summarize_phase_advance(phase_advance) if not phase_advance.empty else _empty_phase_summary()
    nodal_shift = compute_nodal_shift_errors(phase_advance) if not phase_advance.empty else _empty_nodal_shift()
    return OrderedDict(
        [
            ("markers", markers),
            ("marker_points", marker_points),
            ("marker_phase_polar", marker_phase_polar),
            ("phase_advance", phase_advance),
            ("phase_summary", phase_summary),
            ("nodal_shift", nodal_shift),
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


def build_marker_phase_polar_table(marker_points: pd.DataFrame) -> pd.DataFrame:
    """Return one row per polar phase run with marker phases as columns."""

    required_columns = {"marker_name", "s_phase_deg"}
    if marker_points.empty or not required_columns.issubset(marker_points.columns):
        return pd.DataFrame(columns=_polar_phase_output_columns(marker_points))

    table = marker_points[marker_points["marker_name"].isin(POLAR_MARKER_ORDER)].copy()
    if table.empty:
        return pd.DataFrame(columns=_polar_phase_output_columns(marker_points))

    id_columns = _polar_phase_id_columns(table)
    if not id_columns:
        id_columns = ["source_file"] if "source_file" in table else []
    wide = (
        table.groupby([*id_columns, "marker_name"], dropna=False)["s_phase_deg"]
        .first()
        .unstack("marker_name")
        .rename(columns=POLAR_PHASE_COLUMNS)
        .reset_index()
    )
    phase_columns = [POLAR_PHASE_COLUMNS[marker] for marker in POLAR_MARKER_ORDER if POLAR_PHASE_COLUMNS[marker] in wide]
    output = wide[[*id_columns, *phase_columns]].copy()
    return _sort_polar_phase_table(output)


def _compute_phase_advance_when_supported(marker_points: pd.DataFrame) -> pd.DataFrame:
    if not _has_phase_advance_axis(marker_points):
        return pd.DataFrame(columns=PHASE_ADVANCE_COLUMNS)
    return compute_phase_advance(marker_points)


def _has_phase_advance_axis(marker_points: pd.DataFrame) -> bool:
    if "tune_position" not in marker_points:
        return False
    return marker_points["tune_position"].dropna().nunique() > 1


def _empty_phase_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=PHASE_SUMMARY_COLUMNS)


def _empty_nodal_shift() -> pd.DataFrame:
    return pd.DataFrame(columns=NODAL_SHIFT_COLUMNS)


def _polar_phase_id_columns(table: pd.DataFrame) -> list[str]:
    id_columns = [column for column in POLAR_ID_COLUMN_CANDIDATES if column in table]
    if not id_columns:
        return []
    complete_columns: list[str] = []
    for column in id_columns:
        if table[column].notna().any():
            complete_columns.append(column)
    return complete_columns


def _polar_phase_output_columns(marker_points: pd.DataFrame) -> list[str]:
    id_columns = _polar_phase_id_columns(marker_points) if not marker_points.empty else []
    return [*id_columns, *(POLAR_PHASE_COLUMNS[marker] for marker in POLAR_MARKER_ORDER)]


def _sort_polar_phase_table(table: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in (
            "dataset_id",
            "run_id",
            "tune_position",
            "sim_NumDepth",
            "sim_r_c",
            "sim_w_c",
            "source_file",
            "s_name",
            "port_side",
        )
        if column in table
    ]
    if not sort_columns:
        return table.reset_index(drop=True)
    sortable = table.copy()
    helper_columns: list[str] = []
    for column in sort_columns:
        helper = f"_{column}_sort"
        sortable[helper] = pd.to_numeric(sortable[column], errors="coerce")
        helper_columns.append(helper)
    sorted_table = sortable.sort_values([*helper_columns, *sort_columns], kind="mergesort")
    return sorted_table.drop(columns=helper_columns).reset_index(drop=True)


def _presentation_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if "data_kind" in output:
        output = output.drop(columns=["data_kind"])
    if "port_side" in output and output["port_side"].isna().all():
        output = output.drop(columns=["port_side"])
    return output
