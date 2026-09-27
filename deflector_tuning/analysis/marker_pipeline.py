"""One-folder marker analysis pipeline helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.cell_iris_response import compare_cell_and_iris_responses
from deflector_tuning.analysis.coupler_cavity_parameters import (
    build_coupler_cavity_parameter_table,
)
from deflector_tuning.analysis.kyhl_admittance import (
    POINT_OUTPUT_COLUMNS as KYHL_ADMITTANCE_POINT_COLUMNS,
)
from deflector_tuning.analysis.kyhl_admittance import (
    OUTPUT_COLUMNS as KYHL_ADMITTANCE_TRANSITION_COLUMNS,
)
from deflector_tuning.analysis.kyhl_admittance import (
    F2PI3_AUDIT_COLUMNS as KYHL_F2PI3_AUDIT_COLUMNS,
)
from deflector_tuning.analysis.kyhl_admittance import (
    compute_f2pi3_normalized_admittance_audit,
    compute_kyhl_admittance_points,
    compute_kyhl_admittance_transitions,
)
from deflector_tuning.analysis.phase_advance import (
    OUTPUT_COLUMNS as PHASE_ADVANCE_COLUMNS,
)
from deflector_tuning.analysis.phase_advance import compute_phase_advance
from deflector_tuning.analysis.phase_summary import (
    OUTPUT_COLUMNS as PHASE_SUMMARY_COLUMNS,
)
from deflector_tuning.analysis.phase_summary import summarize_phase_advance
from deflector_tuning.analysis.nodal_shift import OUTPUT_COLUMNS as NODAL_SHIFT_COLUMNS
from deflector_tuning.analysis.nodal_shift import compute_nodal_shift_errors
from deflector_tuning.analysis.geometry_phase_response import (
    compute_geometry_phase_response,
)
from deflector_tuning.analysis.grid_rc_line_scan import extract_fixed_width_rc_line_scan
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.markers.frequency_markers import (
    TemperatureHumidityCorrection,
    extract_marker_frequencies,
)
from deflector_tuning.markers.sampling import sample_nearest_markers
from deflector_tuning.table_schema import STANDARD_TABLE_SPECS
from deflector_tuning.table_export import TableSaveResult, save_standard_tables

AnalysisTables = OrderedDict[str, pd.DataFrame]

TABLE_FILENAMES: dict[str, str] = {
    name: spec.filename for name, spec in STANDARD_TABLE_SPECS.items()
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
    "sim_offset_cell_03",
    "sim_r_c",
    "sim_w_c",
)


def build_marker_analysis(
    *,
    sparameter_path: str | Path,
    dispersion_path: str | Path,
    marker_role: str,
    loader: DataLoader | None = None,
    sparameter_table: pd.DataFrame | None = None,
    marker_correction: TemperatureHumidityCorrection | None = None,
    geometry_sweep_axis: str | None = None,
    geometry_sweep_base: float | None = None,
) -> AnalysisTables:
    """Build marker-frequency, marker-point, phase-advance, and summary tables.

    Processes exactly the one S-parameter folder passed as ``sparameter_path``;
    batch traversal belongs in a separate wrapper.
    """

    if sparameter_table is None:
        loader = loader or DataLoader()
        sparameter_table = loader.load(sparameter_path)
    sparameter_table = select_s11_rows(sparameter_table)
    markers = extract_marker_frequencies(
        dispersion_path,
        marker_role=marker_role,
        correction=marker_correction,
    )
    marker_points = sample_nearest_markers(sparameter_table, markers)
    phase_polar = build_marker_phase_polar_table(marker_points)
    rc_line = extract_fixed_width_rc_line_scan(phase_polar)
    kyhl_admit_audit = _compute_f2pi3_normalized_admittance_audit_when_supported(marker_points)
    kyhl_admit_pts = _compute_kyhl_admittance_points_when_supported(marker_points)
    kyhl_admit_steps = _compute_kyhl_admittance_transitions_when_supported(marker_points)
    cell_iris_cmp = (
        compare_cell_and_iris_responses(kyhl_admit_steps)
        if not kyhl_admit_steps.empty
        else _empty_cell_iris_response_comparison()
    )
    coupler_params = build_coupler_cavity_parameter_table(marker_points, markers)
    phase_adv = _compute_phase_advance_when_supported(marker_points)
    phase_stats = summarize_phase_advance(phase_adv) if not phase_adv.empty else _empty_phase_summary()
    nodal_shift = compute_nodal_shift_errors(phase_adv) if not phase_adv.empty else _empty_nodal_shift()
    geom_phase = compute_geometry_phase_response(
        marker_points,
        sweep_axis=geometry_sweep_axis,
        sweep_base=geometry_sweep_base,
    )
    return OrderedDict(
        [
            ("markers", markers),
            ("marker_pts", marker_points),
            ("phase_polar", phase_polar),
            ("kyhl_admit_audit", kyhl_admit_audit),
            ("kyhl_admit_pts", kyhl_admit_pts),
            ("kyhl_admit_steps", kyhl_admit_steps),
            ("cell_iris_cmp", cell_iris_cmp),
            ("coupler_params", coupler_params),
            ("rc_line", rc_line),
            ("phase_adv", phase_adv),
            ("phase_stats", phase_stats),
            ("nodal_shift", nodal_shift),
            ("geom_phase", geom_phase),
        ]
    )


def save_marker_analysis(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> TableSaveResult:
    """Persist the standard tables through the transactional CSV saver."""

    return save_standard_tables(tables, output_dir)


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
    phase_columns = [
        POLAR_PHASE_COLUMNS[marker] for marker in POLAR_MARKER_ORDER if POLAR_PHASE_COLUMNS[marker] in wide
    ]
    output = wide[[*id_columns, *phase_columns]].copy()
    return _sort_polar_phase_table(output)


def _compute_phase_advance_when_supported(marker_points: pd.DataFrame) -> pd.DataFrame:
    if not _has_phase_advance_axis(marker_points):
        return pd.DataFrame(columns=PHASE_ADVANCE_COLUMNS)
    return compute_phase_advance(marker_points)


def _compute_kyhl_admittance_points_when_supported(
    marker_points: pd.DataFrame,
) -> pd.DataFrame:
    if marker_points.empty or "tune_position" not in marker_points:
        return pd.DataFrame(columns=KYHL_ADMITTANCE_POINT_COLUMNS)
    return compute_kyhl_admittance_points(marker_points)


def _compute_f2pi3_normalized_admittance_audit_when_supported(
    marker_points: pd.DataFrame,
) -> pd.DataFrame:
    if marker_points.empty or "tune_position" not in marker_points:
        return pd.DataFrame(columns=KYHL_F2PI3_AUDIT_COLUMNS)
    return compute_f2pi3_normalized_admittance_audit(marker_points)


def _compute_kyhl_admittance_transitions_when_supported(
    marker_points: pd.DataFrame,
) -> pd.DataFrame:
    if not _has_phase_advance_axis(marker_points):
        return pd.DataFrame(columns=KYHL_ADMITTANCE_TRANSITION_COLUMNS)
    return compute_kyhl_admittance_transitions(marker_points)


def _has_phase_advance_axis(marker_points: pd.DataFrame) -> bool:
    if "tune_position" not in marker_points:
        return False
    return marker_points["tune_position"].dropna().nunique() > 1


def _empty_phase_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=PHASE_SUMMARY_COLUMNS)


def _empty_nodal_shift() -> pd.DataFrame:
    return pd.DataFrame(columns=NODAL_SHIFT_COLUMNS)


def _empty_cell_iris_response_comparison() -> pd.DataFrame:
    return compare_cell_and_iris_responses(pd.DataFrame())


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
    return [
        *id_columns,
        *(POLAR_PHASE_COLUMNS[marker] for marker in POLAR_MARKER_ORDER),
    ]


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
