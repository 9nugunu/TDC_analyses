from __future__ import annotations

import pandas as pd

from deflector_tuning.workflows.models import DetectionReport

BASE_ANALYSIS_MODES: tuple[str, ...] = (
    "marker_analysis",
    "s11",
    "phase_advance",
    "nodal_shift",
    "polar",
)
GRID_SCAN_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"data_kind", "sim_r_c", "sim_w_c", "marker_name", "s_phase_deg"}
)
GRID_SCAN_REQUIRED_MARKERS: frozenset[str] = frozenset({"f_2pi3", "f_mean", "f_pi2"})


def detect_analysis_modes(
    tables: dict[str, pd.DataFrame],
    *,
    dataset_category: str | None = None,
) -> tuple[str, ...]:
    modes = list(BASE_ANALYSIS_MODES)
    report = detection_report(tables, dataset_category=dataset_category)
    for mode in (
        "coupler_cavity_parameters",
        "cell_iris_response",
        "geometry_phase_response",
        "grid_scan_spacing",
    ):
        if report[mode]["enabled"]:
            modes.append(mode)
    return tuple(modes)


def detection_report(
    tables: dict[str, pd.DataFrame],
    *,
    dataset_category: str | None = None,
) -> DetectionReport:
    marker_points = tables.get("marker_pts")
    grid_enabled, grid_reason = _is_simulation_grid_scan(
        marker_points,
        dataset_category=dataset_category,
    )
    coupler_enabled, coupler_reason = _has_rows(
        tables.get("coupler_params"),
        "coupler_cavity_parameter_estimates",
        "estimate",
    )
    cell_enabled, cell_reason = _has_rows(
        tables.get("cell_iris_cmp"),
        "cell_iris_response_comparison",
        "matched transition",
    )
    response_enabled, response_reason = _has_geometry_phase_response(
        tables.get("geom_phase")
    )
    return {
        "coupler_cavity_parameters": {
            "enabled": coupler_enabled,
            "reason": coupler_reason,
        },
        "cell_iris_response": {"enabled": cell_enabled, "reason": cell_reason},
        "geometry_phase_response": {
            "enabled": response_enabled,
            "reason": response_reason,
        },
        "grid_scan_spacing": {"enabled": grid_enabled, "reason": grid_reason},
    }


def _has_rows(
    table: pd.DataFrame | None,
    table_name: str,
    row_label: str,
) -> tuple[bool, str]:
    if table is None or table.empty:
        return False, f"{table_name} table is missing or empty"
    return True, f"{table_name} has {len(table)} {row_label} rows"


def _has_geometry_phase_response(table: pd.DataFrame | None) -> tuple[bool, str]:
    if table is None or table.empty:
        return False, "geometry_phase_response table is missing or empty"
    axes = (
        sorted(table["sweep_axis"].dropna().astype(str).unique())
        if "sweep_axis" in table
        else []
    )
    if not axes:
        return False, "geometry_phase_response has no sweep_axis values"
    return True, f"geometry phase response available for sweep axis: {', '.join(axes)}"


def _is_simulation_grid_scan(
    marker_points: pd.DataFrame | None,
    *,
    dataset_category: str | None = None,
) -> tuple[bool, str]:
    if dataset_category is not None and dataset_category != "grid":
        return False, f"dataset category is {dataset_category!r}, not 'grid'"
    if marker_points is None or marker_points.empty:
        return False, "marker_points table is missing or empty"

    missing = sorted(GRID_SCAN_REQUIRED_COLUMNS.difference(marker_points.columns))
    if missing:
        return False, f"marker_points is missing required columns: {missing}"
    data_kinds = set(
        marker_points["data_kind"].astype(str).str.lower().dropna().unique()
    )
    if not data_kinds or not data_kinds.issubset({"sim", "simulation"}):
        return False, f"data_kind is not simulation-only: {sorted(data_kinds)}"

    markers = set(marker_points["marker_name"].dropna().astype(str).unique())
    missing_markers = sorted(GRID_SCAN_REQUIRED_MARKERS.difference(markers))
    if missing_markers:
        return (
            False,
            f"marker_points is missing complete grid-scan markers: {missing_markers}",
        )

    grid_points = marker_points[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
    if len(grid_points) < 2:
        return False, "fewer than two unique sim_r_c/sim_w_c grid points"
    axis_counts = grid_points.nunique(dropna=True)
    if (
        int(axis_counts.get("sim_r_c", 0)) < 2
        and int(axis_counts.get("sim_w_c", 0)) < 2
    ):
        return False, "sim_r_c and sim_w_c do not vary across grid points"
    return (
        True,
        "simulation marker_points include sim_r_c/sim_w_c and complete f_2pi3/f_mean/f_pi2 groups",
    )
