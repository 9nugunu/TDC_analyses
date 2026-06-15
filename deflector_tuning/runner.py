"""High-level one-folder analysis runner."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from deflector_tuning.analysis.grid_scan_spacing import summarize_marker_spacing_for_grid_scan
from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, save_marker_analysis
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.visualization.grid_scan_spacing_maps import plot_grid_scan_spacing_error_maps
from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers

AnalysisPaths = OrderedDict[str, Path]
FigurePaths = OrderedDict[str, OrderedDict[str, Path]]

BASE_ANALYSIS_MODES: tuple[str, ...] = ("marker_analysis", "s11", "phase_advance", "polar")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = Path("sim") / "260505_single_cell_dispersion_step1"
GRID_SCAN_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"data_kind", "sim_r_c", "sim_w_c", "marker_name", "s_phase_deg"}
)
GRID_SCAN_REQUIRED_MARKERS: frozenset[str] = frozenset({"f_2pi3", "f_mean", "f_pi2"})


@dataclass(frozen=True)
class RunResult:
    """Paths and analysis modes produced by :func:`run_folder_analysis`."""

    output_dir: Path
    tables: AnalysisPaths
    figures: FigurePaths
    analysis_modes: tuple[str, ...]
    manifest_path: Path


def run_folder_analysis(
    *,
    sparameter_path: str | Path,
    dispersion_path: str | Path | None = None,
    output_dir: str | Path,
    marker_role: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    loader: DataLoader | None = None,
) -> RunResult:
    """Run the standard one-folder marker workflow and write tables/figures.

    The runner intentionally only orchestrates existing loader, analysis, and
    visualization modules. Optional analyses, such as simulation 2D grid-scan
    spacing maps, are enabled from conservative checks on the analysis tables.
    """

    sparameter_path, dispersion_path = resolve_input_paths(
        sparameter_path,
        dispersion_path=dispersion_path,
        data_root=data_root,
    )
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = loader or DataLoader()
    sparameter_table = loader.load(sparameter_path)
    tables = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        loader=loader,
    )
    table_paths = AnalysisPaths(save_marker_analysis(tables, table_dir))
    modes = detect_analysis_modes(tables)

    figures: FigurePaths = OrderedDict()
    figures["s11"] = OrderedDict(
        plot_s11_with_markers(sparameter_table, tables["marker_points"], figure_root / "s11")
    )
    if _has_rows(tables.get("phase_advance")):
        figures["phase_advance"] = OrderedDict(
            plot_phase_advance(tables["phase_advance"], figure_root / "phase_advance", split_by_family=True)
        )
    if _has_rows(tables.get("marker_points")):
        figures["polar"] = OrderedDict(
            plot_marker_phase_polar_views(tables["marker_points"], figure_root / "polar")
        )

    detection = _detection_report(tables)
    if "grid_scan_spacing" in modes:
        spacing_summary = summarize_marker_spacing_for_grid_scan(tables["marker_points"])
        figures["grid_scan_spacing"] = OrderedDict(
            plot_grid_scan_spacing_error_maps(spacing_summary, figure_root / "grid_scan_spacing")
        )

    manifest_path = output_dir / "manifest.json"
    _write_manifest(
        manifest_path,
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        modes=modes,
        detection=detection,
        tables=table_paths,
        figures=figures,
    )

    return RunResult(
        output_dir=output_dir,
        tables=table_paths,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )


def resolve_input_paths(
    sparameter_path: str | Path,
    *,
    dispersion_path: str | Path | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> tuple[Path, Path]:
    """Resolve data-relative inputs and default dispersion folder."""

    data_root = Path(data_root)
    resolved_sparameter_path = _data_relative_path(sparameter_path, data_root=data_root)
    if dispersion_path is None:
        resolved_dispersion_path = data_root / DEFAULT_DISPERSION_SUBPATH
    else:
        resolved_dispersion_path = _data_relative_path(dispersion_path, data_root=data_root)
    return resolved_sparameter_path, resolved_dispersion_path


def _data_relative_path(path: str | Path, *, data_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.parts[:1] == (data_root.name,):
        return path
    return data_root / path


def detect_analysis_modes(tables: dict[str, pd.DataFrame]) -> tuple[str, ...]:
    """Return standard and auto-detected analysis modes for a table bundle."""

    modes = list(BASE_ANALYSIS_MODES)
    report = _detection_report(tables)
    if report["grid_scan_spacing"]["enabled"]:
        modes.append("grid_scan_spacing")
    return tuple(modes)


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and not table.empty


def _detection_report(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    marker_points = tables.get("marker_points")
    enabled, reason = _is_simulation_grid_scan(marker_points)
    return {"grid_scan_spacing": {"enabled": enabled, "reason": reason}}


def _is_simulation_grid_scan(marker_points: pd.DataFrame | None) -> tuple[bool, str]:
    if marker_points is None or marker_points.empty:
        return False, "marker_points table is missing or empty"

    missing = sorted(GRID_SCAN_REQUIRED_COLUMNS.difference(marker_points.columns))
    if missing:
        return False, f"marker_points is missing required columns: {missing}"

    data_kinds = set(marker_points["data_kind"].astype(str).str.lower().dropna().unique())
    if not data_kinds or not data_kinds.issubset({"sim", "simulation"}):
        return False, f"data_kind is not simulation-only: {sorted(data_kinds)}"

    markers = set(marker_points["marker_name"].dropna().astype(str).unique())
    missing_markers = sorted(GRID_SCAN_REQUIRED_MARKERS.difference(markers))
    if missing_markers:
        return False, f"marker_points is missing complete grid-scan markers: {missing_markers}"

    grid_points = marker_points[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
    if len(grid_points) < 2:
        return False, "fewer than two unique sim_r_c/sim_w_c grid points"

    axis_counts = grid_points.nunique(dropna=True)
    if int(axis_counts.get("sim_r_c", 0)) < 2 and int(axis_counts.get("sim_w_c", 0)) < 2:
        return False, "sim_r_c and sim_w_c do not vary across grid points"

    return True, "simulation marker_points include sim_r_c/sim_w_c and complete f_2pi3/f_mean/f_pi2 groups"


def _write_manifest(
    path: Path,
    *,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    modes: tuple[str, ...],
    detection: dict[str, dict[str, object]],
    tables: AnalysisPaths,
    figures: FigurePaths,
) -> None:
    manifest: dict[str, Any] = {
        "sparameter_path": str(sparameter_path),
        "dispersion_path": str(dispersion_path),
        "marker_role": marker_role,
        "analysis_modes": list(modes),
        "detected_modes": detection,
        "outputs": {
            "tables": {name: str(table_path) for name, table_path in tables.items()},
            "figures": {
                group: {name: str(figure_path) for name, figure_path in paths.items()} for group, paths in figures.items()
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
