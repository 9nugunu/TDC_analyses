"""High-level one-folder analysis runner."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from deflector_tuning.analysis.grid_scan_spacing import summarize_marker_spacing_for_grid_scan
from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, save_marker_analysis
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import dataset_identity_from_path
from deflector_tuning.data_loading.source_layer import detect_data_layer
from deflector_tuning.dispersion import load_cst_dispersion_txt, process_cst_dispersion_txt
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.visualization.grid_scan_spacing_maps import plot_grid_scan_spacing_error_maps
from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers

AnalysisPaths = OrderedDict[str, Path]
FigurePaths = OrderedDict[str, OrderedDict[str, Path]]

BASE_ANALYSIS_MODES: tuple[str, ...] = ("marker_analysis", "s11", "phase_advance", "polar")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = Path("sim") / "sim_260505_dispersion_single_cell_step1"
GRID_SCAN_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"data_kind", "sim_r_c", "sim_w_c", "marker_name", "s_phase_deg"}
)
GRID_SCAN_REQUIRED_MARKERS: frozenset[str] = frozenset({"f_2pi3", "f_mean", "f_pi2"})
logger = logging.getLogger(__name__)


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

    logger.info(
        "Starting folder analysis: sparameter_path=%s dispersion_path=%s output_dir=%s marker_role=%s",
        sparameter_path,
        dispersion_path,
        output_dir,
        marker_role,
    )
    sparameter_path, dispersion_path = resolve_input_paths(
        sparameter_path,
        dispersion_path=dispersion_path,
        data_root=data_root,
    )
    dataset_category = _dataset_category(sparameter_path)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    dispersion_inputs = find_cst_dispersion_inputs(sparameter_path) if dataset_category in {None, "dispersion"} else ()
    if dataset_category == "dispersion" and not dispersion_inputs:
        raise ValueError(f"Dataset category is dispersion, but no parseable CST dispersion txt export was found: {sparameter_path}")
    if dispersion_inputs:
        logger.info("Detected CST dispersion data; running dispersion-only analysis")
        result = _run_dispersion_only_analysis(
            dispersion_inputs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )
        logger.info("Dispersion-only analysis completed successfully")
        return result

    loader = loader or DataLoader()
    logger.info("Resolved input paths: sparameter=%s dispersion=%s", sparameter_path, dispersion_path)
    logger.info("Loading S-parameter table from %s", sparameter_path)
    loaded_sparameter_table = loader.load(sparameter_path)
    logger.info("Loaded S-parameter table with %d rows", len(loaded_sparameter_table))
    sparameter_table = select_s11_rows(loaded_sparameter_table)
    if len(sparameter_table) != len(loaded_sparameter_table):
        logger.info("Selected %d S11 rows for marker analysis outputs", len(sparameter_table))
    logger.info("Building marker analysis tables")
    tables = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        loader=loader,
    )
    logger.info("Built analysis tables: %s", ", ".join(tables.keys()))
    logger.info("Saving analysis tables to %s", table_dir)
    table_paths = AnalysisPaths(save_marker_analysis(tables, table_dir))
    logger.info("Saved %d analysis tables", len(table_paths))
    modes = detect_analysis_modes(tables, dataset_category=dataset_category)
    logger.info("Enabled analysis modes: %s", ", ".join(modes))

    figures: FigurePaths = OrderedDict()
    logger.info("Rendering S11 figures")
    figures["s11"] = OrderedDict(
        plot_s11_with_markers(sparameter_table, tables["marker_points"], figure_root / "s11")
    )
    if _has_rows(tables.get("phase_advance")):
        logger.info("Rendering phase advance figures")
        figures["phase_advance"] = OrderedDict(
            plot_phase_advance(tables["phase_advance"], figure_root / "phase_advance", split_by_family=True)
        )
    else:
        logger.info("Skipping phase advance figures because phase_advance is missing or empty")
    if _has_rows(tables.get("marker_points")):
        logger.info("Rendering polar phase figures")
        figures["polar"] = OrderedDict(
            plot_marker_phase_polar_views(tables["marker_points"], figure_root / "polar")
        )
    else:
        logger.info("Skipping polar phase figures because marker_points is missing or empty")

    detection = _detection_report(tables, dataset_category=dataset_category)
    if "grid_scan_spacing" in modes:
        logger.info("Rendering grid-scan spacing figures")
        spacing_summary = summarize_marker_spacing_for_grid_scan(tables["marker_points"])
        figures["grid_scan_spacing"] = OrderedDict(
            plot_grid_scan_spacing_error_maps(spacing_summary, figure_root / "grid_scan_spacing")
        )
    else:
        logger.info("Skipping grid-scan spacing figures: %s", detection["grid_scan_spacing"]["reason"])

    manifest_path = output_dir / "manifest.json"
    logger.info("Writing manifest to %s", manifest_path)
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
    logger.info("Folder analysis completed successfully")

    return RunResult(
        output_dir=output_dir,
        tables=table_paths,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )


def find_cst_dispersion_inputs(path: str | Path) -> tuple[Path, ...]:
    """Return parseable CST dispersion text exports under ``path``.

    Dispersion exports are frequency-vs-phase eigenmode tables. They should not
    enter the S-parameter marker workflow, because S11, phase-advance, and polar
    marker plots do not describe this data shape.
    """

    input_path = Path(path)
    candidates: list[Path]
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("*.txt"), key=lambda item: item.name.lower())
    else:
        return ()

    dispersion_paths: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.lower() != ".txt":
            continue
        try:
            load_cst_dispersion_txt(candidate)
        except (OSError, ValueError):
            continue
        dispersion_paths.append(candidate)
    return tuple(dispersion_paths)


def _run_dispersion_only_analysis(
    dispersion_inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
) -> RunResult:
    table_paths: AnalysisPaths = AnalysisPaths()
    figures: FigurePaths = OrderedDict()
    dispersion_figures: OrderedDict[str, Path] = OrderedDict()
    csv_dir = table_dir / "dispersion"
    figure_dir = figure_root / "dispersion"

    for input_path in dispersion_inputs:
        outputs = process_cst_dispersion_txt(input_path, output_dir=csv_dir)
        stem = input_path.stem
        table_paths[f"{stem}_long"] = outputs.long_csv
        table_paths[f"{stem}_wide"] = outputs.wide_csv
        table_paths[f"{stem}_summary"] = outputs.summary_csv

        dispersion_table = load_cst_dispersion_txt(input_path)
        figure_path = plot_dispersion_curves(
            dispersion_table,
            figure_dir / f"{stem}_dispersion.png",
            title=_dispersion_plot_title(input_path),
        )
        dispersion_figures[stem] = figure_path

    figures["dispersion"] = dispersion_figures
    modes = ("dispersion",)
    detection = {
        "grid_scan_spacing": {"enabled": False, "reason": "CST dispersion data bypasses S-parameter marker analysis"},
        "dispersion": {
            "enabled": True,
            "reason": "parseable CST phase-vs-frequency dispersion text export detected",
            "input_count": len(dispersion_inputs),
        },
    }
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


def _dispersion_plot_title(input_path: Path) -> str:
    label = input_path.stem.replace("_", " ").replace("-", " ")
    while "  " in label:
        label = label.replace("  ", " ")
    return f"CST dispersion: frequency vs phase advance ({label})"


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


def detect_analysis_modes(tables: dict[str, pd.DataFrame], *, dataset_category: str | None = None) -> tuple[str, ...]:
    """Return standard and auto-detected analysis modes for a table bundle."""

    modes = list(BASE_ANALYSIS_MODES)
    report = _detection_report(tables, dataset_category=dataset_category)
    if report["grid_scan_spacing"]["enabled"]:
        modes.append("grid_scan_spacing")
    return tuple(modes)


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and not table.empty


def _detection_report(
    tables: dict[str, pd.DataFrame],
    *,
    dataset_category: str | None = None,
) -> dict[str, dict[str, object]]:
    marker_points = tables.get("marker_points")
    enabled, reason = _is_simulation_grid_scan(marker_points, dataset_category=dataset_category)
    return {"grid_scan_spacing": {"enabled": enabled, "reason": reason}}


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


def _dataset_category(path: Path) -> str | None:
    try:
        data_layer = detect_data_layer(path)
    except ValueError:
        return None
    return dataset_identity_from_path(path, data_layer).category


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
