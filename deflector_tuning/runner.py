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
from deflector_tuning.analysis.field_energy_ratio import (
    compute_cell_iris_field_energy_ratios,
    summarize_cell_iris_field_energy_ratios,
)
from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, save_marker_analysis
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import dataset_identity_from_path
from deflector_tuning.data_loading.source_layer import detect_data_layer
from deflector_tuning.dispersion import load_cst_dispersion_txt, process_cst_dispersion_txt
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.visualization.grid_scan_phase_line_plots import plot_grid_scan_sparameter_phase_r_c_line_scan
from deflector_tuning.visualization.grid_scan_spacing_maps import plot_grid_scan_spacing_error_maps
from deflector_tuning.visualization.em_field_structure_plots import (
    FieldProfileExport,
    load_field_profile_export,
    plot_field_profile_with_tdc_structure,
)
from deflector_tuning.visualization.geometry_phase_response_plots import plot_geometry_phase_response
from deflector_tuning.visualization.nodal_shift_plots import plot_nodal_shift
from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers
from deflector_tuning.visualization.cell_iris_response_plots import plot_cell_iris_response_comparison
from deflector_tuning.visualization.coupler_cavity_parameter_plots import plot_coupler_cavity_parameters

AnalysisPaths = OrderedDict[str, Path]
FigurePaths = OrderedDict[str, OrderedDict[str, Path]]

BASE_ANALYSIS_MODES: tuple[str, ...] = ("marker_analysis", "s11", "phase_advance", "nodal_shift", "polar")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = Path("sim") / "sim_dispersion_260505_single_cell_step1"
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
    file_workers: int = 1,
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
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_category == "profile":
        profile_inputs = find_cst_profile_inputs(sparameter_path)
        if not profile_inputs:
            raise ValueError(
                "Dataset category is profile, but no parseable CST profile txt export "
                f"was found: {sparameter_path}"
            )
        return _run_profile_only_analysis_from_runner(
            profile_inputs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )

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

    profile_inputs = find_cst_profile_inputs(sparameter_path) if dataset_category is None else ()
    if profile_inputs:
        return _run_profile_only_analysis_from_runner(
            profile_inputs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )

    loader = loader or DataLoader(file_workers=file_workers)
    logger.info("Resolved input paths: sparameter=%s dispersion=%s", sparameter_path, dispersion_path)
    cached_s11_figures = (
        _cached_manifest_figure_group(manifest_path, "s11")
        if dataset_category == "grid"
        else OrderedDict()
    )
    sparameter_table: pd.DataFrame | None = None
    if cached_s11_figures:
        logger.info("Skipping S11 table load because cached grid-scan S11 figures already exist")
    else:
        sparameter_table = _load_s11_table_for_figures(loader, sparameter_path)
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
    if "grid_scan_spacing" in modes and cached_s11_figures:
        logger.info("Skipping S11 figures because cached grid-scan S11 figures already exist")
        figures["s11"] = cached_s11_figures
    else:
        if sparameter_table is None:
            sparameter_table = _load_s11_table_for_figures(loader, sparameter_path)
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
    if _has_rows(tables.get("nodal_shift")):
        logger.info("Rendering nodal-shift figures")
        figures["nodal_shift"] = OrderedDict(
            plot_nodal_shift(tables["nodal_shift"], figure_root / "nodal")
        )
    else:
        logger.info("Skipping nodal-shift figures because nodal_shift is missing or empty")
    if _has_rows(tables.get("cell_iris_response_comparison")):
        logger.info("Rendering cell-iris response figures")
        figures["cell_iris_response"] = OrderedDict(
            plot_cell_iris_response_comparison(
                tables["cell_iris_response_comparison"],
                figure_root / "cell_iris_response",
            )
        )
    else:
        logger.info(
            "Skipping cell-iris response figures because cell_iris_response_comparison is missing or empty"
        )
    if _has_rows(tables.get("coupler_cavity_parameter_estimates")):
        logger.info("Rendering coupler-cavity parameter figures")
        figures["coupler_cavity_parameters"] = OrderedDict(
            plot_coupler_cavity_parameters(
                tables["coupler_cavity_parameter_estimates"],
                figure_root / "coupler_cavity_parameters",
            )
        )
    else:
        logger.info(
            "Skipping coupler-cavity parameter figures because coupler_cavity_parameter_estimates is missing or empty"
        )
    if _has_rows(tables.get("geometry_phase_response")):
        logger.info("Rendering geometry phase-response figures")
        figures["geometry_phase_response"] = OrderedDict(
            plot_geometry_phase_response(tables["geometry_phase_response"], figure_root / "geometry_phase_response")
        )
    else:
        logger.info("Skipping geometry phase-response figures because geometry_phase_response is missing or empty")
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
        logger.info("Rendering grid-scan S-parameter phase r_c line scan")
        figures["grid_scan_sparameter_phase_r_c_line_scan"] = OrderedDict(
            plot_grid_scan_sparameter_phase_r_c_line_scan(
                tables["marker_points"],
                figure_root / "grid_scan_sparameter_phase_r_c_line_scan",
            )
        )
    else:
        logger.info("Skipping grid-scan spacing figures: %s", detection["grid_scan_spacing"]["reason"])

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


def find_cst_profile_inputs(path: str | Path) -> tuple[Path, ...]:
    """Return parseable one-dimensional CST profile exports under ``path``."""

    input_path = Path(path)
    candidates: list[Path]
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("*.txt"), key=lambda item: item.name.lower())
    else:
        return ()

    profile_paths: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.lower() != ".txt":
            continue
        try:
            load_field_profile_export(candidate)
        except (OSError, ValueError):
            continue
        profile_paths.append(candidate)
    return tuple(profile_paths)


def _run_profile_only_analysis_from_runner(
    profile_inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
) -> RunResult:
    logger.info("Detected CST profile data; running profile-only analysis")
    result = _run_profile_only_analysis(
        profile_inputs,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_root=figure_root,
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
    )
    logger.info("Profile-only analysis completed successfully")
    return result


def _run_profile_only_analysis(
    profile_inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
) -> RunResult:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = figure_root / "profile"
    table_paths: AnalysisPaths = AnalysisPaths()
    figures: FigurePaths = OrderedDict()
    profile_figures: OrderedDict[str, Path] = OrderedDict()
    summary_rows: list[dict[str, object]] = []
    field_energy_ratio_tables: list[pd.DataFrame] = []
    field_energy_summary_tables: list[pd.DataFrame] = []

    for input_path in profile_inputs:
        export = load_field_profile_export(input_path)
        summary_rows.extend(_profile_summary_rows(input_path, export))
        field_ratios = _field_energy_ratio_table(input_path, export)
        if not field_ratios.empty:
            field_energy_ratio_tables.append(field_ratios)
            field_summary = summarize_cell_iris_field_energy_ratios(field_ratios)
            field_summary.insert(0, "source_file", input_path.name)
            field_energy_summary_tables.append(field_summary)
        profile_figures[input_path.stem] = plot_field_profile_with_tdc_structure(
            export,
            figure_dir / f"{input_path.stem}.png",
        )
        for figure_key, split_export in _split_phase_profile_export_by_field_kind(input_path, export):
            profile_figures[figure_key] = plot_field_profile_with_tdc_structure(
                split_export,
                figure_dir / f"{figure_key}.png",
            )

    summary_path = table_dir / "profile_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    table_paths["profile_summary"] = summary_path
    if field_energy_ratio_tables:
        field_pairs_path = table_dir / "field_energy_ratio_pairs.csv"
        field_summary_path = table_dir / "field_energy_ratio_summary.csv"
        pd.concat(field_energy_ratio_tables, ignore_index=True).to_csv(field_pairs_path, index=False)
        pd.concat(field_energy_summary_tables, ignore_index=True).to_csv(field_summary_path, index=False)
        table_paths["field_energy_ratio_pairs"] = field_pairs_path
        table_paths["field_energy_ratio_summary"] = field_summary_path
    figures["profile"] = profile_figures
    modes = ("profile",)
    detection = {
        "profile": {
            "enabled": True,
            "reason": "parseable CST quantity-versus-z profile text export detected",
            "input_count": len(profile_inputs),
        },
        "grid_scan_spacing": {"enabled": False, "reason": "profile data bypasses S-parameter marker analysis"},
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


def _profile_summary_rows(input_path: Path, export: FieldProfileExport) -> list[dict[str, object]]:
    rows = []
    for trace in export.traces:
        abs_values = abs(trace.values)
        peak_index = int(abs_values.argmax())
        rows.append(
            {
                "source_file": input_path.name,
                "trace_label": trace.label,
                "field_kind": trace.field_kind,
                "component": trace.component,
                "value_kind": trace.value_kind,
                "sample_count": len(trace.z_mm),
                "z_min_mm": float(trace.z_mm.min()),
                "z_max_mm": float(trace.z_mm.max()),
                "value_min": float(trace.values.min()),
                "value_max": float(trace.values.max()),
                "abs_peak_value": float(abs_values[peak_index]),
                "abs_peak_z_mm": float(trace.z_mm[peak_index]),
            }
        )
    return rows


def _field_energy_ratio_table(input_path: Path, export: FieldProfileExport) -> pd.DataFrame:
    try:
        table = compute_cell_iris_field_energy_ratios(export)
    except ValueError:
        return pd.DataFrame()
    table = table.copy()
    table.insert(0, "source_file", input_path.name)
    return table


def _split_phase_profile_export_by_field_kind(
    input_path: Path, export: FieldProfileExport
) -> tuple[tuple[str, FieldProfileExport], ...]:
    if not export.traces or any(trace.value_kind != "phase" for trace in export.traces):
        return ()

    field_kinds = []
    for trace in export.traces:
        if trace.field_kind not in field_kinds:
            field_kinds.append(trace.field_kind)
    if len(field_kinds) < 2:
        return ()

    outputs: list[tuple[str, FieldProfileExport]] = []
    for field_kind in field_kinds:
        traces = tuple(trace for trace in export.traces if trace.field_kind == field_kind)
        if not traces:
            continue
        outputs.append(
            (
                _phase_profile_figure_key(input_path, field_kind),
                FieldProfileExport(parameters=export.parameters, traces=traces),
            )
        )
    return tuple(outputs)


def _phase_profile_figure_key(input_path: Path, field_kind: str) -> str:
    prefix = {"e": "E", "h": "H"}.get(field_kind.lower()[:1], field_kind.upper())
    stem = input_path.stem
    if stem.startswith("EM_"):
        return f"{prefix}_{stem.removeprefix('EM_')}"
    return f"{prefix}_{stem}"


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

    use_source_prefix = len(dispersion_inputs) > 1
    for input_path in dispersion_inputs:
        outputs = process_cst_dispersion_txt(input_path, output_dir=csv_dir)
        stem = input_path.stem
        table_paths[f"{stem}_long"] = outputs.long_csv
        table_paths[f"{stem}_wide"] = outputs.wide_csv
        table_paths[f"{stem}_summary"] = outputs.summary_csv

        dispersion_table = load_cst_dispersion_txt(input_path)
        figure_prefix = _dispersion_figure_prefix(stem, use_source_prefix=use_source_prefix)
        overview_key = _dispersion_figure_key(figure_prefix, "all_modes")
        overview_path = plot_dispersion_curves(
            dispersion_table,
            figure_dir / f"{overview_key}.png",
            title=_dispersion_plot_title(input_path),
        )
        dispersion_figures[overview_key] = overview_path
        for mode_index in _dispersion_mode_indices(dispersion_table):
            mode_key = _dispersion_figure_key(figure_prefix, f"mode_{mode_index:02d}")
            dispersion_figures[mode_key] = plot_dispersion_curves(
                dispersion_table,
                figure_dir / f"{mode_key}.png",
                mode_indices=(mode_index,),
                title=_dispersion_mode_plot_title(input_path, mode_index),
            )

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
    return "All modes dispersion"


def _dispersion_mode_plot_title(input_path: Path, mode_index: int) -> str:
    return f"Mode {mode_index:02d} dispersion"


def _dispersion_mode_indices(dispersion_table: pd.DataFrame) -> tuple[int, ...]:
    modes = pd.to_numeric(dispersion_table["mode_index"], errors="coerce").dropna().unique()
    return tuple(sorted(int(mode) for mode in modes))


def _dispersion_figure_prefix(stem: str, *, use_source_prefix: bool) -> str:
    if not use_source_prefix:
        return ""
    return stem.lower().replace("-", "_")


def _dispersion_figure_key(prefix: str, label: str) -> str:
    if not prefix:
        return label
    return f"{prefix}_{label}"


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
    if report["coupler_cavity_parameters"]["enabled"]:
        modes.append("coupler_cavity_parameters")
    if report["cell_iris_response"]["enabled"]:
        modes.append("cell_iris_response")
    if report["geometry_phase_response"]["enabled"]:
        modes.append("geometry_phase_response")
    if report["grid_scan_spacing"]["enabled"]:
        modes.append("grid_scan_spacing")
    return tuple(modes)


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and not table.empty


def _load_s11_table_for_figures(loader: DataLoader, sparameter_path: Path) -> pd.DataFrame:
    logger.info("Loading S-parameter table from %s", sparameter_path)
    loaded_sparameter_table = loader.load(sparameter_path)
    logger.info("Loaded S-parameter table with %d rows", len(loaded_sparameter_table))
    sparameter_table = select_s11_rows(loaded_sparameter_table)
    if len(sparameter_table) != len(loaded_sparameter_table):
        logger.info("Selected %d S11 rows for marker analysis outputs", len(sparameter_table))
    return sparameter_table


def _detection_report(
    tables: dict[str, pd.DataFrame],
    *,
    dataset_category: str | None = None,
) -> dict[str, dict[str, object]]:
    marker_points = tables.get("marker_points")
    grid_enabled, grid_reason = _is_simulation_grid_scan(marker_points, dataset_category=dataset_category)
    coupler_enabled, coupler_reason = _has_coupler_cavity_parameters(
        tables.get("coupler_cavity_parameter_estimates")
    )
    cell_iris_enabled, cell_iris_reason = _has_cell_iris_response(tables.get("cell_iris_response_comparison"))
    response_enabled, response_reason = _has_geometry_phase_response(tables.get("geometry_phase_response"))
    return {
        "coupler_cavity_parameters": {"enabled": coupler_enabled, "reason": coupler_reason},
        "cell_iris_response": {"enabled": cell_iris_enabled, "reason": cell_iris_reason},
        "geometry_phase_response": {"enabled": response_enabled, "reason": response_reason},
        "grid_scan_spacing": {"enabled": grid_enabled, "reason": grid_reason},
    }


def _has_coupler_cavity_parameters(table: pd.DataFrame | None) -> tuple[bool, str]:
    if table is None or table.empty:
        return False, "coupler_cavity_parameter_estimates table is missing or empty"
    return True, f"coupler_cavity_parameter_estimates has {len(table)} estimate rows"


def _has_cell_iris_response(table: pd.DataFrame | None) -> tuple[bool, str]:
    if table is None or table.empty:
        return False, "cell_iris_response_comparison table is missing or empty"
    return True, f"cell_iris_response_comparison has {len(table)} matched transition rows"


def _has_geometry_phase_response(table: pd.DataFrame | None) -> tuple[bool, str]:
    if table is None or table.empty:
        return False, "geometry_phase_response table is missing or empty"
    axes = sorted(table["sweep_axis"].dropna().astype(str).unique()) if "sweep_axis" in table else []
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
        return dataset_identity_from_path(path, data_layer).category
    except ValueError:
        return None


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


def _cached_manifest_figure_group(manifest_path: Path, group: str) -> OrderedDict[str, Path]:
    if not manifest_path.exists():
        return OrderedDict()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return OrderedDict()
    group_paths = manifest.get("outputs", {}).get("figures", {}).get(group, {})
    if not isinstance(group_paths, dict) or not group_paths:
        return OrderedDict()

    cached: OrderedDict[str, Path] = OrderedDict()
    for name, raw_path in group_paths.items():
        path = Path(raw_path)
        if not path.exists():
            return OrderedDict()
        cached[str(name)] = path
    return cached
