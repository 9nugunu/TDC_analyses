"""High-level one-folder analysis runner."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.field_energy_ratio import (
    compute_cell_iris_field_energy_ratios,
    summarize_cell_iris_field_energy_ratios,
)
from deflector_tuning.analysis.grid_scan_spacing import (
    summarize_marker_spacing_for_grid_scan,
)
from deflector_tuning.analysis.marker_pipeline import (
    build_marker_analysis,
    save_marker_analysis,
)
from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.one_port_matrix import (
    Lane,
    detect_one_port_matrix_lane,
    extract_one_port_marker_frequencies,
    load_y11_touchstone_folder,
    load_z11_touchstone_folder,
    sample_y11_markers,
    sample_z11_markers,
)
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import dataset_identity_from_path
from deflector_tuning.data_loading.field3d import (
    Field3DPair,
    find_field3d_pairs as _find_field3d_pairs,
)
from deflector_tuning.data_loading.source_layer import detect_data_layer
from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS, ProjectDefaults
from deflector_tuning.dispersion import (
    load_cst_dispersion_txt,
    process_cst_dispersion_txt,
)
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.visualization.em_field_structure_plots import (
    FieldProfileExport as FieldProfileExport,
    load_field_profile_export,
    plot_field_profile_with_tdc_structure,
)
from deflector_tuning.visualization.grid_scan_phase_line_plots import (
    plot_grid_scan_sparameter_phase_r_c_line_scan,
)
from deflector_tuning.visualization.grid_scan_spacing_maps import (
    plot_grid_scan_spacing_error_maps,
)
from deflector_tuning.visualization.geometry_phase_response_plots import (
    plot_geometry_phase_response,
)
from deflector_tuning.visualization.kyhl_admittance_plots import (
    plot_f2pi3_normalized_admittance_view,
)
from deflector_tuning.visualization.nodal_shift_plots import plot_nodal_shift
from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.polar_phase_views import (
    plot_marker_phase_polar_views,
)
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers
from deflector_tuning.visualization.admittance_sweep_plots import plot_z11_marker_sweep
from deflector_tuning.visualization.raw_y11_plots import (
    plot_y11_raw_complex_sweep,
    plot_y11_raw_frequency_with_markers,
    plot_y11_raw_grid,
    plot_y11_raw_polar_views,
)
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.cell_iris_response_plots import (
    plot_cell_iris_response_comparison,
)
from deflector_tuning.visualization.coupler_cavity_parameter_plots import (
    plot_coupler_cavity_parameters,
)
from deflector_tuning.workflows.dispersion import (
    find_cst_dispersion_inputs as _find_cst_dispersion_inputs,
    run_dispersion_analysis as _run_dispersion_only_analysis,
)
from deflector_tuning.workflows.field3d import (
    run_field3d_analysis as _run_field3d_analysis,
)
from deflector_tuning.workflows.manifest import (
    cached_manifest_figures as _cached_manifest_figures,
    cached_manifest_figure_group as _cached_manifest_figure_group,
    write_manifest as _write_manifest,
)
from deflector_tuning.workflows.mode_detection import (
    BASE_ANALYSIS_MODES as BASE_ANALYSIS_MODES,
    GRID_SCAN_REQUIRED_COLUMNS as GRID_SCAN_REQUIRED_COLUMNS,
    GRID_SCAN_REQUIRED_MARKERS as GRID_SCAN_REQUIRED_MARKERS,
    detect_analysis_modes,
    detection_report as _detection_report,
)
from deflector_tuning.workflows.models import (
    AnalysisPaths,
    FigurePaths,
    RunResult,
)
from deflector_tuning.workflows.profile import (
    find_cst_profile_inputs as _find_cst_profile_inputs,
    run_profile_analysis as _run_profile_only_analysis,
)
from deflector_tuning.table_export import TableSaveResult, save_table_contract

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = DEFAULT_PROJECT_DEFAULTS.default_dispersion_subpath
logger = logging.getLogger(__name__)


class _RunScopedDataLoader(DataLoader):
    __slots__ = ("_delegate", "_sparameter_path", "_sparameter_table")

    def __init__(self, delegate: DataLoader, sparameter_path: Path) -> None:
        self._delegate = delegate
        self._sparameter_path = sparameter_path
        self._sparameter_table: pd.DataFrame | None = None

    def load(self, path: str | Path) -> pd.DataFrame:
        if Path(path) == self._sparameter_path and self._sparameter_table is not None:
            return self._sparameter_table
        table = self._delegate.load(path)
        if Path(path) == self._sparameter_path:
            self._sparameter_table = table
        return table


def run_folder_analysis(
    *,
    sparameter_path: str | Path,
    dispersion_path: str | Path | None = None,
    output_dir: str | Path,
    marker_role: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    file_workers: int = 1,
    plot_workers: int = 1,
    tables_only: bool = False,
    loader: DataLoader | None = None,
    project_defaults: ProjectDefaults = DEFAULT_PROJECT_DEFAULTS,
    marker_correction: TemperatureHumidityCorrection | None = None,
) -> RunResult:
    """Run the standard one-folder marker workflow and write tables/figures.

    The runner intentionally only orchestrates existing loader, analysis, and
    visualization modules. Optional analyses, such as simulation 2D grid-scan
    spacing maps, are enabled from conservative checks on the analysis tables.
    """

    logger.info(
        "Starting folder analysis: sparameter_path=%s dispersion_path=%s output_dir=%s marker_role=%s tables_only=%s",
        sparameter_path,
        dispersion_path,
        output_dir,
        marker_role,
        tables_only,
    )
    sparameter_path, dispersion_path = resolve_input_paths(
        sparameter_path,
        dispersion_path=dispersion_path,
        data_root=data_root,
        default_dispersion_subpath=project_defaults.default_dispersion_subpath,
    )
    plot_config = _plot_config_from_defaults(project_defaults)
    dataset_category = _dataset_category(sparameter_path)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_root = output_dir / "figures"
    manifest_path = output_dir / "manifest.json"
    matrix_lane = detect_one_port_matrix_lane(sparameter_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_category == "profile":
        if tables_only:
            raise ValueError("--tables-only is supported only for standard marker-analysis datasets")
        field3d_pairs = find_cst_field3d_pairs(sparameter_path)
        if field3d_pairs:
            return _run_field3d_only_analysis_from_runner(
                field3d_pairs,
                output_dir=output_dir,
                table_dir=table_dir,
                figure_root=figure_root,
                sparameter_path=sparameter_path,
                dispersion_path=dispersion_path,
                marker_role=marker_role,
            )
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

    dispersion_inputs = (
        find_cst_dispersion_inputs(sparameter_path)
        if dataset_category in {None, "dispersion"}
        else ()
    )
    if dataset_category == "dispersion" and not dispersion_inputs:
        raise ValueError(
            f"Dataset category is dispersion, but no parseable CST dispersion txt export was found: {sparameter_path}"
        )
    if dispersion_inputs:
        if tables_only:
            raise ValueError("--tables-only is supported only for standard marker-analysis datasets")
        logger.info("Detected CST dispersion data; running dispersion-only analysis")
        result = _run_dispersion_only_analysis(
            dispersion_inputs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
            load_dispersion=load_cst_dispersion_txt,
            process_dispersion=process_cst_dispersion_txt,
            plot_dispersion=plot_dispersion_curves,
            manifest_writer=_write_manifest,
        )
        logger.info("Dispersion-only analysis completed successfully")
        return result

    field3d_pairs = (
        find_cst_field3d_pairs(sparameter_path) if dataset_category is None else ()
    )
    if field3d_pairs:
        if tables_only:
            raise ValueError("--tables-only is supported only for standard marker-analysis datasets")
        return _run_field3d_only_analysis_from_runner(
            field3d_pairs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )

    profile_inputs = (
        find_cst_profile_inputs(sparameter_path) if dataset_category is None else ()
    )
    if profile_inputs:
        if tables_only:
            raise ValueError("--tables-only is supported only for standard marker-analysis datasets")
        return _run_profile_only_analysis_from_runner(
            profile_inputs,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )

    if matrix_lane is not None:
        return _run_one_port_matrix_analysis_from_runner(
            lane=matrix_lane,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            output_dir=output_dir,
            table_dir=table_dir,
            figure_root=figure_root,
            marker_role=marker_role,
            plot_config=plot_config,
            tables_only=tables_only,
        )

    sparameter_table: pd.DataFrame | None = None
    loader = _RunScopedDataLoader(
        loader or DataLoader(file_workers=file_workers),
        sparameter_path,
    )
    logger.info(
        "Resolved input paths: sparameter=%s dispersion=%s",
        sparameter_path,
        dispersion_path,
    )
    cached_s11_figures = (
        _cached_manifest_figure_group(manifest_path, "s11")
        if dataset_category == "grid" and not tables_only
        else OrderedDict()
    )
    sparameter_table: pd.DataFrame | None = None
    if tables_only:
        logger.info("Skipping figure-only S11 table load in tables-only mode")
    elif cached_s11_figures:
        logger.info(
            "Skipping S11 table load because cached grid-scan S11 figures already exist"
        )
    elif sparameter_table is None:
        sparameter_table = _load_s11_table_for_figures(loader, sparameter_path)
    logger.info("Building marker analysis tables")
    tables = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        loader=loader,
        sparameter_table=sparameter_table,
        marker_correction=marker_correction,
    )
    logger.info("Built analysis tables: %s", ", ".join(tables.keys()))
    logger.info("Saving analysis tables to %s", table_dir)
    save_result = save_marker_analysis(tables, table_dir)
    if isinstance(save_result, TableSaveResult):
        table_paths = AnalysisPaths(save_result.paths)
        table_constants = save_result.constants
    else:
        # Keep test/integration adapters that still return a plain mapping usable.
        table_paths = AnalysisPaths(save_result)
        table_constants = {}
    logger.info("Saved %d analysis tables", len(table_paths))
    modes = detect_analysis_modes(tables, dataset_category=dataset_category)
    logger.info("Enabled analysis modes: %s", ", ".join(modes))

    if tables_only:
        figures = _cached_manifest_figures(manifest_path)
        detection = _detection_report(tables, dataset_category=dataset_category)
        logger.info("Writing tables-only manifest to %s", manifest_path)
        _write_manifest(
            manifest_path,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
            modes=modes,
            detection=detection,
            tables=table_paths,
            figures=figures,
            table_contract="standard",
            table_schema_version=2,
            table_constants=table_constants,
        )
        logger.info("Tables-only analysis completed successfully")
        return RunResult(
            output_dir=output_dir,
            tables=table_paths,
            figures=figures,
            analysis_modes=modes,
            manifest_path=manifest_path,
        )

    figures: FigurePaths = OrderedDict()
    if "grid_scan_spacing" in modes and cached_s11_figures:
        logger.info(
            "Skipping S11 figures because cached grid-scan S11 figures already exist"
        )
        figures["s11"] = cached_s11_figures
    else:
        if sparameter_table is None:
            sparameter_table = _load_s11_table_for_figures(loader, sparameter_path)
        logger.info("Rendering S11 figures")
        figures["s11"] = OrderedDict(
            plot_s11_with_markers(
                sparameter_table,
                tables["marker_pts"],
                figure_root / "s11",
                render_workers=plot_workers,
                config=plot_config,
            )
        )
    if _has_rows(tables.get("phase_adv")):
        logger.info("Rendering phase advance figures")
        figures["phase_advance"] = OrderedDict(
            plot_phase_advance(
                tables["phase_adv"],
                figure_root / "phase_advance",
                split_by_family=True,
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping phase advance figures because phase_advance is missing or empty"
        )
    if _has_rows(tables.get("nodal_shift")):
        logger.info("Rendering nodal-shift figures")
        figures["nodal_shift"] = OrderedDict(
            plot_nodal_shift(
                tables["nodal_shift"],
                figure_root / "nodal",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping nodal-shift figures because nodal_shift is missing or empty"
        )
    if _has_rows(tables.get("cell_iris_cmp")):
        logger.info("Rendering cell-iris response figures")
        figures["cell_iris_response"] = OrderedDict(
            plot_cell_iris_response_comparison(
                tables["cell_iris_cmp"],
                figure_root / "cell_iris_response",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping cell-iris response figures because cell_iris_response_comparison is missing or empty"
        )
    if _has_rows(tables.get("coupler_params")):
        logger.info("Rendering coupler-cavity parameter figures")
        figures["coupler_cavity_parameters"] = OrderedDict(
            plot_coupler_cavity_parameters(
                tables["coupler_params"],
                figure_root / "coupler_cavity_parameters",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping coupler-cavity parameter figures because coupler_cavity_parameter_estimates is missing or empty"
        )
    if _has_rows(tables.get("geom_phase")):
        logger.info("Rendering geometry phase-response figures")
        figures["geometry_phase_response"] = OrderedDict(
            plot_geometry_phase_response(
                tables["geom_phase"],
                figure_root / "geometry_phase_response",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping geometry phase-response figures because geometry_phase_response is missing or empty"
        )
    if _has_rows(tables.get("marker_pts")):
        logger.info("Rendering polar phase figures")
        figures["polar"] = OrderedDict(
            plot_marker_phase_polar_views(
                tables["marker_pts"], figure_root / "polar", config=plot_config
            )
        )
    else:
        logger.info(
            "Skipping polar phase figures because marker_points is missing or empty"
        )
    if _has_rows(tables.get("kyhl_admit_audit")):
        logger.info("Rendering f_2pi3 normalized admittance figure")
        figures["kyhl_normalized_admittance"] = OrderedDict(
            plot_f2pi3_normalized_admittance_view(
                tables["kyhl_admit_audit"],
                figure_root / "kyhl_normalized_admittance",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping f_2pi3 normalized admittance figure because "
            "kyhl_f2pi3_normalized_admittance_audit is missing or empty"
        )

    detection = _detection_report(tables, dataset_category=dataset_category)
    if "grid_scan_spacing" in modes:
        logger.info("Rendering grid-scan spacing figures")
        spacing_summary = summarize_marker_spacing_for_grid_scan(
            tables["marker_pts"]
        )
        figures["grid_scan_spacing"] = OrderedDict(
            plot_grid_scan_spacing_error_maps(
                spacing_summary,
                figure_root / "grid_scan_spacing",
                config=plot_config,
            )
        )
        logger.info("Rendering grid-scan S-parameter phase r_c line scan")
        figures["grid_scan_sparameter_phase_r_c_line_scan"] = OrderedDict(
            plot_grid_scan_sparameter_phase_r_c_line_scan(
                tables["marker_pts"],
                figure_root / "grid_scan_sparameter_phase_r_c_line_scan",
                config=plot_config,
            )
        )
    else:
        logger.info(
            "Skipping grid-scan spacing figures: %s",
            detection["grid_scan_spacing"]["reason"],
        )

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
        table_contract="standard",
        table_schema_version=2,
        table_constants=table_constants,
    )
    logger.info("Folder analysis completed successfully")

    return RunResult(
        output_dir=output_dir,
        tables=table_paths,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )


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
        load_profile=load_field_profile_export,
        plot_profile=plot_field_profile_with_tdc_structure,
        compute_field_ratios=compute_cell_iris_field_energy_ratios,
        summarize_field_ratios=summarize_cell_iris_field_energy_ratios,
        manifest_writer=_write_manifest,
    )
    logger.info("Profile-only analysis completed successfully")
    return result


def _run_field3d_only_analysis_from_runner(
    pairs: tuple[Field3DPair, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
) -> RunResult:
    logger.info("Detected CST 3D complex E/H data; running field3d analysis")
    result = _run_field3d_analysis(
        pairs,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_root=figure_root,
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
    )
    logger.info("3D field analysis completed successfully")
    return result


def _run_one_port_matrix_analysis_from_runner(
    *,
    lane: Lane,
    sparameter_path: Path,
    dispersion_path: Path,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    marker_role: str,
    plot_config: PlotConfig,
    tables_only: bool,
) -> RunResult:
    """Run direct one-port Y11/Z11 analysis outside the S-parameter pipeline."""

    parameter = lane[0].upper() + "11"
    logger.info("Detected direct CST %s Touchstone exports", parameter)
    if lane == "y11":
        matrix_table = load_y11_touchstone_folder(sparameter_path)
    else:
        matrix_table = load_z11_touchstone_folder(sparameter_path)
    markers = extract_one_port_marker_frequencies(
        dispersion_path,
        marker_role=marker_role,
    )
    if lane == "y11":
        marker_points = sample_y11_markers(matrix_table, markers)
        point_key = "y11_pts"
    else:
        marker_points = sample_z11_markers(matrix_table, markers)
        point_key = "z11_pts"
    save_result = save_table_contract(
        OrderedDict((("markers", markers), (point_key, marker_points))),
        table_dir,
        lane,
    )
    tables = AnalysisPaths(save_result.paths)

    manifest_path = output_dir / "manifest.json"
    if tables_only:
        figures = _cached_manifest_figures(manifest_path)
    elif lane == "y11":
        figures = FigurePaths()
        raw_frequency = OrderedDict(
            plot_y11_raw_frequency_with_markers(
                matrix_table,
                marker_points,
                figure_root / "y11_raw",
                config=plot_config,
            )
        )
        raw_frequency["complex_sweep"] = plot_y11_raw_complex_sweep(
            marker_points,
            figure_root / "y11_raw" / "complex_sweep.png",
            config=plot_config,
        )
        figures["y11_raw"] = raw_frequency
        figures["polar_raw"] = OrderedDict(
            plot_y11_raw_polar_views(
                marker_points,
                figure_root / "polar_raw",
                config=plot_config,
            )
        )
        if "sim_r_c" in marker_points and marker_points["sim_r_c"].notna().any():
            figures["grid_raw"] = OrderedDict(
                plot_y11_raw_grid(
                    marker_points,
                    figure_root / "grid_raw",
                    config=plot_config,
                )
            )
    else:
        figure_path = plot_z11_marker_sweep(
            marker_points,
            figure_root / "z11_complex" / "z11_marker_sweep.png",
            config=plot_config,
        )
        figures = FigurePaths(
            (("z11_complex", OrderedDict((("marker_sweep", figure_path),))),)
        )
    mode = "y11_admittance" if lane == "y11" else "z11_impedance"
    modes = (mode,)
    detection = {
        mode: {
            "enabled": True,
            "reason": f"direct CST Touchstone {parameter} files detected",
        }
    }
    _write_manifest(
        manifest_path,
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        modes=modes,
        detection=detection,
        tables=tables,
        figures=figures,
        table_contract=lane,
        table_schema_version=2,
        table_constants=save_result.constants,
    )
    logger.info("Direct %s analysis completed successfully", parameter)
    return RunResult(
        output_dir=output_dir,
        tables=tables,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )


def find_cst_profile_inputs(path: str | Path) -> tuple[Path, ...]:
    return _find_cst_profile_inputs(path, load_profile=load_field_profile_export)


def find_cst_field3d_pairs(path: str | Path) -> tuple[Field3DPair, ...]:
    return _find_field3d_pairs(path)


def find_cst_dispersion_inputs(path: str | Path) -> tuple[Path, ...]:
    return _find_cst_dispersion_inputs(path, load_dispersion=load_cst_dispersion_txt)


def resolve_input_paths(
    sparameter_path: str | Path,
    *,
    dispersion_path: str | Path | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    default_dispersion_subpath: str | Path = DEFAULT_DISPERSION_SUBPATH,
) -> tuple[Path, Path]:
    """Resolve data-relative inputs and default dispersion folder."""

    data_root = Path(data_root)
    resolved_sparameter_path = _data_relative_path(sparameter_path, data_root=data_root)
    if dispersion_path is None:
        resolved_dispersion_path = data_root / default_dispersion_subpath
    else:
        resolved_dispersion_path = _data_relative_path(
            dispersion_path, data_root=data_root
        )
    return resolved_sparameter_path, resolved_dispersion_path


def _plot_config_from_defaults(defaults: ProjectDefaults) -> PlotConfig:
    return PlotConfig(
        design_point_by_axis=dict(defaults.design_point_by_axis),
        ideal_phase_guide_angles_deg=defaults.ideal_phase_guide_angles_deg,
    )


def _data_relative_path(path: str | Path, *, data_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.parts[:1] == (data_root.name,):
        return path
    return data_root / path


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and not table.empty


def _load_s11_table_for_figures(
    loader: DataLoader, sparameter_path: Path
) -> pd.DataFrame:
    logger.info("Loading S-parameter table from %s", sparameter_path)
    loaded_sparameter_table = loader.load(sparameter_path)
    logger.info("Loaded S-parameter table with %d rows", len(loaded_sparameter_table))
    sparameter_table = select_s11_rows(loaded_sparameter_table)
    if len(sparameter_table) != len(loaded_sparameter_table):
        logger.info(
            "Selected %d S11 rows for marker analysis outputs", len(sparameter_table)
        )
    return sparameter_table


def _dataset_category(path: Path) -> str | None:
    try:
        data_layer = detect_data_layer(path)
        return dataset_identity_from_path(path, data_layer).category
    except ValueError:
        return None
