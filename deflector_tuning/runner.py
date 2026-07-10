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
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import dataset_identity_from_path
from deflector_tuning.data_loading.source_layer import detect_data_layer
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
from deflector_tuning.workflows.manifest import (
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

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = Path("sim") / "sim_dispersion_260505_single_cell_step1"
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

    profile_inputs = (
        find_cst_profile_inputs(sparameter_path) if dataset_category is None else ()
    )
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
        if dataset_category == "grid"
        else OrderedDict()
    )
    sparameter_table: pd.DataFrame | None = None
    if cached_s11_figures:
        logger.info(
            "Skipping S11 table load because cached grid-scan S11 figures already exist"
        )
    else:
        sparameter_table = _load_s11_table_for_figures(loader, sparameter_path)
    logger.info("Building marker analysis tables")
    tables = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        loader=loader,
        sparameter_table=sparameter_table,
    )
    logger.info("Built analysis tables: %s", ", ".join(tables.keys()))
    logger.info("Saving analysis tables to %s", table_dir)
    table_paths = AnalysisPaths(save_marker_analysis(tables, table_dir))
    logger.info("Saved %d analysis tables", len(table_paths))
    modes = detect_analysis_modes(tables, dataset_category=dataset_category)
    logger.info("Enabled analysis modes: %s", ", ".join(modes))

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
                sparameter_table, tables["marker_points"], figure_root / "s11"
            )
        )
    if _has_rows(tables.get("phase_advance")):
        logger.info("Rendering phase advance figures")
        figures["phase_advance"] = OrderedDict(
            plot_phase_advance(
                tables["phase_advance"],
                figure_root / "phase_advance",
                split_by_family=True,
            )
        )
    else:
        logger.info(
            "Skipping phase advance figures because phase_advance is missing or empty"
        )
    if _has_rows(tables.get("nodal_shift")):
        logger.info("Rendering nodal-shift figures")
        figures["nodal_shift"] = OrderedDict(
            plot_nodal_shift(tables["nodal_shift"], figure_root / "nodal")
        )
    else:
        logger.info(
            "Skipping nodal-shift figures because nodal_shift is missing or empty"
        )
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
            plot_geometry_phase_response(
                tables["geometry_phase_response"],
                figure_root / "geometry_phase_response",
            )
        )
    else:
        logger.info(
            "Skipping geometry phase-response figures because geometry_phase_response is missing or empty"
        )
    if _has_rows(tables.get("marker_points")):
        logger.info("Rendering polar phase figures")
        figures["polar"] = OrderedDict(
            plot_marker_phase_polar_views(
                tables["marker_points"], figure_root / "polar"
            )
        )
    else:
        logger.info(
            "Skipping polar phase figures because marker_points is missing or empty"
        )
    if _has_rows(tables.get("kyhl_f2pi3_normalized_admittance_audit")):
        logger.info("Rendering f_2pi3 normalized admittance figure")
        figures["kyhl_normalized_admittance"] = OrderedDict(
            plot_f2pi3_normalized_admittance_view(
                tables["kyhl_f2pi3_normalized_admittance_audit"],
                figure_root / "kyhl_normalized_admittance",
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
            tables["marker_points"]
        )
        figures["grid_scan_spacing"] = OrderedDict(
            plot_grid_scan_spacing_error_maps(
                spacing_summary, figure_root / "grid_scan_spacing"
            )
        )
        logger.info("Rendering grid-scan S-parameter phase r_c line scan")
        figures["grid_scan_sparameter_phase_r_c_line_scan"] = OrderedDict(
            plot_grid_scan_sparameter_phase_r_c_line_scan(
                tables["marker_points"],
                figure_root / "grid_scan_sparameter_phase_r_c_line_scan",
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


def find_cst_profile_inputs(path: str | Path) -> tuple[Path, ...]:
    return _find_cst_profile_inputs(path, load_profile=load_field_profile_export)


def find_cst_dispersion_inputs(path: str | Path) -> tuple[Path, ...]:
    return _find_cst_dispersion_inputs(path, load_dispersion=load_cst_dispersion_txt)


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
        resolved_dispersion_path = _data_relative_path(
            dispersion_path, data_root=data_root
        )
    return resolved_sparameter_path, resolved_dispersion_path


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
