"""Resolve one-folder inputs, detect their workflow, and execute it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from deflector_tuning.analysis.field_energy_ratio import (
    compute_cell_iris_field_energy_ratios,
    summarize_cell_iris_field_energy_ratios,
)
from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection
from deflector_tuning.data_loading.one_port_matrix import Lane, detect_one_port_matrix_lane
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.dataset_naming import dataset_identity_from_path
from deflector_tuning.data_loading.field3d import Field3DPair, find_field3d_pairs as _find_field3d_pairs
from deflector_tuning.data_loading.source_layer import detect_data_layer
from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS, ProjectDefaults
from deflector_tuning.dispersion import load_cst_dispersion_txt, process_cst_dispersion_txt
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.visualization.em_field_structure_plots import (
    FieldProfileExport as FieldProfileExport,
    load_field_profile_export,
    plot_field_profile_with_tdc_structure,
)
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.workflows.dispersion import (
    find_cst_dispersion_inputs as _find_cst_dispersion_inputs,
    run_dispersion_analysis as _run_dispersion_only_analysis,
)
from deflector_tuning.workflows.field3d import run_field3d_analysis as _run_field3d_analysis
from deflector_tuning.workflows.manifest import write_manifest as _write_manifest
from deflector_tuning.workflows.marker_analysis import run_marker_analysis
from deflector_tuning.workflows.mode_detection import (
    BASE_ANALYSIS_MODES as BASE_ANALYSIS_MODES,
    GRID_SCAN_REQUIRED_COLUMNS as GRID_SCAN_REQUIRED_COLUMNS,
    GRID_SCAN_REQUIRED_MARKERS as GRID_SCAN_REQUIRED_MARKERS,
    detect_analysis_modes as detect_analysis_modes,
)
from deflector_tuning.workflows.models import (
    AnalysisPaths as AnalysisPaths,
    FigurePaths as FigurePaths,
    RunResult,
)
from deflector_tuning.workflows.one_port import run_one_port_matrix_analysis
from deflector_tuning.workflows.profile import (
    find_cst_profile_inputs as _find_cst_profile_inputs,
    run_profile_analysis as _run_profile_only_analysis,
)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DISPERSION_SUBPATH = DEFAULT_PROJECT_DEFAULTS.default_dispersion_subpath
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FolderWorkflow:
    kind: Literal["marker", "profile", "dispersion", "field3d"]
    inputs: tuple[Path, ...] = ()
    field3d_pairs: tuple[Field3DPair, ...] = ()


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
    geometry_sweep_axis: str | None = None,
    geometry_sweep_base: float | None = None,
) -> RunResult:
    """Resolve inputs, detect the supported data workflow, and write its outputs.

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
    matrix_lane = detect_one_port_matrix_lane(sparameter_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = _detect_folder_workflow(
        sparameter_path, dataset_category=dataset_category, tables_only=tables_only
    )
    if workflow.kind == "profile":
        return _run_profile_only_analysis_from_runner(
            workflow.inputs,
            output_dir=output_dir,
            table_dir=output_dir / "tables",
            figure_root=output_dir / "figures",
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )
    if workflow.kind == "field3d":
        return _run_field3d_only_analysis_from_runner(
            workflow.field3d_pairs,
            output_dir=output_dir,
            table_dir=output_dir / "tables",
            figure_root=output_dir / "figures",
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            marker_role=marker_role,
        )
    if workflow.kind == "dispersion":
        logger.info("Detected CST dispersion data; running dispersion-only analysis")
        result = _run_dispersion_only_analysis(
            workflow.inputs,
            output_dir=output_dir,
            table_dir=output_dir / "tables",
            figure_root=output_dir / "figures",
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
    if matrix_lane is not None:
        return run_one_port_matrix_analysis(
            lane=matrix_lane,
            sparameter_path=sparameter_path,
            dispersion_path=dispersion_path,
            output_dir=output_dir,
            table_dir=output_dir / "tables",
            figure_root=output_dir / "figures",
            marker_role=marker_role,
            plot_config=plot_config,
            tables_only=tables_only,
        )
    return run_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        output_dir=output_dir,
        marker_role=marker_role,
        dataset_category=dataset_category,
        plot_config=plot_config,
        file_workers=file_workers,
        plot_workers=plot_workers,
        tables_only=tables_only,
        loader=loader,
        marker_correction=marker_correction,
        geometry_sweep_axis=geometry_sweep_axis,
        geometry_sweep_base=geometry_sweep_base,
    )


def _detect_folder_workflow(
    sparameter_path: Path,
    *,
    dataset_category: str | None,
    tables_only: bool,
) -> _FolderWorkflow:
    """Preserve category gates and CST format priority before marker analysis."""
    if dataset_category == "profile":
        _require_figures(tables_only)
        field3d_pairs = find_cst_field3d_pairs(sparameter_path)
        if field3d_pairs:
            return _FolderWorkflow("field3d", field3d_pairs=field3d_pairs)
        profile_inputs = find_cst_profile_inputs(sparameter_path)
        if not profile_inputs:
            raise ValueError(
                "Dataset category is profile, but no parseable CST profile txt export "
                f"was found: {sparameter_path}"
            )
        return _FolderWorkflow("profile", inputs=profile_inputs)

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
        _require_figures(tables_only)
        return _FolderWorkflow("dispersion", inputs=dispersion_inputs)

    if dataset_category is None:
        field3d_pairs = find_cst_field3d_pairs(sparameter_path)
        if field3d_pairs:
            _require_figures(tables_only)
            return _FolderWorkflow("field3d", field3d_pairs=field3d_pairs)
        profile_inputs = find_cst_profile_inputs(sparameter_path)
        if profile_inputs:
            _require_figures(tables_only)
            return _FolderWorkflow("profile", inputs=profile_inputs)
    return _FolderWorkflow("marker")


def _require_figures(tables_only: bool) -> None:
    if tables_only:
        raise ValueError("--tables-only is supported only for standard marker-analysis datasets")



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


def _dataset_category(path: Path) -> str | None:
    try:
        data_layer = detect_data_layer(path)
        return dataset_identity_from_path(path, data_layer).category
    except ValueError:
        return None
