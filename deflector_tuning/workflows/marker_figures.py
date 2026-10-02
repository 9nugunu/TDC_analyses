"""Render the standard marker-analysis figure groups in their contract order."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.grid_scan_spacing import summarize_marker_spacing_for_grid_scan
from deflector_tuning.analysis.sparameter_selection import select_s11_rows
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.visualization.grid_scan_phase_line_plots import plot_grid_scan_sparameter_phase_r_c_line_scan
from deflector_tuning.visualization.grid_scan_spacing_maps import plot_grid_scan_spacing_error_maps
from deflector_tuning.visualization.geometry_phase_response_plots import plot_geometry_phase_response
from deflector_tuning.visualization.kyhl_admittance_plots import plot_f2pi3_normalized_admittance_view
from deflector_tuning.visualization.nodal_shift_plots import plot_nodal_shift
from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.position_phase_bar_plots import (
    build_position_phase_bar_table,
    plot_position_phase_advance_bars,
    plot_position_phase_bars,
)
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.cell_iris_response_plots import plot_cell_iris_response_comparison
from deflector_tuning.visualization.coupler_cavity_parameter_plots import plot_coupler_cavity_parameters
from deflector_tuning.workflows.mode_detection import detection_report
from deflector_tuning.workflows.models import DetectionReport, FigurePaths

POSITION_PHASE_BAR_POSITIONS = (1.0, 2.0)
NO_PORT_EXTENSION_SOURCE_PATTERN = r"(?:^|[_-])noporte(?:[_\-.]|$)"
logger = logging.getLogger(__name__)


def render_marker_figures(
    *,
    tables: dict[str, pd.DataFrame],
    modes: tuple[str, ...],
    dataset_category: str | None,
    loader: DataLoader,
    sparameter_path: Path,
    sparameter_table: pd.DataFrame | None,
    figure_root: Path,
    plot_config: PlotConfig,
    plot_workers: int,
    cached_s11_figures: OrderedDict[str, Path],
) -> tuple[FigurePaths, DetectionReport]:
    """Render enabled figures and return the table-based detection report."""
    figures: FigurePaths = OrderedDict()
    if "grid_scan_spacing" in modes and cached_s11_figures:
        logger.info(
            "Skipping S11 figures because cached grid-scan S11 figures already exist"
        )
        figures["s11"] = cached_s11_figures
    else:
        if sparameter_table is None:
            sparameter_table = load_s11_table_for_figures(loader, sparameter_path)
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
        phase_bar_inputs = _select_position_phase_bar_inputs(tables["marker_pts"])
        if phase_bar_inputs is not None:
            logger.info("Rendering position phase bar figures")
            phase_bar_dir = figure_root / "phase_bar"
            figures["phase_bar"] = OrderedDict(
                position_phase=plot_position_phase_bars(
                    phase_bar_inputs,
                    phase_bar_dir / "position_1p0_vs_2p0_phase_bars.png",
                    positions=POSITION_PHASE_BAR_POSITIONS,
                    config=plot_config,
                ),
                phase_advance=plot_position_phase_advance_bars(
                    phase_bar_inputs,
                    phase_bar_dir / "position_1p0_to_2p0_phase_advance_bars.png",
                    positions=POSITION_PHASE_BAR_POSITIONS,
                    config=plot_config,
                ),
            )
        else:
            logger.info(
                "Skipping position phase bar figures because marker_points does not "
                "contain one row for every standard marker at positions 1.0 and 2.0"
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

    detection = detection_report(tables, dataset_category=dataset_category)
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

    return figures, detection


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and not table.empty

def _select_position_phase_bar_inputs(
    marker_points: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return unambiguous 1.0-to-2.0 phase-bar rows when available.

    When a raw folder contains matched ``*_portE`` and ``*_noportE`` traces at
    the same tuning position, retain the port-extension trace for this
    port-extension phase comparison. A no-port-only dataset remains eligible.
    """

    try:
        build_position_phase_bar_table(
            marker_points,
            positions=POSITION_PHASE_BAR_POSITIONS,
        )
    except ValueError:
        if "source_file" not in marker_points:
            return None
        without_no_port_extension = marker_points.loc[
            ~marker_points["source_file"].astype(str).str.contains(
                NO_PORT_EXTENSION_SOURCE_PATTERN,
                case=False,
                regex=True,
            )
        ].copy()
        try:
            build_position_phase_bar_table(
                without_no_port_extension,
                positions=POSITION_PHASE_BAR_POSITIONS,
            )
        except ValueError:
            return None
        return without_no_port_extension
    return marker_points

def load_s11_table_for_figures(
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
