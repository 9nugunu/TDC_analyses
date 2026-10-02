"""Analyze direct CST one-port admittance and impedance exports."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from deflector_tuning.data_loading.one_port_matrix import (
    Lane,
    extract_one_port_marker_frequencies,
    load_y11_touchstone_folder,
    load_z11_touchstone_folder,
    sample_y11_markers,
    sample_z11_markers,
)
from deflector_tuning.table_export import save_table_contract
from deflector_tuning.visualization.admittance_sweep_plots import plot_z11_marker_sweep
from deflector_tuning.visualization.raw_y11_plots import (
    plot_y11_raw_complex_sweep,
    plot_y11_raw_frequency_with_markers,
    plot_y11_raw_grid,
    plot_y11_raw_polar_views,
)
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.workflows.manifest import cached_manifest_figures, write_manifest
from deflector_tuning.workflows.models import AnalysisPaths, FigurePaths, RunResult

logger = logging.getLogger(__name__)


def run_one_port_matrix_analysis(
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
        figures = cached_manifest_figures(manifest_path)
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
    write_manifest(
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
