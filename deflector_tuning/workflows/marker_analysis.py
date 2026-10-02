"""Build, save, and render the standard marker-analysis workflow."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, save_marker_analysis
from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection
from deflector_tuning.table_export import TableSaveResult
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.workflows.manifest import (
    cached_manifest_figures,
    cached_manifest_figure_group,
    write_manifest,
)
from deflector_tuning.workflows.marker_figures import load_s11_table_for_figures, render_marker_figures
from deflector_tuning.workflows.mode_detection import detect_analysis_modes, detection_report
from deflector_tuning.workflows.models import AnalysisPaths, RunResult

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


def run_marker_analysis(
    *,
    sparameter_path: Path,
    dispersion_path: Path,
    output_dir: Path,
    marker_role: str,
    dataset_category: str | None,
    plot_config: PlotConfig,
    file_workers: int = 1,
    plot_workers: int = 1,
    tables_only: bool = False,
    loader: DataLoader | None = None,
    marker_correction: TemperatureHumidityCorrection | None = None,
    geometry_sweep_axis: str | None = None,
    geometry_sweep_base: float | None = None,
) -> RunResult:
    """Execute marker analysis, retaining one folder load and existing figure caches."""
    table_dir = output_dir / "tables"
    figure_root = output_dir / "figures"
    manifest_path = output_dir / "manifest.json"
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
        cached_manifest_figure_group(manifest_path, "s11")
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
        sparameter_table = load_s11_table_for_figures(loader, sparameter_path)
    logger.info("Building marker analysis tables")
    tables = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        loader=loader,
        sparameter_table=sparameter_table,
        marker_correction=marker_correction,
        geometry_sweep_axis=geometry_sweep_axis,
        geometry_sweep_base=geometry_sweep_base,
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
        figures = cached_manifest_figures(manifest_path)
        detection = detection_report(tables, dataset_category=dataset_category)
        logger.info("Writing tables-only manifest to %s", manifest_path)
    else:
        figures, detection = render_marker_figures(
            tables=tables,
            modes=modes,
            dataset_category=dataset_category,
            loader=loader,
            sparameter_path=sparameter_path,
            sparameter_table=sparameter_table,
            figure_root=figure_root,
            plot_config=plot_config,
            plot_workers=plot_workers,
            cached_s11_figures=cached_s11_figures,
        )
        logger.info("Writing manifest to %s", manifest_path)
    write_manifest(
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
        geometry_sweep_axis=geometry_sweep_axis,
        geometry_sweep_base=geometry_sweep_base,
    )
    logger.info(
        "Tables-only analysis completed successfully"
        if tables_only else "Folder analysis completed successfully"
    )
    return RunResult(
        output_dir=output_dir,
        tables=table_paths,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )
