from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from deflector_tuning.dispersion import (
    load_cst_dispersion_txt,
    process_cst_dispersion_txt,
)
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves
from deflector_tuning.workflows.hooks import (
    DispersionPlotter,
    DispersionProcessor,
    ManifestWriter,
)
from deflector_tuning.workflows.manifest import write_manifest
from deflector_tuning.workflows.models import (
    AnalysisPaths,
    DetectionReport,
    FigurePaths,
    RunResult,
)


DispersionLoader = Callable[[str | Path], pd.DataFrame]


def find_cst_dispersion_inputs(
    path: str | Path,
    *,
    load_dispersion: DispersionLoader = load_cst_dispersion_txt,
) -> tuple[Path, ...]:
    input_path = Path(path)
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(
            input_path.glob("*.txt"), key=lambda item: item.name.lower()
        )
    else:
        return ()

    dispersion_paths: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.lower() != ".txt":
            continue
        try:
            load_dispersion(candidate)
        except (OSError, ValueError):
            continue
        dispersion_paths.append(candidate)
    return tuple(dispersion_paths)


def run_dispersion_analysis(
    dispersion_inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    load_dispersion: DispersionLoader = load_cst_dispersion_txt,
    process_dispersion: DispersionProcessor = process_cst_dispersion_txt,
    plot_dispersion: DispersionPlotter = plot_dispersion_curves,
    manifest_writer: ManifestWriter = write_manifest,
) -> RunResult:
    table_paths: AnalysisPaths = AnalysisPaths()
    figures: FigurePaths = OrderedDict()
    dispersion_figures: OrderedDict[str, Path] = OrderedDict()
    csv_dir = table_dir / "dispersion"
    figure_dir = figure_root / "dispersion"

    use_source_prefix = len(dispersion_inputs) > 1
    for input_path in dispersion_inputs:
        outputs = process_dispersion(input_path, output_dir=csv_dir)
        stem = input_path.stem
        table_paths[f"{stem}_long"] = outputs.long_csv
        table_paths[f"{stem}_wide"] = outputs.wide_csv
        table_paths[f"{stem}_summary"] = outputs.summary_csv

        dispersion_table = load_dispersion(input_path)
        figure_prefix = _dispersion_figure_prefix(
            stem, use_source_prefix=use_source_prefix
        )
        overview_key = _dispersion_figure_key(figure_prefix, "all_modes")
        dispersion_figures[overview_key] = plot_dispersion(
            dispersion_table,
            figure_dir / f"{overview_key}.png",
            title="All modes dispersion",
        )
        for mode_index in _dispersion_mode_indices(dispersion_table):
            mode_key = _dispersion_figure_key(figure_prefix, f"mode_{mode_index:02d}")
            dispersion_figures[mode_key] = plot_dispersion(
                dispersion_table,
                figure_dir / f"{mode_key}.png",
                mode_indices=(mode_index,),
                title=f"Mode {mode_index:02d} dispersion",
            )

    figures["dispersion"] = dispersion_figures
    modes = ("dispersion",)
    detection: DetectionReport = {
        "grid_scan_spacing": {
            "enabled": False,
            "reason": "CST dispersion data bypasses S-parameter marker analysis",
        },
        "dispersion": {
            "enabled": True,
            "reason": "parseable CST phase-vs-frequency dispersion text export detected",
            "input_count": len(dispersion_inputs),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_writer(
        manifest_path,
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role=marker_role,
        modes=modes,
        detection=detection,
        tables=table_paths,
        figures=figures,
    )
    return RunResult(output_dir, table_paths, figures, modes, manifest_path)


def _dispersion_mode_indices(dispersion_table: pd.DataFrame) -> tuple[int, ...]:
    modes = (
        pd.to_numeric(dispersion_table["mode_index"], errors="coerce").dropna().unique()
    )
    return tuple(sorted(int(mode) for mode in modes))


def _dispersion_figure_prefix(stem: str, *, use_source_prefix: bool) -> str:
    if not use_source_prefix:
        return ""
    return stem.lower().replace("-", "_")


def _dispersion_figure_key(prefix: str, label: str) -> str:
    if not prefix:
        return label
    return f"{prefix}_{label}"
