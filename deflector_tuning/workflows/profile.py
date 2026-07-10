from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.field_energy_ratio import (
    compute_cell_iris_field_energy_ratios,
    summarize_cell_iris_field_energy_ratios,
)
from deflector_tuning.visualization.em_field_structure_plots import (
    FieldProfileExport,
    load_field_profile_export,
    plot_field_profile_with_tdc_structure,
)
from deflector_tuning.workflows.hooks import ManifestWriter, ProfilePlotter
from deflector_tuning.workflows.manifest import write_manifest
from deflector_tuning.workflows.models import (
    AnalysisPaths,
    DetectionReport,
    FigurePaths,
    RunResult,
)


ProfileLoader = Callable[[str | Path], FieldProfileExport]
FieldRatioCalculator = Callable[[FieldProfileExport], pd.DataFrame]
FieldRatioSummarizer = Callable[[pd.DataFrame], pd.DataFrame]


def find_cst_profile_inputs(
    path: str | Path,
    *,
    load_profile: ProfileLoader = load_field_profile_export,
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

    profile_paths: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.lower() != ".txt":
            continue
        try:
            load_profile(candidate)
        except (OSError, ValueError):
            continue
        profile_paths.append(candidate)
    return tuple(profile_paths)


def run_profile_analysis(
    profile_inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    load_profile: ProfileLoader = load_field_profile_export,
    plot_profile: ProfilePlotter = plot_field_profile_with_tdc_structure,
    compute_field_ratios: FieldRatioCalculator = compute_cell_iris_field_energy_ratios,
    summarize_field_ratios: FieldRatioSummarizer = summarize_cell_iris_field_energy_ratios,
    manifest_writer: ManifestWriter = write_manifest,
) -> RunResult:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = figure_root / "profile"
    table_paths: AnalysisPaths = AnalysisPaths()
    figures: FigurePaths = OrderedDict()
    profile_figures: OrderedDict[str, Path] = OrderedDict()
    summary_rows: list[dict[str, str | int | float]] = []
    field_energy_ratio_tables: list[pd.DataFrame] = []
    field_energy_summary_tables: list[pd.DataFrame] = []

    for input_path in profile_inputs:
        export = load_profile(input_path)
        summary_rows.extend(_profile_summary_rows(input_path, export))
        field_ratios = _field_energy_ratio_table(
            input_path,
            export,
            compute_field_ratios=compute_field_ratios,
        )
        if not field_ratios.empty:
            field_energy_ratio_tables.append(field_ratios)
            field_summary = summarize_field_ratios(field_ratios)
            field_summary.insert(0, "source_file", input_path.name)
            field_energy_summary_tables.append(field_summary)
        profile_figures[input_path.stem] = plot_profile(
            export,
            figure_dir / f"{input_path.stem}.png",
        )
        for figure_key, split_export in _split_phase_profile_export_by_field_kind(
            input_path, export
        ):
            profile_figures[figure_key] = plot_profile(
                split_export,
                figure_dir / f"{figure_key}.png",
            )

    summary_path = table_dir / "profile_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    table_paths["profile_summary"] = summary_path
    if field_energy_ratio_tables:
        field_pairs_path = table_dir / "field_energy_ratio_pairs.csv"
        field_summary_path = table_dir / "field_energy_ratio_summary.csv"
        pd.concat(field_energy_ratio_tables, ignore_index=True).to_csv(
            field_pairs_path, index=False
        )
        pd.concat(field_energy_summary_tables, ignore_index=True).to_csv(
            field_summary_path, index=False
        )
        table_paths["field_energy_ratio_pairs"] = field_pairs_path
        table_paths["field_energy_ratio_summary"] = field_summary_path
    figures["profile"] = profile_figures
    modes = ("profile",)
    detection: DetectionReport = {
        "profile": {
            "enabled": True,
            "reason": "parseable CST quantity-versus-z profile text export detected",
            "input_count": len(profile_inputs),
        },
        "grid_scan_spacing": {
            "enabled": False,
            "reason": "profile data bypasses S-parameter marker analysis",
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


def _profile_summary_rows(
    input_path: Path,
    export: FieldProfileExport,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
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


def _field_energy_ratio_table(
    input_path: Path,
    export: FieldProfileExport,
    *,
    compute_field_ratios: FieldRatioCalculator,
) -> pd.DataFrame:
    try:
        table = compute_field_ratios(export)
    except ValueError:
        return pd.DataFrame()
    table = table.copy()
    table.insert(0, "source_file", input_path.name)
    return table


def _split_phase_profile_export_by_field_kind(
    input_path: Path,
    export: FieldProfileExport,
) -> tuple[tuple[str, FieldProfileExport], ...]:
    if not export.traces or any(trace.value_kind != "phase" for trace in export.traces):
        return ()
    field_kinds = list(dict.fromkeys(trace.field_kind for trace in export.traces))
    if len(field_kinds) < 2:
        return ()

    outputs: list[tuple[str, FieldProfileExport]] = []
    for field_kind in field_kinds:
        traces = tuple(
            trace for trace in export.traces if trace.field_kind == field_kind
        )
        if traces:
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
