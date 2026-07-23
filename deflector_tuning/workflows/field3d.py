"""Reusable one-folder workflow for CST 3D complex E/H field exports."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.analysis.field3d import (
    Field3DCaseSummary,
    combine_field3d_summaries,
    compute_slater_position_table,
    compute_slater_volume_table,
    infer_plunger_radius_mm,
)
from deflector_tuning.data_loading.field3d import (
    Field3DFileSummary,
    Field3DPair,
    summarize_field3d_file,
)
from deflector_tuning.visualization.field3d_plots import (
    plot_field3d_components,
    plot_slater_positions,
    plot_slater_volumes,
)
from deflector_tuning.workflows.manifest import write_manifest
from deflector_tuning.workflows.models import (
    AnalysisPaths,
    FigurePaths,
    RunResult,
)

FIELD3D_SCHEMA_VERSION = 5
logger = logging.getLogger(__name__)
FieldSummarizer = Callable[[str | Path], Field3DFileSummary]


def run_field3d_analysis(
    pairs: tuple[Field3DPair, ...],
    *,
    output_dir: Path,
    table_dir: Path,
    figure_root: Path,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    summarize_file: FieldSummarizer = summarize_field3d_file,
    plunger_radius_mm: float = 8.0,
    first_tip_z_mm: float = 14.574,
    cell_body_mm: float = 29.148,
    iris_thickness_mm: float = 5.84,
) -> RunResult:
    """Analyze matched 3D E/H exports or reuse unchanged saved products."""

    if not pairs:
        raise ValueError("No matched CST 3D E/H field pairs were provided")
    output_dir = Path(output_dir)
    table_dir = Path(table_dir)
    figure_dir = Path(figure_root) / "field3d"
    manifest_path = output_dir / "manifest.json"
    signatures = _source_signatures(pairs)
    cached = load_cached_field3d_result(
        manifest_path,
        source_signatures=signatures,
    )
    if cached is not None:
        logger.info("Reusing unchanged 3D field analysis: %s", output_dir)
        return cached

    logger.info("Summarizing %d CST 3D E/H field pairs", len(pairs))
    cases: list[Field3DCaseSummary] = []
    for pair in pairs:
        logger.info("Summarizing 3D field case %s", pair.case_id)
        cases.append(
            combine_field3d_summaries(
                pair,
                summarize_file(pair.e_path),
                summarize_file(pair.h_path),
            )
        )
    reference = _reference_case(cases)
    plunger_cases = tuple(case for case in cases if case.tip_z_mm is not None)
    if not plunger_cases:
        raise ValueError("No plunger geometry was detected in the 3D field pairs")
    radius_source = min(
        plunger_cases,
        key=lambda case: case.pair.e_path.stat().st_size
        + case.pair.h_path.stat().st_size,
    )
    detected_radius_mm = infer_plunger_radius_mm(radius_source)
    radius_mm = plunger_radius_mm
    slater_table = compute_slater_position_table(
        reference,
        plunger_cases,
        radius_mm=radius_mm,
    )
    slater_volume_table = compute_slater_volume_table(
        reference,
        plunger_cases,
        radius_mm=radius_mm,
        first_tip_z_mm=first_tip_z_mm,
        cell_body_mm=cell_body_mm,
        iris_thickness_mm=iris_thickness_mm,
    )
    source_table = _source_table(cases)
    energy_table = _energy_table(cases)

    table_dir.mkdir(parents=True, exist_ok=True)
    table_paths: AnalysisPaths = OrderedDict()
    for key, table in (
        ("field3d_src", source_table),
        ("field3d_energy", energy_table),
        ("slater_pos", slater_table),
        ("slater_vol", slater_volume_table),
    ):
        path = table_dir / f"{key}.csv"
        table.to_csv(path, index=False)
        table_paths[key] = path

    figures: FigurePaths = OrderedDict()
    figures["field3d"] = OrderedDict(
        (
            (
                "field_comp",
                plot_field3d_components(
                    energy_table,
                    figure_dir / "field_comp.png",
                ),
            ),
            (
                "slater_pos",
                plot_slater_positions(
                    slater_table,
                    figure_dir / "slater_pos.png",
                ),
            ),
            (
                "slater_vol",
                plot_slater_volumes(
                    slater_volume_table,
                    figure_dir / "slater_vol.png",
                ),
            ),
        )
    )
    modes = ("field3d",)
    detection = {
        "field3d": {
            "enabled": True,
            "reason": "paired CST 3D complex E/H field headers detected",
            "input_count": len(pairs) * 2,
        },
        "profile": {
            "enabled": False,
            "reason": "3D field data bypasses the 1D profile loader",
        },
        "grid_scan_spacing": {
            "enabled": False,
            "reason": "3D field data bypasses S-parameter marker analysis",
        },
    }
    write_manifest(
        manifest_path,
        sparameter_path=Path(sparameter_path),
        dispersion_path=Path(dispersion_path),
        marker_role=marker_role,
        modes=modes,
        detection=detection,
        tables=table_paths,
        figures=figures,
    )
    _register_field3d_metadata(
        manifest_path,
        source_signatures=signatures,
        radius_mm=radius_mm,
        radius_source_case=radius_source.pair.case_id,
        detected_radius_mm=detected_radius_mm,
        reference_case=reference.pair.case_id,
        first_tip_z_mm=first_tip_z_mm,
        cell_body_mm=cell_body_mm,
        iris_thickness_mm=iris_thickness_mm,
    )
    return RunResult(
        output_dir=output_dir,
        tables=table_paths,
        figures=figures,
        analysis_modes=modes,
        manifest_path=manifest_path,
    )


def load_cached_field3d_result(
    manifest_path: Path,
    *,
    source_signatures: list[dict[str, int | str]],
) -> RunResult | None:
    """Return a complete unchanged field3d result declared by a manifest."""

    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = manifest.get("field3d", {})
    if (
        metadata.get("schema_version") != FIELD3D_SCHEMA_VERSION
        or metadata.get("sources") != source_signatures
        or manifest.get("analysis_modes") != ["field3d"]
    ):
        return None
    outputs = manifest.get("outputs", {})
    raw_tables = outputs.get("tables", {})
    raw_figures = outputs.get("figures", {})
    required_tables = (
        "field3d_src",
        "field3d_energy",
        "slater_pos",
        "slater_vol",
    )
    if set(raw_tables) != set(required_tables) or "field3d" not in raw_figures:
        return None
    tables: AnalysisPaths = OrderedDict()
    for key in required_tables:
        path = _existing_output_path(raw_tables[key])
        if path is None:
            return None
        tables[key] = path
    figures: FigurePaths = OrderedDict()
    figures["field3d"] = OrderedDict()
    for key in ("field_comp", "slater_pos", "slater_vol"):
        raw_path = raw_figures["field3d"].get(key)
        if raw_path is None:
            return None
        path = _existing_output_path(raw_path)
        if path is None:
            return None
        figures["field3d"][key] = path
    return RunResult(
        output_dir=manifest_path.parent,
        tables=tables,
        figures=figures,
        analysis_modes=("field3d",),
        manifest_path=manifest_path,
    )


def _source_table(cases: list[Field3DCaseSummary]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        for summary in (case.e, case.h):
            stat = summary.path.stat()
            rows.append(
                {
                    "case_id": case.pair.case_id,
                    "field_kind": summary.field_kind,
                    "source_file": summary.path.name,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "row_count": summary.row_count,
                    "nx": summary.grid_shape[0],
                    "ny": summary.grid_shape[1],
                    "nz": summary.grid_shape[2],
                    "x_min_mm": summary.bounds_mm[0],
                    "x_max_mm": summary.bounds_mm[1],
                    "y_min_mm": summary.bounds_mm[2],
                    "y_max_mm": summary.bounds_mm[3],
                    "z_min_mm": summary.bounds_mm[4],
                    "z_max_mm": summary.bounds_mm[5],
                    "dx_mm": summary.spacing_mm[0],
                    "dy_mm": summary.spacing_mm[1],
                    "dz_mm": summary.spacing_mm[2],
                    "tip_z_mm": case.tip_z_mm,
                }
            )
    return pd.DataFrame(rows)


def _energy_table(cases: list[Field3DCaseSummary]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        e_component = np.asarray(case.e.component_energy_j)
        h_component = np.asarray(case.h.component_energy_j)
        rows.append(
            {
                "case_id": case.pair.case_id,
                "tip_z_mm": case.tip_z_mm,
                "e_region_j": case.e.region_energy_j,
                "h_region_j": case.h.region_energy_j,
                "e_over_h": (
                    case.e.region_energy_j / case.h.region_energy_j
                    if case.h.region_energy_j > 0.0
                    else np.nan
                ),
                "ex_pct": _component_percentage(e_component, 0),
                "ey_pct": _component_percentage(e_component, 1),
                "ez_pct": _component_percentage(e_component, 2),
                "hx_pct": _component_percentage(h_component, 0),
                "hy_pct": _component_percentage(h_component, 1),
                "hz_pct": _component_percentage(h_component, 2),
            }
        )
    return pd.DataFrame(rows)


def _reference_case(cases: list[Field3DCaseSummary]) -> Field3DCaseSummary:
    for case in cases:
        if case.pair.case_id.lower() == "noplunger":
            return case
    candidates = [case for case in cases if case.tip_z_mm is None]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Exactly one no-plunger reference case is required")


def _component_percentage(values: np.ndarray, index: int) -> float:
    total = float(values.sum())
    return float(values[index] / total * 100.0) if total > 0.0 else 0.0


def _source_signatures(
    pairs: tuple[Field3DPair, ...],
) -> list[dict[str, int | str]]:
    paths = sorted(
        (path for pair in pairs for path in (pair.e_path, pair.h_path)),
        key=lambda path: path.name.lower(),
    )
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def _register_field3d_metadata(
    manifest_path: Path,
    *,
    source_signatures: list[dict[str, int | str]],
    radius_mm: float,
    radius_source_case: str,
    detected_radius_mm: float,
    reference_case: str,
    first_tip_z_mm: float,
    cell_body_mm: float,
    iris_thickness_mm: float,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["field3d"] = {
        "schema_version": FIELD3D_SCHEMA_VERSION,
        "sources": source_signatures,
        "radius_mm": radius_mm,
        "radius_basis": "configured_physical_radius",
        "detected_radius_mm": detected_radius_mm,
        "radius_source_case": radius_source_case,
        "reference_case": reference_case,
        "plunger_axis": {
            "num_depth_start": 0.5,
            "first_tip_z_mm": first_tip_z_mm,
            "cell_body_mm": cell_body_mm,
            "iris_thickness_mm": iris_thickness_mm,
            "tip_formula": (
                "first_tip_z_mm + (NumDepth - 0.5) "
                "* (cell_body_mm + iris_thickness_mm)"
            ),
            "direction": "tip_to_z_max",
        },
        "slater_quantity": "K_E_minus_H",
        "local_comparison_quantity": "signed_K_over_U",
        "slater_quantities": {
            "baseline": {
                "quantity": "K_E_minus_H",
                "field_basis": reference_case,
                "sample": "equal_volume_at_detected_tip",
                "normalization": f"{reference_case}_export_region_energy",
            },
            "inserted_state": {
                "quantity": "K_E_minus_H",
                "field_basis": "per_position_inserted_case",
                "sample": "vacuum_plane_before_detected_tip",
                "normalization": "same_inserted_case_export_region_energy",
                "frequency_basis": "fixed_frequency_complex_monitor",
            },
            "cumulative_volume": {
                "quantity": "K_E_minus_H",
                "field_basis": reference_case,
                "sample": "radius_limited_cylinder_from_nominal_tip_to_z_max",
                "normalization": f"{reference_case}_export_region_energy",
            },
        },
        "absolute_frequency_shift_claimed": False,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_output_path(raw_path: str) -> Path | None:
    path = Path(raw_path)
    return path if path.is_file() else None
