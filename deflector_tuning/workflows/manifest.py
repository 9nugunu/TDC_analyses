from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from deflector_tuning.workflows.models import (
    AnalysisPaths,
    DetectionReport,
    FigurePaths,
)


def write_manifest(
    path: Path,
    *,
    sparameter_path: Path,
    dispersion_path: Path,
    marker_role: str,
    modes: tuple[str, ...],
    detection: DetectionReport,
    tables: AnalysisPaths,
    figures: FigurePaths,
) -> None:
    manifest = {
        "sparameter_path": str(sparameter_path),
        "dispersion_path": str(dispersion_path),
        "marker_role": marker_role,
        "analysis_modes": list(modes),
        "detected_modes": detection,
        "outputs": {
            "tables": {name: str(table_path) for name, table_path in tables.items()},
            "figures": {
                group: {name: str(figure_path) for name, figure_path in paths.items()}
                for group, paths in figures.items()
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def cached_manifest_figure_group(
    manifest_path: Path, group: str
) -> OrderedDict[str, Path]:
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
