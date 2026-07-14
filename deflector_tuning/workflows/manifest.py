from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from deflector_tuning.workflows.models import (
    AnalysisPaths,
    DetectionReport,
    FigurePaths,
)

FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg"})


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
    table_contract: str | None = None,
    table_schema_version: int | None = None,
    table_constants: dict[str, dict[str, object]] | None = None,
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
    if table_contract is not None:
        manifest["table_contract"] = table_contract
    if table_schema_version is not None:
        manifest["table_schema_version"] = table_schema_version
        manifest["table_constants"] = table_constants or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
        path = _resolve_cached_path(Path(raw_path), manifest_path)
        if path is None:
            return OrderedDict()
        cached[str(name)] = path
    return cached


def cached_manifest_figures(manifest_path: Path) -> FigurePaths:
    """Return only figure entries whose files still exist."""

    if not manifest_path.exists():
        return OrderedDict()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return OrderedDict()
    groups = manifest.get("outputs", {}).get("figures", {})
    if not isinstance(groups, dict):
        return OrderedDict()

    cached: FigurePaths = OrderedDict()
    for group, raw_paths in groups.items():
        if not isinstance(raw_paths, dict):
            continue
        existing: OrderedDict[str, Path] = OrderedDict()
        for name, raw_path in raw_paths.items():
            path = _resolve_cached_path(Path(raw_path), manifest_path)
            if path is not None:
                existing[str(name)] = path
        if existing:
            cached[str(group)] = existing
    if cached:
        return cached
    return _discover_existing_figures(manifest_path.parent / "figures")


def _resolve_cached_path(path: Path, manifest_path: Path) -> Path | None:
    if path.is_absolute():
        return path if path.is_file() else None
    if path.is_file():
        return path
    for ancestor in manifest_path.parents:
        candidate = ancestor / path
        if candidate.is_file():
            return candidate
    return None


def _discover_existing_figures(figure_root: Path) -> FigurePaths:
    if not figure_root.is_dir():
        return OrderedDict()
    discovered: FigurePaths = OrderedDict()
    for path in sorted(figure_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        relative = path.relative_to(figure_root)
        if len(relative.parts) < 2:
            continue
        group = relative.parts[0]
        key = "__".join(Path(*relative.parts[1:]).with_suffix("").parts)
        group_paths = discovered.setdefault(group, OrderedDict())
        if key in group_paths:
            key = f"{key}__{path.suffix.lower().lstrip('.')}"
        group_paths[key] = path
    return discovered
