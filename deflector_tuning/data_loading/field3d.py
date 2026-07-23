"""Header detection and bounded loading for CST 3D complex field exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np

EPSILON_0_F_PER_M = 8.8541878128e-12
MU_0_H_PER_M = 1.25663706212e-6
DEFAULT_CHUNK_ROWS = 100_000
FIELD_PREFIX_PATTERN = re.compile(r"^[eh]-field\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Field3DPair:
    """One matched pair of CST complex E- and H-field exports."""

    case_id: str
    e_path: Path
    h_path: Path


@dataclass(frozen=True)
class Field3DFileSummary:
    """Compact numerical summary of one rectangular CST field export."""

    path: Path
    field_kind: str
    row_count: int
    grid_shape: tuple[int, int, int]
    bounds_mm: tuple[float, float, float, float, float, float]
    spacing_mm: tuple[float, float, float]
    component_abs2_sums: tuple[float, float, float]
    component_energy_j: tuple[float, float, float]
    region_energy_j: float
    axis_abs2_by_z: tuple[tuple[float, float], ...]


def field3d_kind_from_header(path: str | Path) -> str | None:
    """Return ``e`` or ``h`` for a CST 3D complex-field header."""

    candidate = Path(path)
    if candidate.suffix.lower() != ".txt" or not candidate.is_file():
        return None
    try:
        with candidate.open("r", encoding="utf-8") as stream:
            header = stream.readline().strip().lower()
    except (OSError, UnicodeError):
        return None

    if not all(token in header for token in ("x [mm]", "y [mm]", "z [mm]")):
        return None
    if all(token in header for token in ("exre", "exim", "eyre", "eyim", "ezre", "ezim")):
        return "e"
    if all(token in header for token in ("hxre", "hxim", "hyre", "hyim", "hzre", "hzim")):
        return "h"
    return None


def find_field3d_pairs(path: str | Path) -> tuple[Field3DPair, ...]:
    """Find complete E/H pairs without reading their numeric payloads."""

    input_path = Path(path)
    candidates = [input_path] if input_path.is_file() else (
        sorted(input_path.glob("*.txt"), key=lambda item: item.name.lower())
        if input_path.is_dir()
        else []
    )
    grouped: dict[str, dict[str, Path]] = {}
    case_names: dict[str, str] = {}
    for candidate in candidates:
        kind = field3d_kind_from_header(candidate)
        if kind is None:
            continue
        pair_key, case_id = _pair_key_and_case(candidate, kind)
        fields = grouped.setdefault(pair_key, {})
        if kind in fields:
            raise ValueError(
                f"Duplicate {kind.upper()}-field export for pair {pair_key!r}: "
                f"{fields[kind]} and {candidate}"
            )
        fields[kind] = candidate
        case_names[pair_key] = case_id

    incomplete = {
        key: fields
        for key, fields in grouped.items()
        if set(fields) != {"e", "h"}
    }
    if incomplete:
        details = ", ".join(
            f"{key}: {sorted(fields)}" for key, fields in sorted(incomplete.items())
        )
        raise ValueError(f"Incomplete CST 3D E/H field pairs: {details}")

    return tuple(
        Field3DPair(
            case_id=case_names[key],
            e_path=grouped[key]["e"],
            h_path=grouped[key]["h"],
        )
        for key in sorted(
            grouped,
            key=lambda item: (
                case_names[item].lower() != "default",
                case_names[item].lower(),
            ),
        )
    )


def summarize_field3d_file(
    path: str | Path,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Field3DFileSummary:
    """Stream one field file into grid, energy, and axial compact statistics."""

    source = Path(path)
    field_kind = field3d_kind_from_header(source)
    if field_kind is None:
        raise ValueError(f"Not a supported CST 3D complex field export: {source}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    axes: tuple[set[float], set[float], set[float]] = (set(), set(), set())
    component_sums = np.zeros(3, dtype=float)
    axis_stats: dict[float, tuple[float, float, int]] = {}
    row_count = 0

    for rows in _iter_numeric_chunks(source, chunk_rows=chunk_rows):
        if rows.shape[1] != 9:
            raise ValueError(
                f"Expected nine numeric columns in {source}; got {rows.shape[1]}"
            )
        row_count += int(rows.shape[0])
        for axis_index in range(3):
            axes[axis_index].update(float(value) for value in np.unique(rows[:, axis_index]))

        abs2_components = np.column_stack(
            (
                rows[:, 3] ** 2 + rows[:, 4] ** 2,
                rows[:, 5] ** 2 + rows[:, 6] ** 2,
                rows[:, 7] ** 2 + rows[:, 8] ** 2,
            )
        )
        component_sums += abs2_components.sum(axis=0)
        total_abs2 = abs2_components.sum(axis=1)
        radius2 = rows[:, 0] ** 2 + rows[:, 1] ** 2
        for z_mm in np.unique(rows[:, 2]):
            z_mask = rows[:, 2] == z_mm
            local_radius2 = radius2[z_mask]
            local_min = float(local_radius2.min())
            center_mask = np.isclose(
                local_radius2,
                local_min,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            local_sum = float(total_abs2[z_mask][center_mask].sum())
            local_count = int(center_mask.sum())
            previous = axis_stats.get(float(z_mm))
            if previous is None or local_min < previous[0] - 1.0e-12:
                axis_stats[float(z_mm)] = (local_min, local_sum, local_count)
            elif np.isclose(local_min, previous[0], rtol=1.0e-10, atol=1.0e-12):
                axis_stats[float(z_mm)] = (
                    previous[0],
                    previous[1] + local_sum,
                    previous[2] + local_count,
                )

    if row_count == 0:
        raise ValueError(f"No numeric field rows found in {source}")

    axis_values = tuple(np.asarray(sorted(values), dtype=float) for values in axes)
    spacing_mm = tuple(_uniform_spacing(values, source) for values in axis_values)
    grid_shape = tuple(len(values) for values in axis_values)
    expected_rows = int(np.prod(grid_shape))
    if expected_rows != row_count:
        raise ValueError(
            f"Expected rectangular grid with {expected_rows} rows in {source}; "
            f"found {row_count}"
        )
    voxel_volume_m3 = float(np.prod(spacing_mm)) * 1.0e-9
    material_factor = (
        EPSILON_0_F_PER_M if field_kind == "e" else MU_0_H_PER_M
    )
    component_energy = 0.25 * material_factor * component_sums * voxel_volume_m3
    bounds_mm = (
        float(axis_values[0][0]),
        float(axis_values[0][-1]),
        float(axis_values[1][0]),
        float(axis_values[1][-1]),
        float(axis_values[2][0]),
        float(axis_values[2][-1]),
    )
    axis_abs2_by_z = tuple(
        (z_mm, values[1] / values[2])
        for z_mm, values in sorted(axis_stats.items())
    )
    return Field3DFileSummary(
        path=source,
        field_kind=field_kind,
        row_count=row_count,
        grid_shape=grid_shape,
        bounds_mm=bounds_mm,
        spacing_mm=spacing_mm,
        component_abs2_sums=tuple(float(value) for value in component_sums),
        component_energy_j=tuple(float(value) for value in component_energy),
        region_energy_j=float(component_energy.sum()),
        axis_abs2_by_z=axis_abs2_by_z,
    )


def iter_field3d_chunks(
    path: str | Path,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
):
    """Yield bounded numeric chunks from a validated CST field export."""

    source = Path(path)
    if field3d_kind_from_header(source) is None:
        raise ValueError(f"Not a supported CST 3D complex field export: {source}")
    yield from _iter_numeric_chunks(source, chunk_rows=chunk_rows)


def _iter_numeric_chunks(path: Path, *, chunk_rows: int):
    with path.open("r", encoding="utf-8") as stream:
        stream.readline()
        stream.readline()
        while True:
            lines = [line for line in islice(stream, chunk_rows) if line.strip()]
            if not lines:
                break
            values = np.fromstring("".join(lines), sep=" ")
            if values.size % 9 != 0:
                raise ValueError(
                    f"Expected nine numeric columns per row in {path}; "
                    f"parsed {values.size} values"
                )
            yield values.reshape((-1, 9))


def _pair_key_and_case(path: Path, kind: str) -> tuple[str, str]:
    stem = path.stem
    if FIELD_PREFIX_PATTERN.match(stem) is None:
        raise ValueError(
            f"3D {kind.upper()}-field filename must start with e-field or h-field: {path}"
        )
    common = FIELD_PREFIX_PATTERN.sub("", stem, count=1)
    suffix = common.rsplit("]", maxsplit=1)[1] if "]" in common else common
    case_id = suffix.strip(" _-") or "default"
    return common.lower(), case_id


def _uniform_spacing(values: np.ndarray, path: Path) -> float:
    if len(values) < 2:
        raise ValueError(f"3D field axis must contain at least two points: {path}")
    differences = np.diff(values)
    spacing = float(np.median(differences))
    if not np.allclose(differences, spacing, rtol=1.0e-5, atol=1.0e-9):
        raise ValueError(f"3D field grid is not uniformly spaced: {path}")
    return spacing
