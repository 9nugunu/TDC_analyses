"""Extract a physical grid plane without changing complex CST field values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from deflector_tuning.data_loading.field3d import iter_field3d_chunks


@dataclass(frozen=True)
class FieldSection:
    """Complex vector field on one transverse-coordinate plane.

    ``values`` has axes (remaining transverse coordinate, z, component).
    The selected coordinate is an exported grid value; no interpolation is used.
    """

    fixed_axis: str
    transverse_axis: str
    requested_coordinate_mm: float
    actual_coordinate_mm: float
    transverse_mm: np.ndarray
    z_mm: np.ndarray
    values: np.ndarray
    row_count: int
    full_grid_shape: tuple[int, int, int]
    full_bounds_mm: tuple[tuple[float, float], ...]
    full_spacing_mm: tuple[float, float, float]
    full_component_abs2_sums: np.ndarray


def load_field_section(
    path: str | Path,
    *,
    fixed_axis: Literal["x", "y"] = "x",
    coordinate_mm: float = 0.0,
    chunk_rows: int = 100_000,
) -> FieldSection:
    """Stream a field export and retain its nearest complete grid plane.

    Equidistant planes use the lower coordinate. Full-export component
    squared sums are retained for independent checks against existing tables.
    """
    if fixed_axis not in ("x", "y") or not np.isfinite(coordinate_mm) or chunk_rows < 1:
        raise ValueError("Use a finite transverse coordinate and positive chunk size")
    axis = {"x": 0, "y": 1}[fixed_axis]
    transverse_axis = 1 - axis
    best: tuple[float, float] | None = None
    parts = []
    axes = [set(), set(), set()]
    full_sums = np.zeros(3)
    row_count = 0
    for rows in iter_field3d_chunks(path, chunk_rows=chunk_rows):
        if not np.isfinite(rows).all():
            raise ValueError("Field export contains nonfinite values")
        row_count += len(rows)
        for index in range(3):
            axes[index].update(np.unique(rows[:, index]).tolist())
        full_sums += (rows[:, 3::2] ** 2 + rows[:, 4::2] ** 2).sum(axis=0)
        local = min((abs(float(value) - coordinate_mm), float(value))
                    for value in np.unique(rows[:, axis]))
        if best is None or local < best:
            best = local
            parts = []
        selected = rows[rows[:, axis] == best[1]]
        if len(selected):
            parts.append(selected.copy())
    shape = tuple(len(values) for values in axes)
    if best is None or row_count != int(np.prod(shape)):
        raise ValueError("Field export is not a complete rectangular grid")
    rows = np.concatenate(parts)
    transverse = np.array(sorted(axes[transverse_axis]))
    z_mm = np.array(sorted(axes[2]))
    iy = np.searchsorted(transverse, rows[:, transverse_axis])
    iz = np.searchsorted(z_mm, rows[:, 2])
    flat = iy * len(z_mm) + iz
    if len(np.unique(flat)) != len(rows):
        raise ValueError("Selected plane contains duplicate coordinates")
    if len(rows) != len(transverse) * len(z_mm):
        raise ValueError("Selected plane is not a complete rectangular grid")
    values = np.empty((len(transverse), len(z_mm), 3), dtype=complex)
    values[iy, iz] = rows[:, 3::2] + 1j * rows[:, 4::2]
    return FieldSection(
        fixed_axis=fixed_axis,
        transverse_axis=("x", "y")[transverse_axis],
        requested_coordinate_mm=coordinate_mm,
        actual_coordinate_mm=best[1],
        transverse_mm=transverse,
        z_mm=z_mm,
        values=values,
        row_count=row_count,
        full_grid_shape=shape,
        full_bounds_mm=tuple((min(values), max(values)) for values in axes),
        full_spacing_mm=tuple(float(np.median(np.diff(sorted(values)))) for values in axes),
        full_component_abs2_sums=full_sums,
    )


def normalized_field_magnitude(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return complex-vector magnitude divided by its own nonzero maximum."""
    field = np.asarray(values, dtype=complex)
    if field.ndim != 3 or field.shape[-1] != 3 or not np.isfinite(field).all():
        raise ValueError("Expected finite complex vector-field grid")
    magnitude = np.linalg.norm(field, axis=-1)
    peak = float(magnitude.max(initial=0))
    if peak <= 0:
        raise ValueError("Cannot normalize an all-zero field section")
    return magnitude / peak, peak
