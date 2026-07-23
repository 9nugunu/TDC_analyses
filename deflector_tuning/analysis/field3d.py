"""Geometry inference and local Slater terms for CST 3D field exports."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.field3d import (
    EPSILON_0_F_PER_M,
    MU_0_H_PER_M,
    Field3DFileSummary,
    Field3DPair,
    iter_field3d_chunks,
)


@dataclass(frozen=True)
class Field3DCaseSummary:
    """Matched E/H compact summaries for one exported geometry case."""

    pair: Field3DPair
    e: Field3DFileSummary
    h: Field3DFileSummary
    tip_z_mm: float | None


def combine_field3d_summaries(
    pair: Field3DPair,
    e_summary: Field3DFileSummary,
    h_summary: Field3DFileSummary,
) -> Field3DCaseSummary:
    """Validate and combine one matched E/H field pair."""

    if e_summary.field_kind != "e" or h_summary.field_kind != "h":
        raise ValueError("Expected E summary followed by H summary")
    if e_summary.grid_shape != h_summary.grid_shape:
        raise ValueError(f"E/H grid shape mismatch for case {pair.case_id}")
    if not np.allclose(e_summary.bounds_mm, h_summary.bounds_mm):
        raise ValueError(f"E/H bounds mismatch for case {pair.case_id}")
    if not np.allclose(e_summary.spacing_mm, h_summary.spacing_mm):
        raise ValueError(f"E/H spacing mismatch for case {pair.case_id}")
    return Field3DCaseSummary(
        pair=pair,
        e=e_summary,
        h=h_summary,
        tip_z_mm=detect_plunger_tip_mm(e_summary, h_summary),
    )


def detect_plunger_tip_mm(
    e_summary: Field3DFileSummary,
    h_summary: Field3DFileSummary,
) -> float | None:
    """Return the first plane of a sustained axial zero-field region."""

    e_axis = np.asarray(e_summary.axis_abs2_by_z, dtype=float)
    h_axis = np.asarray(h_summary.axis_abs2_by_z, dtype=float)
    if e_axis.shape != h_axis.shape or not np.allclose(e_axis[:, 0], h_axis[:, 0]):
        raise ValueError("E/H axial sample locations do not match")
    combined = e_axis[:, 1] + h_axis[:, 1]
    scale = float(combined.max(initial=0.0))
    if scale == 0.0:
        return None
    zero = combined <= scale * 1.0e-24
    minimum_run = max(2, int(np.ceil(len(zero) * 0.2)))
    for index in range(0, len(zero) - minimum_run + 1):
        if zero[index] and bool(np.all(zero[index:])):
            return float(e_axis[index, 0])
    return None


def infer_plunger_radius_mm(case: Field3DCaseSummary) -> float:
    """Infer the equivalent radius of the axis-connected PEC zero region."""

    if case.tip_z_mm is None:
        raise ValueError(f"Case {case.pair.case_id} has no detected plunger tip")
    z_values = np.asarray([item[0] for item in case.e.axis_abs2_by_z], dtype=float)
    tip_index = int(np.argmin(np.abs(z_values - case.tip_z_mm)))
    probe_index = min(tip_index + 1, len(z_values) - 1)
    probe_z_mm = float(z_values[probe_index])
    e_plane = _load_field_planes(case.pair.e_path, (probe_z_mm,))[probe_z_mm]
    h_plane = _load_field_planes(case.pair.h_path, (probe_z_mm,))[probe_z_mm]
    _validate_plane_coordinates(e_plane, h_plane)

    e_abs2 = _component_abs2(e_plane).sum(axis=1)
    h_abs2 = _component_abs2(h_plane).sum(axis=1)
    combined = e_abs2 + h_abs2
    scale = float(combined.max(initial=0.0))
    zero = combined <= max(scale * 1.0e-24, np.finfo(float).tiny)
    x_values = np.unique(e_plane[:, 0])
    y_values = np.unique(e_plane[:, 1])
    zero_grid = np.zeros((len(y_values), len(x_values)), dtype=bool)
    x_index = np.searchsorted(x_values, e_plane[:, 0])
    y_index = np.searchsorted(y_values, e_plane[:, 1])
    zero_grid[y_index, x_index] = zero

    zero_rows = np.flatnonzero(zero)
    if zero_rows.size == 0:
        raise ValueError(f"No PEC zero region found for case {case.pair.case_id}")
    radius2 = e_plane[zero_rows, 0] ** 2 + e_plane[zero_rows, 1] ** 2
    seed_row = int(zero_rows[int(np.argmin(radius2))])
    seed = (int(y_index[seed_row]), int(x_index[seed_row]))
    if not zero_grid[seed]:
        raise ValueError(f"Axis-adjacent field is not zero for case {case.pair.case_id}")
    component_count = _connected_zero_count(zero_grid, seed)
    dx_mm, dy_mm = case.e.spacing_mm[:2]
    return float(np.sqrt(component_count * dx_mm * dy_mm / np.pi))


def compute_slater_position_table(
    no_plunger: Field3DCaseSummary,
    plunger_cases: tuple[Field3DCaseSummary, ...],
    *,
    radius_mm: float,
) -> pd.DataFrame:
    """Integrate local E/H energy terms at detected plunger-tip positions."""

    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be positive")
    if no_plunger.tip_z_mm is not None:
        raise ValueError("The Slater reference case must not contain a plunger")
    z_reference = np.asarray(
        [item[0] for item in no_plunger.e.axis_abs2_by_z],
        dtype=float,
    )
    grouped: dict[float, list[Field3DCaseSummary]] = {}
    for case in plunger_cases:
        if case.tip_z_mm is None:
            continue
        ref_z_mm = float(z_reference[int(np.argmin(np.abs(z_reference - case.tip_z_mm)))])
        grouped.setdefault(ref_z_mm, []).append(case)
    if not grouped:
        raise ValueError("No plunger-tip positions were detected")

    target_z = tuple(sorted(grouped))
    e_planes = _load_field_planes(no_plunger.pair.e_path, target_z)
    h_planes = _load_field_planes(no_plunger.pair.h_path, target_z)
    dx_mm, dy_mm, dz_mm = no_plunger.e.spacing_mm
    voxel_volume_m3 = dx_mm * dy_mm * dz_mm * 1.0e-9
    total_reference_energy = (
        no_plunger.e.region_energy_j + no_plunger.h.region_energy_j
    )
    if total_reference_energy <= 0.0:
        raise ValueError("No-plunger export-region energy must be positive")

    rows: list[dict[str, object]] = []
    for position_index, ref_z_mm in enumerate(target_z, start=1):
        e_plane = e_planes[ref_z_mm]
        h_plane = h_planes[ref_z_mm]
        disk, e_component_j, h_component_j = _integrate_disk_energy(
            e_plane,
            h_plane,
            radius_mm=radius_mm,
            voxel_volume_m3=voxel_volume_m3,
            z_mm=ref_z_mm,
        )
        e_j = float(e_component_j.sum())
        h_j = float(h_component_j.sum())
        k_j = e_j - h_j
        cases = grouped[ref_z_mm]
        inserted_case = _select_inserted_case(cases)
        inserted_sample_z_mm, inserted_step_mm = _vacuum_side_sample(
            inserted_case
        )
        inserted_e_plane = _load_field_planes(
            inserted_case.pair.e_path,
            (inserted_sample_z_mm,),
        )[inserted_sample_z_mm]
        inserted_h_plane = _load_field_planes(
            inserted_case.pair.h_path,
            (inserted_sample_z_mm,),
        )[inserted_sample_z_mm]
        inserted_dx_mm, inserted_dy_mm, _ = inserted_case.e.spacing_mm
        inserted_voxel_volume_m3 = (
            inserted_dx_mm
            * inserted_dy_mm
            * inserted_step_mm
            * 1.0e-9
        )
        (
            inserted_disk,
            inserted_e_component_j,
            inserted_h_component_j,
        ) = _integrate_disk_energy(
            inserted_e_plane,
            inserted_h_plane,
            radius_mm=radius_mm,
            voxel_volume_m3=inserted_voxel_volume_m3,
            z_mm=inserted_sample_z_mm,
        )
        inserted_e_j = float(inserted_e_component_j.sum())
        inserted_h_j = float(inserted_h_component_j.sum())
        inserted_k_j = inserted_e_j - inserted_h_j
        inserted_total_energy = (
            inserted_case.e.region_energy_j + inserted_case.h.region_energy_j
        )
        if inserted_total_energy <= 0.0:
            raise ValueError(
                f"Inserted-case export-region energy must be positive: "
                f"{inserted_case.pair.case_id}"
            )
        baseline_k_over_u = k_j / total_reference_energy
        inserted_k_over_u = inserted_k_j / inserted_total_energy
        rows.append(
            {
                "position_index": position_index,
                "case_ids": ";".join(case.pair.case_id for case in cases),
                "tip_z_mm": float(np.mean([case.tip_z_mm for case in cases])),
                "ref_z_mm": ref_z_mm,
                "radius_mm": radius_mm,
                "voxel_count": int(disk.sum()),
                "e_j": e_j,
                "h_j": h_j,
                "k_e_minus_h_j": k_j,
                "k_over_u": baseline_k_over_u,
                "abs_k_over_u": abs(k_j) / total_reference_energy,
                "ex_pct": _percentage(e_component_j, 0),
                "ey_pct": _percentage(e_component_j, 1),
                "ez_pct": _percentage(e_component_j, 2),
                "hx_pct": _percentage(h_component_j, 0),
                "hy_pct": _percentage(h_component_j, 1),
                "hz_pct": _percentage(h_component_j, 2),
                "inserted_case_id": inserted_case.pair.case_id,
                "inserted_sample_z_mm": inserted_sample_z_mm,
                "inserted_step_mm": inserted_step_mm,
                "inserted_voxel_count": int(inserted_disk.sum()),
                "inserted_e_j": inserted_e_j,
                "inserted_h_j": inserted_h_j,
                "inserted_k_e_minus_h_j": inserted_k_j,
                "inserted_k_over_u": inserted_k_over_u,
                "inserted_abs_k_over_u": abs(inserted_k_j)
                / inserted_total_energy,
                "inserted_to_baseline_k_ratio": (
                    inserted_k_over_u / baseline_k_over_u
                    if baseline_k_over_u != 0.0
                    else np.nan
                ),
                "inserted_field_basis": (
                    "inserted_case_fixed_frequency_vacuum_side"
                ),
            }
        )
    table = pd.DataFrame(rows)
    max_abs_k = float(table["k_e_minus_h_j"].abs().max())
    table["k_rel"] = (
        table["k_e_minus_h_j"] / max_abs_k if max_abs_k > 0.0 else 0.0
    )
    return table


def compute_slater_volume_table(
    no_plunger: Field3DCaseSummary,
    plunger_cases: tuple[Field3DCaseSummary, ...],
    *,
    radius_mm: float,
    first_tip_z_mm: float = 14.574,
    cell_body_mm: float = 29.148,
    iris_thickness_mm: float = 5.84,
) -> pd.DataFrame:
    """Integrate NoPlunger E/H energy over each full inserted cylinder.

    ``NumDepth=0.5`` starts at ``first_tip_z_mm``. Each subsequent half-step
    advances the tip by ``(cell_body_mm + iris_thickness_mm) / 2``.
    """

    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be positive")
    if cell_body_mm <= 0.0 or iris_thickness_mm < 0.0:
        raise ValueError("The cell-body and iris dimensions must be valid")
    if no_plunger.tip_z_mm is not None:
        raise ValueError("The Slater reference case must not contain a plunger")
    z_reference = np.asarray(
        [item[0] for item in no_plunger.e.axis_abs2_by_z],
        dtype=float,
    )
    grouped: dict[float, list[Field3DCaseSummary]] = {}
    for case in plunger_cases:
        if case.tip_z_mm is None:
            continue
        grouped.setdefault(_num_depth([case]), []).append(case)
    if not grouped:
        raise ValueError("No plunger-tip positions were detected")

    num_depths = tuple(sorted(grouped))
    period_mm = cell_body_mm + iris_thickness_mm
    nominal_tip_z = tuple(
        first_tip_z_mm + (num_depth - 0.5) * period_mm
        for num_depth in num_depths
    )
    integration_start_z: list[float] = []
    for tip_z_mm in nominal_tip_z:
        candidates = z_reference[z_reference >= tip_z_mm - 1.0e-8]
        if candidates.size == 0:
            raise ValueError(
                f"Plunger tip z={tip_z_mm:g} mm lies beyond the "
                "NoPlunger field export"
            )
        integration_start_z.append(float(candidates[0]))

    e_component_sums = np.zeros((len(num_depths), 3), dtype=float)
    h_component_sums = np.zeros((len(num_depths), 3), dtype=float)
    voxel_counts = np.zeros(len(num_depths), dtype=np.int64)
    radius2_limit = radius_mm**2

    e_chunks = iter_field3d_chunks(no_plunger.pair.e_path)
    h_chunks = iter_field3d_chunks(no_plunger.pair.h_path)
    for e_chunk, h_chunk in zip(e_chunks, h_chunks, strict=True):
        _validate_plane_coordinates(e_chunk, h_chunk)
        radius2 = e_chunk[:, 0] ** 2 + e_chunk[:, 1] ** 2
        e_abs2 = _component_abs2(e_chunk)
        h_abs2 = _component_abs2(h_chunk)
        for index, tip_z_mm in enumerate(nominal_tip_z):
            cylinder = (radius2 <= radius2_limit) & (
                e_chunk[:, 2] >= tip_z_mm - 1.0e-8
            )
            voxel_counts[index] += int(cylinder.sum())
            e_component_sums[index] += e_abs2[cylinder].sum(axis=0)
            h_component_sums[index] += h_abs2[cylinder].sum(axis=0)

    dx_mm, dy_mm, dz_mm = no_plunger.e.spacing_mm
    voxel_volume_m3 = dx_mm * dy_mm * dz_mm * 1.0e-9
    e_component_j = (
        0.25
        * EPSILON_0_F_PER_M
        * e_component_sums
        * voxel_volume_m3
    )
    h_component_j = (
        0.25
        * MU_0_H_PER_M
        * h_component_sums
        * voxel_volume_m3
    )
    total_reference_energy = (
        no_plunger.e.region_energy_j + no_plunger.h.region_energy_j
    )
    if total_reference_energy <= 0.0:
        raise ValueError("No-plunger export-region energy must be positive")

    rows: list[dict[str, object]] = []
    z_end_mm = float(no_plunger.e.bounds_mm[5])
    for index, num_depth in enumerate(num_depths):
        e_j = float(e_component_j[index].sum())
        h_j = float(h_component_j[index].sum())
        k_j = e_j - h_j
        cases = grouped[num_depth]
        detected_tips = [
            float(case.tip_z_mm)
            for case in cases
            if case.tip_z_mm is not None
        ]
        rows.append(
            {
                "position_index": index + 1,
                "case_ids": ";".join(case.pair.case_id for case in cases),
                "num_depth": num_depth,
                "tip_z_mm": nominal_tip_z[index],
                "integration_start_z_mm": integration_start_z[index],
                "detected_tip_z_mm": float(np.mean(detected_tips)),
                "z_end_mm": z_end_mm,
                "radius_mm": radius_mm,
                "voxel_count": int(voxel_counts[index]),
                "e_j": e_j,
                "h_j": h_j,
                "k_e_minus_h_j": k_j,
                "abs_k_over_u": abs(k_j) / total_reference_energy,
                "ex_pct": _percentage(e_component_j[index], 0),
                "ey_pct": _percentage(e_component_j[index], 1),
                "ez_pct": _percentage(e_component_j[index], 2),
                "hx_pct": _percentage(h_component_j[index], 0),
                "hy_pct": _percentage(h_component_j[index], 1),
                "hz_pct": _percentage(h_component_j[index], 2),
                "integration_basis": (
                    "NoPlunger field over nominal-tip-to-zmax cylinder"
                ),
            }
        )
    table = pd.DataFrame(rows)
    max_abs_k = float(table["k_e_minus_h_j"].abs().max())
    table["k_rel"] = (
        table["k_e_minus_h_j"] / max_abs_k if max_abs_k > 0.0 else 0.0
    )
    return table


def _num_depth(cases: list[Field3DCaseSummary]) -> float:
    """Return the unique NumDepth value represented by grouped cases."""

    values = {
        float(match.group(1))
        for case in cases
        if (
            match := re.search(
                r"NumDepth\s*([0-9]+(?:\.[0-9]+)?)",
                case.pair.case_id,
                re.IGNORECASE,
            )
        )
        is not None
    }
    if len(values) > 1:
        raise ValueError(f"Conflicting NumDepth values at one tip: {sorted(values)}")
    return next(iter(values)) if values else np.nan


def _select_inserted_case(
    cases: list[Field3DCaseSummary],
) -> Field3DCaseSummary:
    """Select the most complete deterministic case at a duplicate tip."""

    return max(
        cases,
        key=lambda case: (
            min(case.e.row_count, case.h.row_count),
            min(case.e.grid_shape[2], case.h.grid_shape[2]),
            case.pair.case_id.lower().startswith("numdepth"),
            case.pair.case_id.lower(),
        ),
    )


def _vacuum_side_sample(case: Field3DCaseSummary) -> tuple[float, float]:
    """Return the plane and slab depth immediately before an inserted tip."""

    if case.tip_z_mm is None:
        raise ValueError(f"Case {case.pair.case_id} has no detected plunger tip")
    z_values = np.asarray(
        [item[0] for item in case.e.axis_abs2_by_z],
        dtype=float,
    )
    tip_indices = np.flatnonzero(
        np.isclose(z_values, case.tip_z_mm, rtol=0.0, atol=1.0e-8)
    )
    if tip_indices.size != 1:
        raise ValueError(
            f"Detected tip plane is not unique for case {case.pair.case_id}"
        )
    tip_index = int(tip_indices[0])
    if tip_index == 0:
        raise ValueError(
            f"No vacuum plane before plunger tip for case {case.pair.case_id}"
        )
    sample_z_mm = float(z_values[tip_index - 1])
    step_mm = float(case.tip_z_mm - sample_z_mm)
    if step_mm <= 0.0:
        raise ValueError(
            f"Vacuum-side grid step must be positive for case {case.pair.case_id}"
        )
    return sample_z_mm, step_mm


def _integrate_disk_energy(
    e_plane: np.ndarray,
    h_plane: np.ndarray,
    *,
    radius_mm: float,
    voxel_volume_m3: float,
    z_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate E and H component energies over one circular slab."""

    _validate_plane_coordinates(e_plane, h_plane)
    disk = e_plane[:, 0] ** 2 + e_plane[:, 1] ** 2 <= radius_mm**2
    if not np.any(disk):
        raise ValueError(
            f"No grid points fall inside radius {radius_mm:g} mm at z={z_mm:g} mm"
        )
    e_component_j = (
        0.25
        * EPSILON_0_F_PER_M
        * _component_abs2(e_plane[disk]).sum(axis=0)
        * voxel_volume_m3
    )
    h_component_j = (
        0.25
        * MU_0_H_PER_M
        * _component_abs2(h_plane[disk]).sum(axis=0)
        * voxel_volume_m3
    )
    return disk, e_component_j, h_component_j


def _load_field_planes(
    path,
    target_z_mm: tuple[float, ...],
) -> dict[float, np.ndarray]:
    collected: dict[float, list[np.ndarray]] = {value: [] for value in target_z_mm}
    for chunk in iter_field3d_chunks(path):
        for value in target_z_mm:
            selected = chunk[np.isclose(chunk[:, 2], value, rtol=0.0, atol=1.0e-8)]
            if selected.size:
                collected[value].append(selected)
    missing = [value for value, blocks in collected.items() if not blocks]
    if missing:
        raise ValueError(f"Missing requested z planes {missing} in {path}")
    return {
        value: np.concatenate(blocks, axis=0)
        for value, blocks in collected.items()
    }


def _component_abs2(rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            rows[:, 3] ** 2 + rows[:, 4] ** 2,
            rows[:, 5] ** 2 + rows[:, 6] ** 2,
            rows[:, 7] ** 2 + rows[:, 8] ** 2,
        )
    )


def _validate_plane_coordinates(e_plane: np.ndarray, h_plane: np.ndarray) -> None:
    if e_plane.shape != h_plane.shape or not np.allclose(
        e_plane[:, :3],
        h_plane[:, :3],
    ):
        raise ValueError("E/H plane coordinates do not match")


def _connected_zero_count(
    zero_grid: np.ndarray,
    seed: tuple[int, int],
) -> int:
    queue = deque([seed])
    visited = {seed}
    while queue:
        y_index, x_index = queue.popleft()
        for neighbor in (
            (y_index - 1, x_index),
            (y_index + 1, x_index),
            (y_index, x_index - 1),
            (y_index, x_index + 1),
        ):
            y_neighbor, x_neighbor = neighbor
            if (
                0 <= y_neighbor < zero_grid.shape[0]
                and 0 <= x_neighbor < zero_grid.shape[1]
                and zero_grid[neighbor]
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited)


def _percentage(values: np.ndarray, index: int) -> float:
    total = float(values.sum())
    return float(values[index] / total * 100.0) if total > 0.0 else 0.0
