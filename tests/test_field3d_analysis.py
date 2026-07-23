from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deflector_tuning.analysis import field3d as field3d_analysis
from deflector_tuning.analysis.field3d import (
    Field3DCaseSummary,
    combine_field3d_summaries,
    compute_slater_position_table,
    detect_plunger_tip_mm,
    infer_plunger_radius_mm,
)
from deflector_tuning.data_loading.field3d import (
    EPSILON_0_F_PER_M,
    Field3DPair,
    summarize_field3d_file,
)


def _write_field(
    path: Path,
    *,
    kind: str,
    plunger_tip_mm: float | None,
    plunger_radius_mm: float = 1.1,
    amplitude: float = 1.0,
    amplitude_by_z: dict[int, float] | None = None,
    z_values: tuple[int, ...] = tuple(range(5)),
) -> Path:
    prefix = kind.upper()
    component = "E" if kind == "e" else "H"
    units = "V/m" if kind == "e" else "A/m"
    header = (
        f"x [mm] y [mm] z [mm] "
        f"{component}xRe [{units}] {component}xIm [{units}] "
        f"{component}yRe [{units}] {component}yIm [{units}] "
        f"{component}zRe [{units}] {component}zIm [{units}]"
    )
    rows = [header, "-" * len(header)]
    for z_mm in z_values:
        for y_mm in range(-2, 3):
            for x_mm in range(-2, 3):
                in_plunger = (
                    plunger_tip_mm is not None
                    and z_mm >= plunger_tip_mm
                    and x_mm**2 + y_mm**2 <= plunger_radius_mm**2
                )
                plane_amplitude = (
                    amplitude_by_z.get(z_mm, amplitude)
                    if amplitude_by_z is not None
                    else amplitude
                )
                value = 0.0 if in_plunger else plane_amplitude
                if prefix == "H" and amplitude == 0.0:
                    value = 0.0
                rows.append(
                    f"{x_mm} {y_mm} {z_mm} {value} 0 0 0 0 0"
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _pair(
    folder: Path,
    *,
    case_id: str,
    plunger_tip_mm: float | None,
    e_amplitude: float = 1.0,
    h_amplitude: float = 1.0,
    e_amplitude_by_z: dict[int, float] | None = None,
    h_amplitude_by_z: dict[int, float] | None = None,
    z_values: tuple[int, ...] = tuple(range(5)),
) -> Field3DPair:
    suffix = "" if case_id == "default" else f"_{case_id}"
    return Field3DPair(
        case_id=case_id,
        e_path=_write_field(
            folder / f"e-field [1]{suffix}.txt",
            kind="e",
            plunger_tip_mm=plunger_tip_mm,
            amplitude=e_amplitude,
            amplitude_by_z=e_amplitude_by_z,
            z_values=z_values,
        ),
        h_path=_write_field(
            folder / f"h-field [1]{suffix}.txt",
            kind="h",
            plunger_tip_mm=plunger_tip_mm,
            amplitude=h_amplitude,
            amplitude_by_z=h_amplitude_by_z,
            z_values=z_values,
        ),
    )


def test_detects_plunger_tip_and_infers_axis_connected_zero_radius(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, case_id="NumDepth0.5", plunger_tip_mm=2.0)
    e_summary = summarize_field3d_file(pair.e_path, chunk_rows=17)
    h_summary = summarize_field3d_file(pair.h_path, chunk_rows=19)
    case = combine_field3d_summaries(pair, e_summary, h_summary)

    assert detect_plunger_tip_mm(e_summary, h_summary) == pytest.approx(2.0)
    radius = infer_plunger_radius_mm(case)

    assert radius == pytest.approx(np.sqrt(5.0 / np.pi))


def test_slater_table_integrates_no_plunger_disk_and_normalizes_to_export_energy(
    tmp_path: Path,
) -> None:
    no_pair = _pair(
        tmp_path,
        case_id="NoPlunger",
        plunger_tip_mm=None,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    plunger_pair = _pair(
        tmp_path,
        case_id="NumDepth0.5",
        plunger_tip_mm=2.0,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    no_case = combine_field3d_summaries(
        no_pair,
        summarize_field3d_file(no_pair.e_path),
        summarize_field3d_file(no_pair.h_path),
    )
    plunger_case = combine_field3d_summaries(
        plunger_pair,
        summarize_field3d_file(plunger_pair.e_path),
        summarize_field3d_file(plunger_pair.h_path),
    )

    table = compute_slater_position_table(
        no_case,
        (plunger_case,),
        radius_mm=1.1,
    )

    assert table["case_ids"].tolist() == ["NumDepth0.5"]
    assert table["tip_z_mm"].tolist() == pytest.approx([2.0])
    assert table["ref_z_mm"].tolist() == pytest.approx([2.0])
    expected_e_j = 0.25 * EPSILON_0_F_PER_M * (5 * 4.0) * 1.0e-9
    assert table.loc[0, "e_j"] == pytest.approx(expected_e_j)
    assert table.loc[0, "h_j"] == pytest.approx(0.0)
    assert table.loc[0, "k_e_minus_h_j"] == pytest.approx(expected_e_j)
    expected_total = no_case.e.region_energy_j + no_case.h.region_energy_j
    assert table.loc[0, "abs_k_over_u"] == pytest.approx(
        expected_e_j / expected_total
    )


def test_slater_table_adds_inserted_state_vacuum_side_term(
    tmp_path: Path,
) -> None:
    no_pair = _pair(
        tmp_path,
        case_id="NoPlunger",
        plunger_tip_mm=None,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    inserted_pair = _pair(
        tmp_path,
        case_id="NumDepth0.5",
        plunger_tip_mm=2.0,
        e_amplitude=1.0,
        e_amplitude_by_z={1: 3.0},
        h_amplitude=0.0,
    )
    no_case = combine_field3d_summaries(
        no_pair,
        summarize_field3d_file(no_pair.e_path),
        summarize_field3d_file(no_pair.h_path),
    )
    inserted_case = combine_field3d_summaries(
        inserted_pair,
        summarize_field3d_file(inserted_pair.e_path),
        summarize_field3d_file(inserted_pair.h_path),
    )

    table = compute_slater_position_table(
        no_case,
        (inserted_case,),
        radius_mm=1.1,
    )

    baseline_e_j = 0.25 * EPSILON_0_F_PER_M * (5 * 4.0) * 1.0e-9
    inserted_e_j = 0.25 * EPSILON_0_F_PER_M * (5 * 9.0) * 1.0e-9
    inserted_total_j = (
        inserted_case.e.region_energy_j + inserted_case.h.region_energy_j
    )
    assert table.loc[0, "e_j"] == pytest.approx(baseline_e_j)
    assert table.loc[0, "inserted_case_id"] == "NumDepth0.5"
    assert table.loc[0, "inserted_sample_z_mm"] == pytest.approx(1.0)
    assert table.loc[0, "inserted_step_mm"] == pytest.approx(1.0)
    assert table.loc[0, "inserted_voxel_count"] == 5
    assert table.loc[0, "inserted_e_j"] == pytest.approx(inserted_e_j)
    assert table.loc[0, "inserted_h_j"] == pytest.approx(0.0)
    assert table.loc[0, "inserted_k_e_minus_h_j"] == pytest.approx(inserted_e_j)
    baseline_total_j = no_case.e.region_energy_j + no_case.h.region_energy_j
    baseline_k_over_u = baseline_e_j / baseline_total_j
    inserted_k_over_u = inserted_e_j / inserted_total_j
    assert table.loc[0, "k_over_u"] == pytest.approx(baseline_k_over_u)
    assert table.loc[0, "inserted_k_over_u"] == pytest.approx(
        inserted_k_over_u
    )
    assert table.loc[0, "inserted_abs_k_over_u"] == pytest.approx(
        inserted_e_j / inserted_total_j
    )
    assert table.loc[0, "inserted_to_baseline_k_ratio"] == pytest.approx(
        inserted_k_over_u / baseline_k_over_u
    )
    assert table.loc[0, "inserted_field_basis"] == (
        "inserted_case_fixed_frequency_vacuum_side"
    )


def test_slater_volume_table_integrates_full_inserted_cylinder(
    tmp_path: Path,
) -> None:
    no_pair = _pair(
        tmp_path,
        case_id="NoPlunger",
        plunger_tip_mm=None,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    inserted_pair = _pair(
        tmp_path,
        case_id="NumDepth0.5",
        plunger_tip_mm=2.0,
        e_amplitude=1.0,
        h_amplitude=0.0,
    )

    def summarize(pair: Field3DPair) -> Field3DCaseSummary:
        return combine_field3d_summaries(
            pair,
            summarize_field3d_file(pair.e_path),
            summarize_field3d_file(pair.h_path),
        )

    table = field3d_analysis.compute_slater_volume_table(
        summarize(no_pair),
        (summarize(inserted_pair),),
        radius_mm=1.1,
        first_tip_z_mm=2.0,
        cell_body_mm=2.0,
        iris_thickness_mm=0.0,
    )

    expected_e_j = (
        0.25
        * EPSILON_0_F_PER_M
        * (3 * 5 * 4.0)
        * 1.0e-9
    )
    assert table["case_ids"].tolist() == ["NumDepth0.5"]
    assert table.loc[0, "num_depth"] == pytest.approx(0.5)
    assert table.loc[0, "tip_z_mm"] == pytest.approx(2.0)
    assert table.loc[0, "integration_start_z_mm"] == pytest.approx(2.0)
    assert table.loc[0, "z_end_mm"] == pytest.approx(4.0)
    assert table.loc[0, "voxel_count"] == 15
    assert table.loc[0, "e_j"] == pytest.approx(expected_e_j)
    assert table.loc[0, "h_j"] == pytest.approx(0.0)
    assert table.loc[0, "k_e_minus_h_j"] == pytest.approx(expected_e_j)
    assert table.loc[0, "integration_basis"] == (
        "NoPlunger field over nominal-tip-to-zmax cylinder"
    )


def test_slater_table_prefers_full_named_case_at_duplicate_tip(
    tmp_path: Path,
) -> None:
    no_pair = _pair(
        tmp_path,
        case_id="NoPlunger",
        plunger_tip_mm=None,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    cropped_pair = _pair(
        tmp_path,
        case_id="default",
        plunger_tip_mm=2.0,
        e_amplitude=1.0,
        e_amplitude_by_z={1: 7.0},
        h_amplitude=0.0,
        z_values=(0, 1, 2, 3),
    )
    full_pair = _pair(
        tmp_path,
        case_id="NumDepth0.5",
        plunger_tip_mm=2.0,
        e_amplitude=1.0,
        e_amplitude_by_z={1: 3.0},
        h_amplitude=0.0,
    )

    def summarize(pair: Field3DPair):
        return combine_field3d_summaries(
            pair,
            summarize_field3d_file(pair.e_path),
            summarize_field3d_file(pair.h_path),
        )

    table = compute_slater_position_table(
        summarize(no_pair),
        (summarize(cropped_pair), summarize(full_pair)),
        radius_mm=1.1,
    )

    expected_full_e_j = (
        0.25 * EPSILON_0_F_PER_M * (5 * 9.0) * 1.0e-9
    )
    assert table.loc[0, "case_ids"] == "default;NumDepth0.5"
    assert table.loc[0, "inserted_case_id"] == "NumDepth0.5"
    assert table.loc[0, "inserted_e_j"] == pytest.approx(expected_full_e_j)


def test_slater_table_rejects_tip_without_preceding_vacuum_plane(
    tmp_path: Path,
) -> None:
    no_pair = _pair(
        tmp_path,
        case_id="NoPlunger",
        plunger_tip_mm=None,
        e_amplitude=2.0,
        h_amplitude=0.0,
    )
    inserted_pair = _pair(
        tmp_path,
        case_id="NumDepth0.5",
        plunger_tip_mm=0.0,
        e_amplitude=1.0,
        h_amplitude=0.0,
    )

    def summarize(pair: Field3DPair):
        return combine_field3d_summaries(
            pair,
            summarize_field3d_file(pair.e_path),
            summarize_field3d_file(pair.h_path),
        )

    inserted_summary = summarize(inserted_pair)
    inserted_at_first_plane = Field3DCaseSummary(
        pair=inserted_summary.pair,
        e=inserted_summary.e,
        h=inserted_summary.h,
        tip_z_mm=0.0,
    )

    with pytest.raises(ValueError, match="No vacuum plane before plunger tip"):
        compute_slater_position_table(
            summarize(no_pair),
            (inserted_at_first_plane,),
            radius_mm=1.1,
        )
