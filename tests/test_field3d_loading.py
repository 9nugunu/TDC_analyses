from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deflector_tuning.data_loading.field3d import (
    field3d_kind_from_header,
    find_field3d_pairs,
    summarize_field3d_file,
)


def _write_field3d(
    path: Path,
    *,
    kind: str,
    x_values: tuple[float, ...] = (-1.0, 1.0),
    y_values: tuple[float, ...] = (-1.0, 1.0),
    z_values: tuple[float, ...] = (0.0, 1.0),
    vector: tuple[complex, complex, complex] = (1.0 + 0.0j, 0.0j, 0.0j),
) -> Path:
    prefix = "E" if kind == "e" else "H"
    units = "V/m" if kind == "e" else "A/m"
    header = (
        f"x [mm] y [mm] z [mm] "
        f"{prefix}xRe [{units}] {prefix}xIm [{units}] "
        f"{prefix}yRe [{units}] {prefix}yIm [{units}] "
        f"{prefix}zRe [{units}] {prefix}zIm [{units}]"
    )
    rows = [header, "-" * len(header)]
    for z_mm in z_values:
        for y_mm in y_values:
            for x_mm in x_values:
                rows.append(
                    " ".join(
                        str(value)
                        for value in (
                            x_mm,
                            y_mm,
                            z_mm,
                            vector[0].real,
                            vector[0].imag,
                            vector[1].real,
                            vector[1].imag,
                            vector[2].real,
                            vector[2].imag,
                        )
                    )
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_field3d_header_detection_and_pairing_use_content_and_generic_suffix(
    tmp_path: Path,
) -> None:
    default_e = _write_field3d(
        tmp_path / "e-field (f=2.857) [1].txt",
        kind="e",
    )
    default_h = _write_field3d(
        tmp_path / "h-field (f=2.857) [1].txt",
        kind="h",
    )
    depth_e = _write_field3d(
        tmp_path / "e-field (f=2.857) [1]_NumDepth0.5.txt",
        kind="e",
    )
    depth_h = _write_field3d(
        tmp_path / "h-field (f=2.857) [1]_NumDepth0.5.txt",
        kind="h",
    )
    (tmp_path / "notes.txt").write_text("not a CST field export", encoding="utf-8")

    assert field3d_kind_from_header(default_e) == "e"
    assert field3d_kind_from_header(default_h) == "h"
    pairs = find_field3d_pairs(tmp_path)

    assert [(pair.case_id, pair.e_path, pair.h_path) for pair in pairs] == [
        ("default", default_e, default_h),
        ("NumDepth0.5", depth_e, depth_h),
    ]


def test_summarize_field3d_file_integrates_complex_components_in_bounded_chunks(
    tmp_path: Path,
) -> None:
    path = _write_field3d(
        tmp_path / "e-field (f=2.857) [1].txt",
        kind="e",
        vector=(3.0 + 4.0j, 2.0 + 0.0j, 0.0j),
    )

    summary = summarize_field3d_file(path, chunk_rows=3)

    assert summary.field_kind == "e"
    assert summary.row_count == 8
    assert summary.grid_shape == (2, 2, 2)
    assert summary.spacing_mm == pytest.approx((2.0, 2.0, 1.0))
    assert summary.component_abs2_sums == pytest.approx((8 * 25.0, 8 * 4.0, 0.0))
    expected_energy = (
        0.25
        * 8.8541878128e-12
        * (8 * 29.0)
        * (2.0 * 2.0 * 1.0)
        * 1.0e-9
    )
    assert summary.region_energy_j == pytest.approx(expected_energy)
    assert np.asarray(summary.axis_abs2_by_z)[:, 0].tolist() == [0.0, 1.0]


def test_summarize_field3d_file_accepts_cst_coordinate_text_rounding(
    tmp_path: Path,
) -> None:
    path = _write_field3d(
        tmp_path / "e-field (f=2.857) [1].txt",
        kind="e",
        z_values=(
            98.549533,
            99.7158,
            100.88207,
            102.04833,
            103.2146,
            104.38087,
            105.54713,
        ),
    )

    summary = summarize_field3d_file(path)

    assert summary.grid_shape == (2, 2, 7)
    assert summary.spacing_mm[2] == pytest.approx(1.166267, abs=1.0e-5)
