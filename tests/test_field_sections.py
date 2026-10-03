"""Numerical contracts for sections of complex CST fields."""

import numpy as np
import pytest

from deflector_tuning.analysis.field_sections import (
    load_field_section,
    normalized_field_magnitude,
)


def write_field(path, rows):
    header = "x [mm] y [mm] z [mm] ExRe [V/m] ExIm [V/m] EyRe [V/m] EyIm [V/m] EzRe [V/m] EzIm [V/m]"
    np.savetxt(path, np.asarray(rows), header=header + "\n---", comments="")


def sample_rows():
    return np.array([[x, y, z, 0, x, y, 1, z, 0]
                     for z in (0, 3) for y in (-2, 0, 2) for x in (-1, 1)], float)


def test_nearest_section_preserves_complex_components_and_grid(tmp_path):
    path = tmp_path / "field.txt"
    rows = sample_rows()
    np.random.default_rng(12).shuffle(rows)
    write_field(path, rows)
    section = load_field_section(path, coordinate_mm=0, chunk_rows=3)
    assert section.actual_coordinate_mm == -1
    assert section.full_grid_shape == (2, 3, 2)
    assert section.row_count == 12
    np.testing.assert_array_equal(section.transverse_mm, [-2, 0, 2])
    np.testing.assert_array_equal(section.z_mm, [0, 3])
    assert section.values.shape == (3, 2, 3)
    np.testing.assert_array_equal(section.values[0, 1], [-1j, -2 + 1j, 3])
    expected = (rows[:, 3::2] ** 2 + rows[:, 4::2] ** 2).sum(axis=0)
    np.testing.assert_allclose(section.full_component_abs2_sums, expected)


def test_y_section_uses_x_as_transverse_coordinate(tmp_path):
    path = tmp_path / "field.txt"
    write_field(path, sample_rows())
    section = load_field_section(path, fixed_axis="y", coordinate_mm=0.2, chunk_rows=2)
    assert section.actual_coordinate_mm == 0
    assert section.transverse_axis == "x"
    assert section.values.shape == (2, 2, 3)


def test_normalized_magnitude_is_global_phase_and_scale_invariant():
    values = np.array([[[3+4j, 0, 0], [0, 0, 0]], [[1j, 2, 2j], [1, 1j, 1]]])
    result, peak = normalized_field_magnitude(values)
    transformed, transformed_peak = normalized_field_magnitude(values * 7 * np.exp(0.63j))
    assert peak == 5
    assert transformed_peak == pytest.approx(35)
    np.testing.assert_allclose(result, transformed, atol=1e-14)
    assert result[0, 1] == 0
    assert result.max() == 1


def test_duplicate_section_coordinates_are_rejected(tmp_path):
    path = tmp_path / "field.txt"
    rows = sample_rows()
    rows[-2] = rows[0]
    write_field(path, rows)
    with pytest.raises(ValueError, match="duplicate|rectangular"):
        load_field_section(path)


def test_incomplete_grid_is_rejected(tmp_path):
    path = tmp_path / "field.txt"
    write_field(path, sample_rows()[:-1])
    with pytest.raises(ValueError, match="rectangular"):
        load_field_section(path)


@pytest.mark.parametrize("bad", [np.zeros((2, 2, 3)), np.full((2, 2, 3), np.nan)])
def test_invalid_normalization_is_rejected(bad):
    with pytest.raises(ValueError):
        normalized_field_magnitude(bad)
