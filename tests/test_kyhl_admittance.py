import math

import pandas as pd
import pytest

from deflector_tuning.analysis.kyhl_admittance import (
    compute_kyhl_admittance_points,
    compute_kyhl_admittance_transitions,
    nearest_operation_axis,
    operation_mode_axes,
    operation_mode_scale,
)


def test_operation_mode_axes_for_two_pi_over_three_are_60_180_300() -> None:
    assert operation_mode_axes(120.0) == (60.0, 180.0, 300.0)
    assert operation_mode_scale(120.0) == pytest.approx(math.sqrt(3.0))


def test_nearest_operation_axis_uses_wrapped_error() -> None:
    assert nearest_operation_axis(355.0, (60.0, 180.0, 300.0)) == 300.0
    assert nearest_operation_axis(2.0, (60.0, 180.0, 300.0)) == 60.0


def test_compute_kyhl_admittance_transitions_matches_2pi3_branch_axis() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_1.s1p",
                "tune_position": 0.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_real": 1.0,
                "s_imag": 0.0,
                "s_db": 0.0,
                "s_phase_deg": 0.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_2.s1p",
                "tune_position": 1.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_real": 0.2,
                "s_imag": -0.4,
                "s_db": -6.9897,
                "s_phase_deg": -63.4349,
            },
        ]
    )

    result = compute_kyhl_admittance_transitions(marker_points, operation_mode_deg=120.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["position_family"] == "cell"
    assert row["operation_axes_deg"] == "60;180;300"
    assert row["operation_mode_scale"] == pytest.approx(math.sqrt(3.0))
    assert row["delta_raw_pseudo_admittance_real"] == pytest.approx(0.5)
    assert row["delta_raw_pseudo_admittance_imag"] == pytest.approx(0.5)
    assert row["delta_operation_scaled_admittance_real"] == pytest.approx(0.5)
    assert row["delta_operation_scaled_admittance_imag"] == pytest.approx(math.sqrt(3.0) / 2.0)
    assert row["kyhl_operation_angle_deg"] == pytest.approx(60.0)
    assert row["operation_scaled_admittance_angle_deg"] == pytest.approx(60.0)
    assert row["nearest_operation_axis_deg"] == pytest.approx(60.0)
    assert row["operation_axis_error_deg"] == pytest.approx(0.0)


def test_compute_kyhl_admittance_points_reports_nearest_branch_axis() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_1.s1p",
                "tune_position": 0.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_real": 0.2,
                "s_imag": -0.4,
                "s_db": -6.9897,
                "s_phase_deg": -63.4349,
            },
        ]
    )

    result = compute_kyhl_admittance_points(marker_points, operation_mode_deg=120.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["operation_axes_deg"] == "60;180;300"
    assert row["raw_pseudo_admittance_real"] == pytest.approx(0.5)
    assert row["raw_pseudo_admittance_imag"] == pytest.approx(0.5)
    assert row["operation_scaled_admittance_real"] == pytest.approx(0.5)
    assert row["operation_scaled_admittance_imag"] == pytest.approx(math.sqrt(3.0) / 2.0)
    assert row["kyhl_operation_angle_deg"] == pytest.approx(60.0)
    assert row["operation_scaled_admittance_angle_deg"] == pytest.approx(60.0)
    assert row["nearest_operation_axis_deg"] == pytest.approx(60.0)
