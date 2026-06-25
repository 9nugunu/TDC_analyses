import pandas as pd
import pytest

from deflector_tuning.analysis.cell_iris_response import compare_cell_and_iris_responses


def test_compare_cell_and_iris_responses_pairs_matching_transition_indices() -> None:
    transitions = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "port_side": pd.NA,
                "s_name": "S11",
                "position_family": "cell",
                "from_source_file": "cell_0p5.s1p",
                "to_source_file": "cell_1p5.s1p",
                "from_tune_position": 0.5,
                "to_tune_position": 1.5,
                "target_freq_ghz": 2.856,
                "operation_mode_deg": 120.0,
                "operation_axes_deg": "60;180;300",
                "delta_operation_scaled_admittance_real": 3.0,
                "delta_operation_scaled_admittance_imag": 4.0,
                "operation_axis_error_deg": 15.0,
                "abs_operation_axis_error_deg": 15.0,
                "raw_signed_phase_step_deg": -100.0,
                "raw_phase_advance_0to360_deg": 260.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "port_side": pd.NA,
                "s_name": "S11",
                "position_family": "iris",
                "from_source_file": "iris_1p0.s1p",
                "to_source_file": "iris_2p0.s1p",
                "from_tune_position": 1.0,
                "to_tune_position": 2.0,
                "target_freq_ghz": 2.856,
                "operation_mode_deg": 120.0,
                "operation_axes_deg": "60;180;300",
                "delta_operation_scaled_admittance_real": 6.0,
                "delta_operation_scaled_admittance_imag": 8.0,
                "operation_axis_error_deg": 5.0,
                "abs_operation_axis_error_deg": 5.0,
                "raw_signed_phase_step_deg": -150.0,
                "raw_phase_advance_0to360_deg": 245.0,
            },
        ]
    )

    result = compare_cell_and_iris_responses(transitions, target_phase_advance_deg=240.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["transition_pair_index"] == 1
    assert row["cell_from_tune_position"] == pytest.approx(0.5)
    assert row["iris_from_tune_position"] == pytest.approx(1.0)
    assert row["cell_operation_scaled_admittance_delta_abs"] == pytest.approx(5.0)
    assert row["iris_operation_scaled_admittance_delta_abs"] == pytest.approx(10.0)
    assert row["operation_scaled_admittance_response_ratio_iris_over_cell"] == pytest.approx(2.0)
    assert row["phase_step_response_ratio_iris_over_cell"] == pytest.approx(1.5)
    assert row["cell_phase_residual_from_target_deg"] == pytest.approx(20.0)
    assert row["iris_phase_residual_from_target_deg"] == pytest.approx(5.0)
    assert row["supports_iris_larger_admittance_response"] is True
    assert row["supports_iris_better_branch_alignment"] is True
    assert row["supports_iris_lower_phase_residual"] is True
