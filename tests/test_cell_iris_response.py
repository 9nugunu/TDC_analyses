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
                "file_from": "cell_0p5.s1p",
                "file_to": "cell_1p5.s1p",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "freq_target_ghz": 2.856,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_delta_re": 3.0,
                "op_admit_delta_im": 4.0,
                "op_admit_axis_err_deg": 15.0,
                "op_admit_axis_err_abs_deg": 15.0,
                "phase_step_deg": -100.0,
                "phase_adv_deg": 260.0,
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
                "file_from": "iris_1p0.s1p",
                "file_to": "iris_2p0.s1p",
                "pos_from": 1.0,
                "pos_to": 2.0,
                "freq_target_ghz": 2.856,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_delta_re": 6.0,
                "op_admit_delta_im": 8.0,
                "op_admit_axis_err_deg": 5.0,
                "op_admit_axis_err_abs_deg": 5.0,
                "phase_step_deg": -150.0,
                "phase_adv_deg": 245.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "port_side": pd.NA,
                "s_name": "S11",
                "position_family": "cell",
                "file_from": "cell_9p5.s1p",
                "file_to": "cell_10p5.s1p",
                "pos_from": 9.5,
                "pos_to": 10.5,
                "freq_target_ghz": 2.856,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_delta_re": 30.0,
                "op_admit_delta_im": 40.0,
                "op_admit_axis_err_deg": 1.0,
                "op_admit_axis_err_abs_deg": 1.0,
                "phase_step_deg": -120.0,
                "phase_adv_deg": 240.0,
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
                "file_from": "iris_10p0.s1p",
                "file_to": "iris_11p0.s1p",
                "pos_from": 10.0,
                "pos_to": 11.0,
                "freq_target_ghz": 2.856,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_delta_re": 60.0,
                "op_admit_delta_im": 80.0,
                "op_admit_axis_err_deg": 1.0,
                "op_admit_axis_err_abs_deg": 1.0,
                "phase_step_deg": -120.0,
                "phase_adv_deg": 240.0,
            },
        ]
    )

    result = compare_cell_and_iris_responses(transitions, target_phase_advance_deg=240.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["pair_index"] == 1
    assert row["cell_pos_from"] == pytest.approx(0.5)
    assert row["iris_pos_from"] == pytest.approx(1.0)
    assert row["cell_admit_delta_mag"] == pytest.approx(5.0)
    assert row["iris_admit_delta_mag"] == pytest.approx(10.0)
    assert row["admit_ratio_iris_cell"] == pytest.approx(2.0)
    assert row["phase_ratio_iris_cell"] == pytest.approx(1.5)
    assert row["cell_phase_err_deg"] == pytest.approx(20.0)
    assert row["iris_phase_err_deg"] == pytest.approx(5.0)
    assert row["supports_iris_admit"] is True
    assert row["supports_iris_axis"] is True
    assert row["supports_iris_phase"] is True
