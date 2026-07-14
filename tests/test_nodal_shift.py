import pandas as pd
import pytest

from deflector_tuning.analysis.nodal_shift import compute_nodal_shift_errors


def test_compute_nodal_shift_errors_uses_marker_specific_targets_and_skips_mean() -> None:
    phase_advance = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "marker_name": "f_2pi3",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "phase_adv_deg": 245.0,
            },
            {
                "dataset_id": "dataset",
                "marker_name": "f_mean",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "phase_adv_deg": 230.0,
            },
            {
                "dataset_id": "dataset",
                "marker_name": "f_pi2",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "phase_adv_deg": 170.0,
            },
        ]
    )

    result = compute_nodal_shift_errors(phase_advance)

    assert result["marker_name"].tolist() == ["f_2pi3", "f_pi2"]
    by_marker = result.set_index("marker_name")
    assert by_marker.loc["f_2pi3", "phase_target_deg"] == 240.0
    assert by_marker.loc["f_2pi3", "phase_err_deg"] == pytest.approx(5.0)
    assert by_marker.loc["f_pi2", "phase_target_deg"] == 180.0
    assert by_marker.loc["f_pi2", "phase_err_deg"] == pytest.approx(-10.0)
    assert by_marker.loc["f_pi2", "phase_err_abs_deg"] == pytest.approx(10.0)


def test_compute_nodal_shift_errors_preserves_sim_geometry_metadata() -> None:
    phase_advance = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "sim_r_c": 54.5,
                "sim_w_c": 18.5,
                "pos_from": 0.5,
                "pos_to": 1.5,
                "phase_adv_deg": 240.0,
            }
        ]
    )

    result = compute_nodal_shift_errors(phase_advance)

    assert result.loc[0, "sim_r_c"] == 54.5
    assert result.loc[0, "sim_w_c"] == 18.5
