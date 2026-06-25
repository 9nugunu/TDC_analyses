from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.visualization.cell_iris_response_plots import (
    plot_cell_iris_response_comparison,
)


def _comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "transition_pair_index": 1,
                "cell_from_tune_position": 0.5,
                "cell_to_tune_position": 1.5,
                "iris_from_tune_position": 1.0,
                "iris_to_tune_position": 2.0,
                "cell_operation_scaled_admittance_delta_abs": 5.0,
                "iris_operation_scaled_admittance_delta_abs": 10.0,
                "operation_scaled_admittance_response_ratio_iris_over_cell": 2.0,
                "cell_signed_phase_step_deg": -100.0,
                "iris_signed_phase_step_deg": -150.0,
                "phase_step_response_ratio_iris_over_cell": 1.5,
                "cell_phase_residual_from_target_deg": 20.0,
                "iris_phase_residual_from_target_deg": 5.0,
                "cell_abs_operation_axis_error_deg": 15.0,
                "iris_abs_operation_axis_error_deg": 5.0,
            },
            {
                "marker_name": "f_mean",
                "transition_pair_index": 1,
                "cell_from_tune_position": 0.5,
                "cell_to_tune_position": 1.5,
                "iris_from_tune_position": 1.0,
                "iris_to_tune_position": 2.0,
                "cell_operation_scaled_admittance_delta_abs": 4.0,
                "iris_operation_scaled_admittance_delta_abs": 3.0,
                "operation_scaled_admittance_response_ratio_iris_over_cell": 0.75,
                "cell_signed_phase_step_deg": -80.0,
                "iris_signed_phase_step_deg": -60.0,
                "phase_step_response_ratio_iris_over_cell": 0.75,
                "cell_phase_residual_from_target_deg": 12.0,
                "iris_phase_residual_from_target_deg": 18.0,
                "cell_abs_operation_axis_error_deg": 7.0,
                "iris_abs_operation_axis_error_deg": 11.0,
            },
        ]
    )


def test_plot_cell_iris_response_comparison_writes_named_outputs(tmp_path: Path) -> None:
    paths = plot_cell_iris_response_comparison(_comparison_table(), tmp_path)

    assert list(paths) == [
        "iris_over_cell_admittance_response_ratio",
        "iris_over_cell_phase_step_ratio",
        "cell_vs_iris_phase_residual",
        "cell_vs_iris_operation_axis_error",
    ]
    assert [path.name for path in paths.values()] == [
        "iris_over_cell_admittance_response_ratio.png",
        "iris_over_cell_phase_step_ratio.png",
        "cell_vs_iris_phase_residual.png",
        "cell_vs_iris_operation_axis_error.png",
    ]
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_cell_iris_response_comparison_rejects_missing_columns(tmp_path: Path) -> None:
    table = _comparison_table().drop(columns=["phase_step_response_ratio_iris_over_cell"])

    with pytest.raises(ValueError, match="cell_iris_response_comparison is missing required columns"):
        plot_cell_iris_response_comparison(table, tmp_path)
