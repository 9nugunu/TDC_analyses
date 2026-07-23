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
                "pair_index": 1,
                "cell_pos_from": 0.5,
                "cell_pos_to": 1.5,
                "iris_pos_from": 1.0,
                "iris_pos_to": 2.0,
                "cell_admit_delta_mag": 5.0,
                "iris_admit_delta_mag": 10.0,
                "admit_ratio_iris_cell": 2.0,
                "cell_signed_phase_step_deg": -100.0,
                "iris_signed_phase_step_deg": -150.0,
                "phase_ratio_iris_cell": 1.5,
                "cell_phase_err_deg": 20.0,
                "iris_phase_err_deg": 5.0,
                "cell_admit_axis_err_abs_deg": 15.0,
                "iris_admit_axis_err_abs_deg": 5.0,
            },
            {
                "marker_name": "f_mean",
                "pair_index": 1,
                "cell_pos_from": 0.5,
                "cell_pos_to": 1.5,
                "iris_pos_from": 1.0,
                "iris_pos_to": 2.0,
                "cell_admit_delta_mag": 4.0,
                "iris_admit_delta_mag": 3.0,
                "admit_ratio_iris_cell": 0.75,
                "cell_signed_phase_step_deg": -80.0,
                "iris_signed_phase_step_deg": -60.0,
                "phase_ratio_iris_cell": 0.75,
                "cell_phase_err_deg": 12.0,
                "iris_phase_err_deg": 18.0,
                "cell_admit_axis_err_abs_deg": 7.0,
                "iris_admit_axis_err_abs_deg": 11.0,
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
    table = _comparison_table().drop(columns=["phase_ratio_iris_cell"])

    with pytest.raises(ValueError, match="cell_iris_response_comparison is missing required columns"):
        plot_cell_iris_response_comparison(table, tmp_path)
