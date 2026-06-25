from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.kyhl_admittance_plots import plot_kyhl_operation_polar


def test_plot_kyhl_operation_polar_writes_png(tmp_path: Path) -> None:
    points = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "source_file": "run_1.s1p",
                "tune_position": 0.5,
                "operation_mode_deg": 120.0,
                "operation_axes_deg": "60;180;300",
                "kyhl_operation_angle_deg": 90.0,
                "nearest_operation_axis_deg": 60.0,
                "operation_axis_error_deg": 30.0,
            },
            {
                "marker_name": "f_2pi3",
                "source_file": "run_2.s1p",
                "tune_position": 1.0,
                "operation_mode_deg": 120.0,
                "operation_axes_deg": "60;180;300",
                "kyhl_operation_angle_deg": 270.0,
                "nearest_operation_axis_deg": 300.0,
                "operation_axis_error_deg": -30.0,
            },
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "position_family": "cell",
                "from_tune_position": 0.5,
                "to_tune_position": 1.5,
                "operation_mode_deg": 120.0,
                "operation_axes_deg": "60;180;300",
                "kyhl_operation_angle_deg": 270.0,
                "nearest_operation_axis_deg": 300.0,
                "operation_axis_error_deg": -30.0,
            }
        ]
    )

    output_path = plot_kyhl_operation_polar(points, transitions, tmp_path / "kyhl.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
