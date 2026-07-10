from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.kyhl_admittance_plots import (
    plot_f2pi3_normalized_admittance_view,
    plot_kyhl_operation_polar,
)


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


def test_plot_f2pi3_normalized_admittance_view_writes_only_f2pi3_png(tmp_path: Path) -> None:
    points = pd.DataFrame(
        [
            _normalized_point("f_2pi3", 0.5, 240.0, 1.0, 0.2, -1.7),
            _normalized_point("f_2pi3", 1.0, 120.0, 0.95, 0.1, 1.5),
            _normalized_point("f_pi2", 0.5, 30.0, 0.5, 10.0, 10.0),
            _normalized_point("f_pi2", 1.0, 45.0, 0.4, 12.0, 12.0),
        ]
    )

    paths = plot_f2pi3_normalized_admittance_view(points, tmp_path)

    assert set(paths) == {"f_2pi3"}
    assert paths["f_2pi3"].name == "f_2pi3_normalized_admittance.png"
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def _normalized_point(
    marker_name: str,
    tune_position: float,
    reflection_angle_deg: float,
    reflection_abs: float,
    admittance_real: float,
    admittance_imag: float,
) -> dict[str, object]:
    return {
        "marker_name": marker_name,
        "source_file": f"run_{tune_position:g}.s1p",
        "tune_position": tune_position,
        "mode_normalized_admittance_real": admittance_real,
        "mode_normalized_admittance_imag": admittance_imag,
        "mode_reflection_real": reflection_abs,
        "mode_reflection_imag": 0.0,
        "mode_reflection_abs": reflection_abs,
        "mode_reflection_angle_deg": reflection_angle_deg,
        "operation_mode_deg": 120.0,
        "operation_axes_deg": "60;180;300",
        "kyhl_operation_angle_deg": reflection_angle_deg,
        "nearest_operation_axis_deg": 60.0 if reflection_angle_deg < 180.0 else 300.0,
        "operation_axis_error_deg": 30.0,
    }


def _transition(
    marker_name: str,
    position_family: str,
    from_tune_position: float,
    to_tune_position: float,
    angle_deg: float,
    from_admittance_real: float = 1.0,
    from_admittance_imag: float = 0.0,
    to_admittance_real: float = 1.5,
    to_admittance_imag: float = 0.5,
) -> dict[str, object]:
    return {
        "marker_name": marker_name,
        "position_family": position_family,
        "from_tune_position": from_tune_position,
        "to_tune_position": to_tune_position,
        "from_admittance_real": from_admittance_real,
        "from_admittance_imag": from_admittance_imag,
        "to_admittance_real": to_admittance_real,
        "to_admittance_imag": to_admittance_imag,
        "operation_mode_deg": 120.0,
        "operation_axes_deg": "60;180;300",
        "kyhl_operation_angle_deg": angle_deg,
        "nearest_operation_axis_deg": 300.0,
        "operation_axis_error_deg": -30.0,
    }
