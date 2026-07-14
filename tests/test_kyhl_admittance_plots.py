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
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 90.0,
                "op_admit_axis_deg": 60.0,
                "op_admit_axis_err_deg": 30.0,
            },
            {
                "marker_name": "f_2pi3",
                "source_file": "run_2.s1p",
                "tune_position": 1.0,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 270.0,
                "op_admit_axis_deg": 300.0,
                "op_admit_axis_err_deg": -30.0,
            },
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 270.0,
                "op_admit_axis_deg": 300.0,
                "op_admit_axis_err_deg": -30.0,
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
        "mode_admit_re": admittance_real,
        "mode_admit_im": admittance_imag,
        "mode_gamma_re": reflection_abs,
        "mode_gamma_im": 0.0,
        "mode_gamma_mag": reflection_abs,
        "mode_gamma_ang_deg": reflection_angle_deg,
        "op_mode_deg": 120.0,
        "op_admit_axes_deg": "60;180;300",
        "op_admit_ang_deg": reflection_angle_deg,
        "op_admit_axis_deg": 60.0 if reflection_angle_deg < 180.0 else 300.0,
        "op_admit_axis_err_deg": 30.0,
    }


def _transition(
    marker_name: str,
    position_family: str,
    pos_from: float,
    pos_to: float,
    angle_deg: float,
    from_admittance_real: float = 1.0,
    from_admittance_imag: float = 0.0,
    to_admittance_real: float = 1.5,
    to_admittance_imag: float = 0.5,
) -> dict[str, object]:
    return {
        "marker_name": marker_name,
        "position_family": position_family,
        "pos_from": pos_from,
        "pos_to": pos_to,
        "from_admittance_real": from_admittance_real,
        "from_admittance_imag": from_admittance_imag,
        "to_admittance_real": to_admittance_real,
        "to_admittance_imag": to_admittance_imag,
        "op_mode_deg": 120.0,
        "op_admit_axes_deg": "60;180;300",
        "op_admit_ang_deg": angle_deg,
        "op_admit_axis_deg": 300.0,
        "op_admit_axis_err_deg": -30.0,
    }
