"""Shared deterministic marker tables and figure stubs for runner tests."""

from pathlib import Path

import pandas as pd


def _tables() -> dict[str, pd.DataFrame]:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_mean",
                "freq_ghz": 2.866,
                "s_db": -2.0,
                "s_phase_deg": 60.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_pi2",
                "freq_ghz": 2.876,
                "s_db": -3.0,
                "s_phase_deg": 120.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_mean",
                "freq_ghz": 2.866,
                "s_db": -2.0,
                "s_phase_deg": 70.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_pi2",
                "freq_ghz": 2.876,
                "s_db": -3.0,
                "s_phase_deg": 140.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
        ]
    )
    return {
        "markers": pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
        "marker_pts": marker_points,
        "phase_adv": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "pos_from": 0.5,
                    "pos_to": 1.5,
                    "phase_adv_deg": 240.0,
                    "phase_err_240_deg": 0.0,
                }
            ]
        ),
        "phase_stats": pd.DataFrame([{"marker_name": "f_2pi3", "n_steps": 1}]),
        "nodal_shift": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "pos_from": 0.5,
                    "pos_to": 1.5,
                    "position_family": "cell",
                    "phase_err_deg": 0.0,
                    "phase_err_abs_deg": 0.0,
                }
            ]
        ),
    }


def _cell_iris_response_comparison_table() -> pd.DataFrame:
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
                "cell_phase_step_deg": -100.0,
                "iris_phase_step_deg": -150.0,
                "phase_ratio_iris_cell": 1.5,
                "cell_phase_err_deg": 20.0,
                "iris_phase_err_deg": 5.0,
                "cell_admit_axis_err_abs_deg": 15.0,
                "iris_admit_axis_err_abs_deg": 5.0,
            }
        ]
    )


def _coupler_cavity_parameter_estimates_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "coupler_freq_ghz": 2.866,
                "q_ext": 50.0,
                "beta": 0.95,
                "q_ext_target": 57.7,
                "is_valid": True,
            }
        ]
    )


def _kyhl_admittance_points_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "op_admit_re": 1.0,
                "op_admit_im": 0.5,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 60.0,
                "op_admit_axis_deg": 60.0,
                "op_admit_axis_err_deg": 0.0,
            }
        ]
    )


def _kyhl_f2pi3_normalized_admittance_audit_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "mode_admit_re": 0.2,
                "mode_admit_im": -1.7,
                "mode_gamma_re": -0.5,
                "mode_gamma_im": -0.8660254,
                "mode_gamma_mag": 1.0,
                "mode_gamma_ang_deg": 240.0,
            }
        ]
    )


def _kyhl_admittance_transitions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "op_admit_delta_re": 0.5,
                "op_admit_delta_im": -1.0,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 60.0,
                "op_admit_axis_deg": 60.0,
                "op_admit_axis_err_deg": 0.0,
            }
        ]
    )


def fake_figure_plot(name: str, *, recorded_kwargs: dict[str, object] | None = None):
    """Create a figure stub using the same folder argument as its real plotter."""
    def plot(*args, **kwargs):
        if recorded_kwargs is not None:
            recorded_kwargs.update(kwargs)
        folder = Path(args[2] if name == "s11" else args[1])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        path.write_text(name, encoding="utf-8")
        return {"overview": path}
    return plot
