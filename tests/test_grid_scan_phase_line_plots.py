from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.grid_scan_phase_line_plots import (
    plot_grid_scan_sparameter_phase_r_c_line_scan,
)


def test_plot_grid_scan_sparameter_phase_r_c_line_scan_writes_named_unwrapped_phase_plot(tmp_path: Path) -> None:
    marker_points = pd.DataFrame(
        [
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -20.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -30.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 18.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -40.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_mean",
                "s_phase_deg": -170.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_mean",
                "s_phase_deg": 170.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_pi2",
                "s_phase_deg": 30.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_pi2",
                "s_phase_deg": 20.0,
            },
        ]
    )

    paths = plot_grid_scan_sparameter_phase_r_c_line_scan(marker_points, tmp_path)

    assert list(paths) == ["sparameter_phase_vs_r_c_at_w_c_19p3224"]
    assert paths["sparameter_phase_vs_r_c_at_w_c_19p3224"].name == "sparameter_phase_vs_r_c_at_w_c_19p3224.png"
    assert paths["sparameter_phase_vs_r_c_at_w_c_19p3224"].stat().st_size > 0
    csv = pd.read_csv(tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224.csv")
    assert csv["sim_w_c"].tolist() == [19.3224] * 6
    assert csv["sim_r_c"].tolist() == [54.59, 55.59, 54.59, 55.59, 54.59, 55.59]
    assert csv["marker_name"].tolist() == ["f_2pi3", "f_2pi3", "f_mean", "f_mean", "f_pi2", "f_pi2"]
    assert csv["s_phase_deg"].tolist() == [-30.0, -20.0, 170.0, -170.0, 20.0, 30.0]
    assert csv["s_phase_deg_0_360"].tolist() == [330.0, 340.0, 170.0, 190.0, 20.0, 30.0]
    assert csv["s_phase_deg_unwrapped"].tolist() == [-30.0, -20.0, 170.0, 190.0, 20.0, 30.0]


def test_plot_grid_scan_sparameter_phase_r_c_line_scan_keeps_one_point_per_radius(tmp_path: Path) -> None:
    marker_points = pd.DataFrame(
        [
            {
                "sim_r_c": 56.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -176.0,
                "source_file": "run_1.s1p",
                "run_id": 1,
            },
            {
                "sim_r_c": 56.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": 62.0,
                "source_file": "run_2.s1p",
                "run_id": 2,
            },
        ]
    )

    plot_grid_scan_sparameter_phase_r_c_line_scan(marker_points, tmp_path)

    csv = pd.read_csv(tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224.csv")
    assert csv["s_phase_deg"].tolist() == [62.0]
    assert csv["s_phase_deg_0_360"].tolist() == [62.0]
    assert csv["s_phase_deg_unwrapped"].tolist() == [62.0]
    assert csv["source_file"].tolist() == ["run_2.s1p"]
