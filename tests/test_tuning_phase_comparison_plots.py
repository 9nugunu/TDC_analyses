from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.tuning_phase_comparison_plots import (
    plot_before_after_sparameter_phase_position_scan,
)


def test_plot_before_after_phase_position_scan_uses_only_common_finite_positions(tmp_path: Path) -> None:
    before = pd.DataFrame(
        [
            {"tune_position": 1.0, "marker_name": "f_2pi3", "s_phase_deg": -165.0},
            {"tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": 74.0},
            {"tune_position": 3.0, "marker_name": "f_2pi3", "s_phase_deg": -59.0},
            {"tune_position": 4.0, "marker_name": "f_2pi3", "s_phase_deg": 20.0},
            {"tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": -179.0},
            {"tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 27.0},
            {"tune_position": 3.0, "marker_name": "f_mean", "s_phase_deg": -150.0},
        ]
    )
    after = pd.DataFrame(
        [
            {"tune_position": 1.0, "marker_name": "f_2pi3", "s_phase_deg": -166.0},
            {"tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": 70.0},
            {"tune_position": 3.0, "marker_name": "f_2pi3", "s_phase_deg": -63.0},
            {"tune_position": None, "marker_name": "f_2pi3", "s_phase_deg": -115.0},
            {"tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": -179.0},
            {"tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 20.0},
            {"tune_position": 3.0, "marker_name": "f_mean", "s_phase_deg": -152.0},
        ]
    )

    path = plot_before_after_sparameter_phase_position_scan(
        before,
        after,
        tmp_path,
        before_label="Before tuning",
        after_label="After 13.5 torque",
    )

    assert path.name == "sparameter_phase_before_after_position_scan.png"
    assert path.stat().st_size > 0
    comparison = pd.read_csv(tmp_path / "sparameter_phase_before_after_position_scan.csv")
    assert comparison["tune_position"].tolist() == [1.0, 2.0, 3.0] * 4
    assert comparison["marker_name"].tolist() == ["f_2pi3"] * 6 + ["f_mean"] * 6
    assert comparison["tune_state"].tolist() == ["before"] * 3 + ["after"] * 3 + ["before"] * 3 + ["after"] * 3
    assert comparison["tune_position"].isna().sum() == 0
