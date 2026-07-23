from __future__ import annotations

import pandas as pd
import pytest

from deflector_tuning.analysis.phase_offset_response import (
    build_phase_offset_response,
)
from deflector_tuning.visualization.phase_offset_plots import (
    plot_phase_offset_response,
)


def _marker_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    phases = {
        "f_2pi3": {
            "2_s003_tune_-1mmoffset.S2P": 60.0,
            "2_s003_tune_zerooffset.S2P": 70.0,
            "2_s003_tune_zerooffset_again.S2P": 72.0,
            "2_s003_tune_+1mmoffset.S2P": 80.0,
        },
        "f_mean": {
            "2_s003_tune_-1mmoffset.S2P": 10.0,
            "2_s003_tune_zerooffset.S2P": 20.0,
            "2_s003_tune_zerooffset_again.S2P": 24.0,
            "2_s003_tune_+1mmoffset.S2P": 30.0,
        },
    }
    for marker_name, by_file in phases.items():
        for source_file, phase in by_file.items():
            rows.append(
                {
                    "dataset_id": "raw_sweep_260721_tune_s003_plungersensitivity",
                    "source_file": source_file,
                    "tune_position": 2.0,
                    "marker_name": marker_name,
                    "s_phase_deg": phase,
                }
            )
    return pd.DataFrame(rows)


def test_build_phase_offset_response_uses_zero_offset_reference_per_mode() -> None:
    response = build_phase_offset_response(_marker_points())

    assert response.columns.tolist() == [
        "dataset_id",
        "source_file",
        "tune_position",
        "marker_name",
        "plunger_offset_mm",
        "phase_deg",
        "phase_delta_deg",
        "reference_phase_deg",
    ]
    f_mean = response[response["marker_name"] == "f_mean"].set_index("source_file")
    assert f_mean.loc["2_s003_tune_-1mmoffset.S2P", "plunger_offset_mm"] == -1.0
    assert f_mean.loc["2_s003_tune_+1mmoffset.S2P", "plunger_offset_mm"] == 1.0
    assert f_mean.loc["2_s003_tune_zerooffset.S2P", "reference_phase_deg"] == pytest.approx(22.0)
    assert f_mean.loc["2_s003_tune_-1mmoffset.S2P", "phase_delta_deg"] == pytest.approx(-12.0)
    assert f_mean.loc["2_s003_tune_+1mmoffset.S2P", "phase_delta_deg"] == pytest.approx(8.0)


def test_build_phase_offset_response_requires_offset_tokens() -> None:
    marker_points = _marker_points()
    marker_points.loc[0, "source_file"] = "2_s003_tune_unknown.S2P"

    with pytest.raises(ValueError, match="plunger offset"):
        build_phase_offset_response(marker_points)


def test_plot_phase_offset_response_writes_delta_phase_figure(tmp_path) -> None:
    response = build_phase_offset_response(_marker_points())

    path = plot_phase_offset_response(response, tmp_path)

    assert path.name == "phase_vs_plunger_offset.png"
    assert path.is_file()
