import pandas as pd
import pytest

from deflector_tuning.analysis.phase_summary import summarize_phase_advance


def test_summarize_phase_advance_reports_error_metrics_and_worst_transition() -> None:
    phase_table = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_tune_position": 0.5,
                "to_tune_position": 1.0,
                "phase_advance_0to360_deg": 230.0,
                "phase_error_from_240_deg": -10.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_tune_position": 1.0,
                "to_tune_position": 1.5,
                "phase_advance_0to360_deg": 245.0,
                "phase_error_from_240_deg": 5.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_tune_position": 1.5,
                "to_tune_position": 2.0,
                "phase_advance_0to360_deg": 220.0,
                "phase_error_from_240_deg": -20.0,
            },
        ]
    )

    summary = summarize_phase_advance(phase_table)

    assert list(summary.columns) == [
        "dataset_id",
        "data_kind",
        "data_layer",
        "marker_name",
        "marker_role",
        "port_side",
        "s_name",
        "transition_count",
        "mean_phase_advance_deg",
        "mean_phase_error_deg",
        "mean_abs_phase_error_deg",
        "rms_phase_error_deg",
        "max_abs_phase_error_deg",
        "worst_from_tune_position",
        "worst_to_tune_position",
        "worst_phase_error_deg",
    ]
    row = summary.iloc[0]
    assert pd.isna(row["port_side"])
    assert row["transition_count"] == 3
    assert row["mean_phase_advance_deg"] == pytest.approx((230 + 245 + 220) / 3)
    assert row["mean_phase_error_deg"] == pytest.approx((-10 + 5 - 20) / 3)
    assert row["mean_abs_phase_error_deg"] == pytest.approx((10 + 5 + 20) / 3)
    assert row["rms_phase_error_deg"] == pytest.approx(((100 + 25 + 400) / 3) ** 0.5)
    assert row["max_abs_phase_error_deg"] == 20.0
    assert row["worst_from_tune_position"] == 1.5
    assert row["worst_to_tune_position"] == 2.0
    assert row["worst_phase_error_deg"] == -20.0


def test_summarize_phase_advance_keeps_markers_and_port_sides_separate() -> None:
    phase_table = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "raw",
                "marker_name": marker,
                "marker_role": "exp",
                "port_side": port_side,
                "s_name": "S11",
                "from_tune_position": 0.5,
                "to_tune_position": 1.5,
                "phase_advance_0to360_deg": 240.0 + error,
                "phase_error_from_240_deg": error,
            }
            for marker, error in [("f_2pi3", 5.0), ("f_mean", -15.0)]
            for port_side in ["in", "out"]
        ]
    )

    summary = summarize_phase_advance(phase_table)

    assert len(summary) == 4
    assert set(summary["marker_name"]) == {"f_2pi3", "f_mean"}
    assert set(summary["port_side"]) == {"in", "out"}
