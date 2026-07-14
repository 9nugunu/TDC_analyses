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
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.0,
                "phase_adv_deg": 230.0,
                "phase_err_240_deg": -10.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "position_family": "cell",
                "pos_from": 1.0,
                "pos_to": 1.5,
                "phase_adv_deg": 245.0,
                "phase_err_240_deg": 5.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "position_family": "cell",
                "pos_from": 1.5,
                "pos_to": 2.0,
                "phase_adv_deg": 220.0,
                "phase_err_240_deg": -20.0,
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
        "position_family",
        "n_steps",
        "phase_adv_mean_deg",
        "phase_err_mean_deg",
        "phase_err_abs_mean_deg",
        "phase_err_rms_deg",
        "phase_err_abs_max_deg",
        "worst_pos_from",
        "worst_pos_to",
        "worst_phase_err_deg",
    ]
    row = summary.iloc[0]
    assert pd.isna(row["port_side"])
    assert row["position_family"] == "cell"
    assert row["n_steps"] == 3
    assert row["phase_adv_mean_deg"] == pytest.approx((230 + 245 + 220) / 3)
    assert row["phase_err_mean_deg"] == pytest.approx((-10 + 5 - 20) / 3)
    assert row["phase_err_abs_mean_deg"] == pytest.approx((10 + 5 + 20) / 3)
    assert row["phase_err_rms_deg"] == pytest.approx(((100 + 25 + 400) / 3) ** 0.5)
    assert row["phase_err_abs_max_deg"] == 20.0
    assert row["worst_pos_from"] == 1.5
    assert row["worst_pos_to"] == 2.0
    assert row["worst_phase_err_deg"] == -20.0


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
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "phase_adv_deg": 240.0 + error,
                "phase_err_240_deg": error,
            }
            for marker, error in [("f_2pi3", 5.0), ("f_mean", -15.0)]
            for port_side in ["in", "out"]
        ]
    )

    summary = summarize_phase_advance(phase_table)

    assert len(summary) == 4
    assert set(summary["marker_name"]) == {"f_2pi3", "f_mean"}
    assert set(summary["port_side"]) == {"in", "out"}
