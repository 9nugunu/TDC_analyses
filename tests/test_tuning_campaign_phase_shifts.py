from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from deflector_tuning.workflows.tuning_phase_shifts import (
    _position_label,
    build_phase_shift_table,
    run_tuning_campaign_phase_shifts,
)


def _points(*rows: tuple[float, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file": "portE.S2P",
                "tune_position": position,
                "marker_name": marker_name,
                "s_phase_deg": phase_deg,
            }
            for position, marker_name, phase_deg in rows
        ]
    )


def test_build_phase_shift_table_uses_first_available_state_as_each_position_reference() -> None:
    state_points = {
        "s001": _points(
            (2.0, "f_2pi3", 179.0),
            (2.0, "f_mean", 20.0),
            (2.0, "f_pi2", -40.0),
        ),
        "s002": _points(
            (2.0, "f_2pi3", 178.0),
            (2.0, "f_mean", 18.0),
            (2.0, "f_pi2", -43.0),
            (1.5, "f_2pi3", -50.0),
            (1.5, "f_mean", -100.0),
            (1.5, "f_pi2", -130.0),
        ),
        "s003": _points(
            (2.0, "f_2pi3", -178.0),
            (2.0, "f_mean", 15.0),
            (2.0, "f_pi2", -45.0),
            (1.5, "f_2pi3", -25.0),
            (1.5, "f_mean", -85.0),
            (1.5, "f_pi2", -120.0),
        ),
    }

    shifts = build_phase_shift_table(
        state_points,
        state_order=("s001", "s002", "s003"),
        positions={"Iris 2.0": 2.0, "Cell 1.5": 1.5},
    )

    iris_s003 = shifts[
        (shifts["state_id"] == "s003")
        & (shifts["position_label"] == "Iris 2.0")
        & (shifts["marker_name"] == "f_2pi3")
    ].iloc[0]
    assert iris_s003["baseline_state"] == "s001"
    assert iris_s003["delta_phase_deg"] == pytest.approx(3.0)

    cell_s002 = shifts[
        (shifts["state_id"] == "s002")
        & (shifts["position_label"] == "Cell 1.5")
        & (shifts["marker_name"] == "f_mean")
    ].iloc[0]
    assert cell_s002["baseline_state"] == "s002"
    assert cell_s002["delta_phase_deg"] == pytest.approx(0.0)
    assert not ((shifts["state_id"] == "s001") & (shifts["position_label"] == "Cell 1.5")).any()


def test_position_label_preserves_requested_decimal_position() -> None:
    assert _position_label("iris", 2.0) == "Iris 2.0"
    assert _position_label("cell", 1.5) == "Cell 1.5"


def test_phase_shifts_start_at_s002_and_ignore_later_states(tmp_path) -> None:
    analysis_root = tmp_path / "analyses"
    for state_id, phase_shift in (("s001", 0.0), ("s002", 1.0), ("s003", 2.0)):
        table_path = analysis_root / f"dataset_{state_id}" / "tables" / "marker_pts.csv"
        table_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "source_file": "portE.S2P",
                    "tune_position": position,
                    "marker_name": marker_name,
                    "s_phase_deg": base_phase + phase_shift,
                }
                for position, base_phase in ((2.0, 10.0), (1.5, -20.0))
                for marker_name in ("f_2pi3", "f_mean", "f_pi2")
            ]
        ).to_csv(table_path, index=False)
    campaign = SimpleNamespace(
        baseline_state_id="s000",
        states={
            state_id: SimpleNamespace(dataset=f"dataset_{state_id}")
            for state_id in ("s000", "s001", "s002", "s003")
        },
        simulation_references={
            "iris": SimpleNamespace(experiment_positions=(1.0, 2.0)),
            "cell": SimpleNamespace(experiment_positions=(0.5, 1.5)),
        },
    )

    outputs = run_tuning_campaign_phase_shifts(
        campaign,
        analysis_root=analysis_root,
        output_dir=tmp_path / "output",
        current_state_id="s002",
    )

    assert outputs is not None
    summary = pd.read_csv(outputs.table_path)
    assert set(summary["state_id"]) == {"s001", "s002"}
