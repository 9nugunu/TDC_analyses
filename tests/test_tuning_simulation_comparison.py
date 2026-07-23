from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deflector_tuning.tuning_campaign import PhaseReference, SimulationReference
from deflector_tuning.workflows.tuning_simulation_comparison import (
    _simulation_rc_line,
    build_anchor_check,
    build_position_phase_change,
    build_tuning_family_comparison,
    measurement_family_for_position,
)


def test_simulation_rc_line_sorts_each_marker_by_radius_for_plotting() -> None:
    table = pd.DataFrame(
        [
            {"sim_r_c": 56.6, "marker_name": "f_mean", "s_phase_deg": 0.0},
            {"sim_r_c": 56.4, "marker_name": "f_mean", "s_phase_deg": 20.0},
            {"sim_r_c": 56.5, "marker_name": "f_mean", "s_phase_deg": 10.0},
        ]
    )

    line = _simulation_rc_line(table, marker_names=["f_mean"])

    assert line["sim_r_c"].tolist() == [56.4, 56.5, 56.6]


@pytest.mark.parametrize(
    ("position", "expected"),
    [(1.0, "iris"), (2.0, "iris"), (0.5, "cell"), (1.5, "cell")],
)
def test_measurement_family_for_position(position: float, expected: str) -> None:
    assert measurement_family_for_position(position) == expected


def test_anchor_check_prefers_port_extended_trace_and_wraps_180_degrees() -> None:
    table = pd.DataFrame(
        [
            {"source_file": "0.5_beforecal.S2P", "tune_position": 0.5, "marker_name": "f_mean", "s_phase_deg": 20.0},
            {"source_file": "0.5_portE.S2P", "tune_position": 0.5, "marker_name": "f_mean", "s_phase_deg": -178.9},
        ]
    )

    check = build_anchor_check(
        table,
        position=0.5,
        expected_phase_deg=180.0,
        tolerance_deg=5.0,
        port_extension_applied=True,
    )

    assert check["source_file"].tolist() == ["0.5_portE.S2P"]
    assert check["phase_error_deg"].iloc[0] == pytest.approx(1.1)
    assert bool(check["verified"].iloc[0]) is True


def test_position_phase_change_uses_wrapped_after_minus_before() -> None:
    before = pd.DataFrame(
        [
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": 179.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 25.0},
        ]
    )
    after = pd.DataFrame(
        [
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": -178.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 20.0},
        ]
    )

    change = build_position_phase_change(
        before,
        after,
        position=2.0,
        port_extension_applied=True,
    )

    assert change["marker_name"].tolist() == ["f_2pi3", "f_mean"]
    np.testing.assert_allclose(change["phase_delta_deg"], [3.0, -5.0])


def test_build_tuning_family_comparison_fits_current_then_baseline() -> None:
    simulation = pd.DataFrame(
        [
            {
                "sim_r_c": radius,
                "marker_name": marker,
                "s_phase_deg": slope * (radius - 56.60),
            }
            for marker, slope in {"f_2pi3": -100.0, "f_mean": -150.0, "f_pi2": -80.0}.items()
            for radius in (56.40, 56.50, 56.60)
        ]
    )
    before = pd.DataFrame(
        [
            {"source_file": "1_portE.S2P", "tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": -179.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": 24.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 13.5},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_pi2", "s_phase_deg": -6.8},
        ]
    )
    after = pd.DataFrame(
        [
            {"source_file": "1_portE.S2P", "tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": -178.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_2pi3", "s_phase_deg": 20.0},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_mean", "s_phase_deg": 7.5},
            {"source_file": "2_portE.S2P", "tune_position": 2.0, "marker_name": "f_pi2", "s_phase_deg": -10.0},
        ]
    )

    result = build_tuning_family_comparison(
        simulation,
        before,
        after,
        reference=SimulationReference(
            family="iris",
            dataset="sim_rc",
            experiment_positions=(1.0, 2.0),
        ),
        phase_reference=PhaseReference(port_extension="applied"),
        design_r_c_mm=56.60,
    )

    assert result.family == "iris"
    assert result.anchor_check["state"].tolist() == ["baseline", "current"]
    fit = result.rc_fit.set_index("state")["r_c_mm"]
    assert fit["baseline"] == pytest.approx(56.51)
    assert fit["current"] == pytest.approx(56.55)
    assert result.rc_fit["interpretation"].unique().tolist() == [
        "simulation-equivalent coordinate; not physical bolt travel"
    ]

    ambiguous_grid = pd.concat(
        [simulation.assign(sim_w_c=19.0), simulation.assign(sim_w_c=20.0)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="one-dimensional r_c.*sim_w_c"):
        build_tuning_family_comparison(
            ambiguous_grid,
            before,
            after,
            reference=SimulationReference(
                family="iris",
                dataset="sim_grid",
                experiment_positions=(1.0, 2.0),
            ),
            phase_reference=PhaseReference(port_extension="applied"),
            design_r_c_mm=56.60,
        )
