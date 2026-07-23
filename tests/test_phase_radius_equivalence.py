import pandas as pd
import pytest

from deflector_tuning.analysis.phase_radius_equivalence import (
    build_experiment_simulation_phase_comparison,
    build_raw_anchored_phase_position,
    estimate_phase_radius_equivalence,
    fit_rc_states,
)
from deflector_tuning.visualization.phase_radius_equivalence_plots import (
    plot_phase_cmp_bars,
)


def test_fit_rc_states_anchors_current_then_fits_baseline_from_three_markers() -> None:
    radii = [56.40, 56.50, 56.60]
    slopes = {"f_2pi3": -100.0, "f_mean": -150.0, "f_pi2": -80.0}
    line = pd.DataFrame(
        [
            {
                "sim_r_c": radius,
                "marker_name": marker,
                "s_phase_deg": slope * (radius - 56.60),
            }
            for marker, slope in slopes.items()
            for radius in radii
        ]
    )
    observation = pd.DataFrame(
        [
            {"marker_name": "f_2pi3", "before_phase_deg": 24.0, "after_phase_deg": 20.0},
            {"marker_name": "f_mean", "before_phase_deg": 13.5, "after_phase_deg": 7.5},
            {"marker_name": "f_pi2", "before_phase_deg": -6.8, "after_phase_deg": -10.0},
        ]
    )

    fit = fit_rc_states(line, observation, design_r_c_mm=56.60, step_um=0.1)

    radii_by_state = fit.set_index("state")["r_c_mm"]
    assert radii_by_state["baseline"] == pytest.approx(56.51, abs=1e-9)
    assert radii_by_state["current"] == pytest.approx(56.55, abs=1e-9)
    assert radii_by_state["design"] == pytest.approx(56.60, abs=1e-9)
    assert fit.loc[fit["state"] == "baseline", "rms_residual_deg"].iloc[0] == pytest.approx(0.0)


def test_estimate_phase_radius_equivalence_recovers_shared_radius_shift() -> None:
    line = pd.DataFrame(
        [
            {"sim_r_c": 56.59, "marker_name": "f_2pi3", "s_phase_deg": 10.0},
            {"sim_r_c": 56.64, "marker_name": "f_2pi3", "s_phase_deg": 5.0},
            {"sim_r_c": 56.69, "marker_name": "f_2pi3", "s_phase_deg": 0.0},
            {"sim_r_c": 56.59, "marker_name": "f_mean", "s_phase_deg": 20.0},
            {"sim_r_c": 56.64, "marker_name": "f_mean", "s_phase_deg": 12.5},
            {"sim_r_c": 56.69, "marker_name": "f_mean", "s_phase_deg": 5.0},
        ]
    )
    observed = pd.DataFrame(
        [
            {"marker_name": "f_2pi3", "phase_delta_deg": -5.0},
            {"marker_name": "f_mean", "phase_delta_deg": -7.5},
        ]
    )

    estimate, curves = estimate_phase_radius_equivalence(
        line,
        observed,
        baseline_r_c_mm=56.59,
        min_delta_r_c_um=0.0,
        max_delta_r_c_um=100.0,
        step_um=1.0,
    )

    assert estimate["equivalent_delta_r_c_um"].tolist() == [50.0, 50.0]
    assert estimate["marker_equivalent_delta_r_c_um"].tolist() == [50.0, 50.0]
    assert estimate["phase_residual_deg"].tolist() == [0.0, 0.0]
    assert curves["delta_r_c_um"].min() == 0.0
    assert curves["delta_r_c_um"].max() == 100.0


def test_raw_anchored_phase_position_places_after_measurement_on_radius_line() -> None:
    line = pd.DataFrame(
        [
            {"sim_r_c": 56.59, "marker_name": "f_mean", "s_phase_deg": 10.0},
            {"sim_r_c": 56.64, "marker_name": "f_mean", "s_phase_deg": 5.0},
            {"sim_r_c": 56.69, "marker_name": "f_mean", "s_phase_deg": 0.0},
        ]
    )
    raw = pd.DataFrame(
        [{"marker_name": "f_mean", "before_phase_deg": 20.0, "after_phase_deg": 15.0}]
    )

    position, curves = build_raw_anchored_phase_position(
        line,
        raw,
        baseline_r_c_mm=56.59,
        target_r_c_mm=56.69,
        step_um=1.0,
    )

    assert position.loc[0, "marker_equivalent_r_c_mm"] == pytest.approx(56.64)
    assert position.loc[0, "shared_equivalent_r_c_mm"] == pytest.approx(56.64)
    assert position.loc[0, "remaining_to_target_um"] == pytest.approx(50.0)
    assert curves["raw_anchored_phase_deg"].tolist() == [20.0, 15.0, 10.0]


def test_experiment_simulation_phase_comparison_and_bar_plot(tmp_path) -> None:
    line = pd.DataFrame(
        [
            {"sim_r_c": 56.10, "marker_name": "f_mean", "s_phase_deg": 30.0},
            {"sim_r_c": 56.20, "marker_name": "f_mean", "s_phase_deg": 20.0},
        ]
    )
    raw = pd.DataFrame(
        [{"marker_name": "f_mean", "before_phase_deg": 5.0, "after_phase_deg": -5.0}]
    )

    comparison = build_experiment_simulation_phase_comparison(
        line,
        raw,
        before_r_c_mm=56.10,
        current_r_c_mm=56.20,
    )
    output = plot_phase_cmp_bars(comparison, tmp_path / "phase_cmp_bars.png")

    assert comparison.loc[0, "measured_phase_delta_deg"] == pytest.approx(-10.0)
    assert comparison.loc[0, "simulated_phase_delta_deg"] == pytest.approx(-10.0)
    assert comparison.loc[0, "delta_residual_deg"] == pytest.approx(0.0)
    assert output.stat().st_size > 0
