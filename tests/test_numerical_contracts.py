"""Analytical regression cases for audited numerical input contracts."""

import numpy as np
import pandas as pd
import pytest

from deflector_tuning.analysis.grid_scan_spacing import summarize_marker_phase_sensitivity_for_grid_scan
from deflector_tuning.analysis.phase_advance import compute_phase_advance
from deflector_tuning.analysis.phase_radius_equivalence import build_experiment_simulation_phase_comparison
from deflector_tuning.markers.frequency_markers import extract_marker_frequencies
from deflector_tuning.analysis.kyhl_validation import summarize_no_plunger_phase_residuals


def test_central_difference_preserves_adjacent_phase_branches():
    table = pd.DataFrame([dict(data_kind="sim", source_file=str(r), marker_name="f_mean",
                              s_phase_deg=p, sim_r_c=r, sim_w_c=1.0)
                          for r, p in [(1, -170), (2, 0), (3, 170)]])
    result = summarize_marker_phase_sensitivity_for_grid_scan(table)
    np.testing.assert_allclose(result.dphase_d_sim_r_c_deg_per_mm, 170)


def test_missing_period_is_not_a_local_phase_advance():
    rows = [dict(tune_position=x, s_phase_deg=p, source_file=str(x),
                 freq_target_ghz=2.8, freq_ghz=2.8, s_db=-1)
            for x, p in [(0.5, 0), (2.5, 120), (3.5, 0)]]
    with pytest.warns(UserWarning, match="nonlocal"):
        result = compute_phase_advance(pd.DataFrame(rows))
    assert list(zip(result.pos_from, result.pos_to)) == [(2.5, 3.5)]
    assert result.phase_adv_deg.iloc[0] == 240


def test_local_phase_advance_is_invariant_to_input_row_order():
    rows = pd.DataFrame([dict(tune_position=x, s_phase_deg=p, source_file=str(x),
                             freq_target_ghz=2.8, freq_ghz=2.8, s_db=-1)
                         for x, p in [(0.5, 0), (1.5, -120), (2.5, 120)]])
    pd.testing.assert_frame_equal(compute_phase_advance(rows), compute_phase_advance(rows.iloc[::-1]))


@pytest.mark.parametrize("before,current", [(0, 1.5), (3, 1.5), (1.5, 0), (1.5, 3), (np.nan, 1.5)])
def test_both_candidate_radii_require_full_range(before, current):
    curve = pd.DataFrame(dict(sim_r_c=[1, 2], marker_name=["m", "m"], s_phase_deg=[0, 10]))
    observations = pd.DataFrame([dict(marker_name="m", before_phase_deg=0, after_phase_deg=1)])
    with pytest.raises(ValueError, match="outside"):
        build_experiment_simulation_phase_comparison(curve, observations, before_r_c_mm=before, current_r_c_mm=current)


def test_duplicate_radius_cannot_silently_select_another_curve():
    curve = pd.DataFrame(dict(sim_r_c=[1, 2, 2], marker_name=["m"] * 3, s_phase_deg=[0, 10, 50]))
    observations = pd.DataFrame([dict(marker_name="m", before_phase_deg=0, after_phase_deg=1)])
    with pytest.raises(ValueError, match="duplicate"):
        build_experiment_simulation_phase_comparison(curve, observations, before_r_c_mm=1, current_r_c_mm=2)


def test_dispersion_requires_explicit_summary_when_ambiguous(tmp_path):
    for name, freq in [("a_summary.csv", 2.1), ("b_summary.csv", 2.8)]:
        (tmp_path / name).write_text(f"mode_index,freq_90_GHz,freq_120_GHz\n1,{freq},{freq}\n")
    with pytest.raises(ValueError, match="Multiple"):
        extract_marker_frequencies(tmp_path)
    result = extract_marker_frequencies(tmp_path / "b_summary.csv")
    np.testing.assert_allclose(result.freq_ghz, 2.8)


def test_phase_summary_preserves_and_separates_provenance():
    residuals = pd.DataFrame([dict(dataset_id=d, source=s, ideal_slope_deg_per_half_cell=60,
                                   phase_residual_deg=e)
                              for d, s, e in [("custom", "phase.txt", 3), ("other", "other.txt", 4)]])
    result = summarize_no_plunger_phase_residuals(residuals)
    assert list(result.dataset_id) == ["custom", "other"]
    assert list(result.source) == ["phase.txt", "other.txt"]
    assert list(result.rms_phase_error_deg) == [3, 4]


def test_validation_field_energy_adapter_preserves_csv_and_analytical_ratios():
    from deflector_tuning.analysis.kyhl_validation import build_field_energy_pairs, summarize_field_energy_pairs
    from deflector_tuning.data_loading.field_profiles import FieldProfileExport, FieldProfileTrace
    # Nine cells, alternating iris amplitude 2 and cell amplitude 1: ratio = 4.
    export = FieldProfileExport(parameters={"d": 4.0, "t": 2.0}, traces=(
        FieldProfileTrace(label="E", field_kind="e", component="y", value_kind="real",
                          z_mm=np.arange(1, 56, 3), values=np.tile([2., 1.], 10)[:19]),
    ))
    pairs = build_field_energy_pairs(export)
    assert list(pairs.columns) == [
        "pair_index", "cell_label", "upstream_iris_label", "downstream_iris_label",
        "cell_center_z_mm", "upstream_iris_z_mm", "downstream_iris_z_mm",
        "E_cell", "E_upstream_iris", "E_downstream_iris", "E_cell_squared",
        "E_upstream_iris_squared", "E_downstream_iris_squared", "ratio_upstream_iris_to_cell",
        "ratio_downstream_iris_to_cell", "ratio_adjacent_average_iris_to_cell", "ratio_adjacent_max_iris_to_cell",
    ]
    summary = summarize_field_energy_pairs(pairs)
    np.testing.assert_allclose(summary.mean_ratio, 4)
    np.testing.assert_allclose(summary.std_ratio, 0)
    np.testing.assert_array_equal(summary.pair_count, 9)


def test_phase_points_receive_true_provenance_and_recover_ideal_slope():
    from deflector_tuning.data_loading.field_profiles import FieldPhaseExport, FieldPhaseTrace
    from deflector_tuning.analysis.kyhl_validation import build_no_plunger_phase_points, build_no_plunger_phase_residuals
    z = np.arange(1, 56, 3)
    phase = (60 * np.arange(19) + 180) % 360 - 180
    export = FieldPhaseExport(parameters={"d": 4., "t": 2.}, traces=(
        FieldPhaseTrace(label="E", field_kind="e", component="y", z_mm=z, phase_deg=phase),
    ))
    points = build_no_plunger_phase_points(export, dataset_id="custom", source="true_phase.txt")
    summary = summarize_no_plunger_phase_residuals(build_no_plunger_phase_residuals(points))
    assert set(summary.dataset_id) == {"custom"}
    assert set(summary.source) == {"true_phase.txt"}
    assert summary.loc[summary.ideal_slope_deg_per_half_cell == 60, "rms_phase_error_deg"].iloc[0] == pytest.approx(0, abs=1e-10)
    assert build_no_plunger_phase_points(export).dataset_id.isna().all()
