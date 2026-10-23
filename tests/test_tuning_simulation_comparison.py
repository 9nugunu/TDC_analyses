from __future__ import annotations

from pathlib import Path

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


def test_anchor_check_does_not_treat_noportE_as_port_extended() -> None:
    table = pd.DataFrame(
        [
            {"source_file": "1_noportE.S2P", "tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": 10.0},
            {"source_file": "1_portE.S2P", "tune_position": 1.0, "marker_name": "f_mean", "s_phase_deg": -179.0},
        ]
    )

    check = build_anchor_check(
        table,
        position=1.0,
        expected_phase_deg=180.0,
        tolerance_deg=5.0,
        port_extension_applied=True,
    )

    assert check["source_file"].tolist() == ["1_portE.S2P"]


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


@pytest.mark.parametrize("explicit_reference", [False, True])
def test_reference_preparation_matches_direct_correction_and_invalidates_stale_inputs(monkeypatch, tmp_path, explicit_reference):
    from dataclasses import replace
    from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection
    from deflector_tuning.project_defaults import ProjectDefaults
    from deflector_tuning.workflows import dataset_execution, tuning_simulation_comparison as comparison

    data_root = tmp_path / "data"
    dataset = "raw_sweep_261023_reference"
    source = data_root / "raw" / dataset
    source.mkdir(parents=True)
    trace = source / "1_portE.s1p"
    trace.write_text("# GHz S DB R 50\n2.80 0.8 0\n2.85 0.8 50\n2.90 0.8 100\n2.95 0.8 150\n")
    reference = data_root / "reference"
    reference.mkdir()
    summary = reference / "summary.csv"
    summary.write_text("mode_index,freq_90_GHz,freq_120_GHz\n1,2.88,2.86\n")
    defaults = ProjectDefaults(default_dispersion_subpath=Path("reference"))
    explicit_csv = data_root / "override.csv"
    explicit_csv.write_text("mode_index,freq_90_GHz,freq_120_GHz\n1,2.91,2.89\n")
    selected_reference = explicit_csv if explicit_reference else None
    uncorrected_mean = 2.90 if explicit_reference else 2.87
    correction = TemperatureHumidityCorrection(temp_op_C=20, temp_meas_C=45)
    monkeypatch.setattr(dataset_execution, "resolve_tuning_marker_correction", lambda *a, **k: correction)
    monkeypatch.setattr(comparison, "resolve_tuning_marker_correction", lambda *a, **k: correction)
    monkeypatch.setattr(dataset_execution, "register_matching_tuning_campaign", lambda *a, **k: None)
    direct = dataset_execution.run_dataset_analysis(
        dataset_id=dataset, sparameter_path=source, output_dir=tmp_path / "direct" / dataset,
        marker_role="exp", data_root=data_root, tables_only=True, project_defaults=defaults,
        dispersion_path=selected_reference,
    )
    calls = []
    actual_runner = comparison.run_folder_analysis
    def tracked_runner(**kwargs):
        calls.append(kwargs)
        return actual_runner(**kwargs)
    monkeypatch.setattr(comparison, "run_folder_analysis", tracked_runner)
    options = dict(dataset=dataset, layer="raw", marker_role="exp", data_root=data_root,
                   analysis_root=tmp_path / "fallback", file_workers=1, plot_workers=1, project_defaults=defaults,
                   dispersion_path=selected_reference)
    fallback = comparison._ensure_marker_table(**options)
    pd.testing.assert_frame_equal(fallback, comparison.read_marker_table(direct.analysis.tables["marker_pts"]))
    assert calls[-1]["marker_correction"] == correction
    markers = pd.read_csv(tmp_path / "fallback" / dataset / "tables" / "markers.csv")
    assert markers.loc[markers.marker_name == "f_mean", "freq_ghz"].iloc[0] == pytest.approx(uncorrected_mean * correction.frequency_scale_factor)
    comparison._ensure_marker_table(**options)
    assert len(calls) == 1
    # A direct run uses the same signature and is immediately reusable.
    comparison._ensure_marker_table(**{**options, "analysis_root": tmp_path / "direct"})
    assert len(calls) == 1
    trace.write_text(trace.read_text().replace("0.8 50", "0.8 55").replace("0.8 100", "0.8 105"))
    changed = comparison._ensure_marker_table(**options)
    assert len(calls) == 2
    assert not changed.s_phase_deg.equals(fallback.s_phase_deg)
    correction = replace(correction, temp_meas_C=30)
    comparison._ensure_marker_table(**options)
    assert len(calls) == 3
    active_summary = explicit_csv if explicit_reference else summary
    active_summary.write_text(active_summary.read_text().replace("2.91", "2.92").replace("2.88", "2.89"))
    comparison._ensure_marker_table(**options)
    assert len(calls) == 4
    cached_path = tmp_path / "fallback" / dataset / "tables" / "marker_pts.csv"
    cached_path.write_text(cached_path.read_text() + "\n")
    comparison._ensure_marker_table(**options)
    assert len(calls) == 5



def test_simulation_reference_line_rejects_duplicate_geometry_or_port():
    table = pd.DataFrame([
        {"sim_r_c": 56.5, "marker_name": "f_mean", "s_phase_deg": 10, "s_parameter": "S11"},
        {"sim_r_c": 56.5, "marker_name": "f_mean", "s_phase_deg": 20, "s_parameter": "S22"},
    ])
    with pytest.raises(ValueError, match="Multiple simulation traces"):
        _simulation_rc_line(table, marker_names=["f_mean"])


def test_campaign_comparison_passes_reference_and_actual_width_to_all_products(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from collections import OrderedDict
    import json
    from deflector_tuning.table_schema import STANDARD_TABLE_SPECS
    from deflector_tuning.table_export import save_standard_tables
    from deflector_tuning.project_defaults import ProjectDefaults
    from deflector_tuning.workflows import tuning_simulation_comparison as workflow
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    marker_path = current_dir / "marker_pts.csv"
    markers = pd.DataFrame({"tune_position": [1., 2.], "sim_w_c": [20., 20.]})
    markers.to_csv(marker_path, index=False)
    state = SimpleNamespace(dataset="current", torque_nm=5)
    campaign = SimpleNamespace(
        baseline_state_id="baseline", states={"baseline": SimpleNamespace(dataset="baseline"), "current": state},
        simulation_references={"iris": SimulationReference(family="iris", dataset="simulation", experiment_positions=(1., 2.))},
        phase=PhaseReference(), design_r_c_mm=56.6,
    )
    match = SimpleNamespace(state_id="current", campaign=campaign)
    references, plots = [], []
    standard_tables = OrderedDict((key, pd.DataFrame({"dataset_id": ["simulation"]})) for key in STANDARD_TABLE_SPECS)
    standard_tables["marker_pts"] = markers
    saved = save_standard_tables(standard_tables, tmp_path / "simulation" / "tables")
    (tmp_path / "simulation" / "manifest.json").write_text(json.dumps({"table_constants": saved.constants}))
    assert "sim_w_c" not in pd.read_csv(saved.paths["marker_pts"])
    restored = workflow.read_marker_table(saved.paths["marker_pts"])
    assert restored.sim_w_c.tolist() == [20., 20.]
    monkeypatch.setattr(workflow, "_ensure_marker_table", lambda dataset, **kwargs: references.append((dataset, kwargs)) or restored)
    comparison = SimpleNamespace(anchor_check=markers, rc_fit=markers, phase_cmp=markers, simulation_line=markers)
    monkeypatch.setattr(workflow, "build_tuning_family_comparison", lambda *a, **k: comparison)
    monkeypatch.setattr(workflow, "plot_phase_rc_map", lambda *a, **k: plots.append(k) or a[2])
    monkeypatch.setattr(workflow, "plot_phase_cmp_bars", lambda *a, **k: a[1])
    monkeypatch.setattr(workflow, "register_tuning_cmp", lambda *a, **k: None)
    explicit = tmp_path / "selected.csv"
    defaults = ProjectDefaults(default_dispersion_subpath=Path("different_default"), design_point_by_axis={"sim_w_c": 19.3224})
    workflow.run_tuning_cmp(match, current_result=SimpleNamespace(output_dir=current_dir, tables={"marker_pts": marker_path}, manifest_path=current_dir / "manifest.json"),
                            data_root=tmp_path, project_defaults=defaults, dispersion_path=explicit)
    assert [dataset for dataset, _ in references] == ["baseline", "simulation"]
    assert all(options["dispersion_path"] == explicit for _, options in references)
    assert plots[0]["fixed_w_c"] == 20.
    assert plots[0]["config"].design_point_by_axis == defaults.design_point_by_axis


def test_simulation_radius_line_preserves_width_metadata():
    line = _simulation_rc_line(pd.DataFrame([
        {"sim_r_c": 56.5, "sim_w_c": 20., "marker_name": "f_mean", "s_phase_deg": 10.},
        {"sim_r_c": 56.6, "sim_w_c": 20., "marker_name": "f_mean", "s_phase_deg": 20.},
    ]), marker_names=["f_mean"])
    assert line.sim_w_c.tolist() == [20., 20.]


def test_reference_cache_signature_protects_projected_metadata(tmp_path):
    import json
    from deflector_tuning.workflows.reference_marker_cache import record_marker_signature, marker_signature_matches
    tables = tmp_path / "tables"
    tables.mkdir()
    marker = tables / "marker_pts.csv"
    marker.write_text("marker_name,s_phase_deg\nf_mean,10\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"table_constants": {"marker_pts": {"sim_w_c": 20.}}}))
    record_marker_signature(marker, "inputs")
    assert marker_signature_matches(marker, "inputs")
    manifest.write_text(json.dumps({"table_constants": {"marker_pts": {"sim_w_c": 19.}}}))
    assert not marker_signature_matches(marker, "inputs")
