"""Check directed phase, exported-sample selection, and pair compatibility."""

import numpy as np
import pytest

from scripts.build_paper_cst_bundle import (
    nearest_sample_index,
    phase_pair_metrics,
    radius_phase_response,
    validate_pair_metadata,
)


@pytest.mark.parametrize(
    "first_deg,second_deg,target_deg,delta_deg,residual_deg",
    [(170, -70, 120, 120, 0), (-179, 179, 0, 358, -2),
     (0, -120, 240, 240, 0), (0, 180, 0, 180, -180),
     (30, 29, 120, 359, -121)],
)
def test_directed_phase_and_target_branch_wrap(
    first_deg, second_deg, target_deg, delta_deg, residual_deg
):
    first = 0.8 * np.exp(1j * np.deg2rad(first_deg))
    second = 1.1 * np.exp(1j * np.deg2rad(second_deg))
    result = phase_pair_metrics(first, second, target_deg)
    assert result["delta_deg"] == pytest.approx(delta_deg)
    assert result["signed_target_residual_deg"] == pytest.approx(residual_deg)
    assert result["first_modulus"] == pytest.approx(0.8)
    assert result["second_modulus"] == pytest.approx(1.1)


@pytest.mark.parametrize("first,second", [(0j, 1j), (1j, 0j), (complex(np.nan, 0), 1j)])
def test_undefined_phase_is_rejected(first, second):
    with pytest.raises(ValueError, match="finite and nonzero"):
        phase_pair_metrics(first, second, 180)


def test_nearest_exported_sample_and_deterministic_tie():
    assert nearest_sample_index(np.array([2.85, 2.86, 2.87]), 2.8571160335532) == 1
    assert nearest_sample_index(np.array([1.0, 3.0]), 2.0) == 0


@pytest.mark.parametrize("frequencies,target", [([2, 1], 1.5), ([1, 2], 3), ([], 1), ([1, np.nan], 1)])
def test_invalid_frequency_grid_or_out_of_range_marker_is_rejected(frequencies, target):
    with pytest.raises(ValueError):
        nearest_sample_index(np.array(frequencies), target)


def _metadata(depth, plunger):
    return {"project": "case.cst", "option_line": "# GHz S RI R 0",
            "port_assignments": ["! Touchstone port 1 = CST MWS port 2 (\"\")"],
            "parameters": {"NumDepth": depth, "DepthPlunger": plunger,
                           "R_plunger": 8, "NumCell": 9, "Nmesh": 180}}


def test_pair_metadata_allows_only_declared_short_state_changes():
    result = validate_pair_metadata(_metadata(0.5, 100), _metadata(1.5, 65),
                                    ["NumDepth", "DepthPlunger"])
    assert result["consistent"] is True
    assert set(result["varying_short_state_fields"]) == {"NumDepth", "DepthPlunger"}


@pytest.mark.parametrize("changed_field", ["R_plunger", "Nmesh", "new_geometry_parameter"])
def test_pair_metadata_exposes_geometry_mismatch(changed_field):
    second = _metadata(1.5, 65)
    second["parameters"][changed_field] = 999
    with pytest.raises(ValueError, match=changed_field):
        validate_pair_metadata(_metadata(0.5, 100), second, ["NumDepth", "DepthPlunger"])


def test_pair_metadata_exposes_changed_port_assignment():
    second = _metadata(1.5, 65)
    second["port_assignments"] = ["! Touchstone port 1 = CST MWS port 1 (\"\")"]
    with pytest.raises(ValueError, match="port_assignments"):
        validate_pair_metadata(_metadata(0.5, 100), second, ["NumDepth", "DepthPlunger"])


def test_radius_unwrap_preserves_nominal_raw_phase_and_centered_slope():
    result = radius_phase_response(np.array([56.58, 56.59, 56.60]),
                                   np.array([170.0, 179.0, -172.0]), 56.59)
    assert result["unwrapped_phase_deg"] == pytest.approx([170, 179, 188])
    assert result["nominal_central_slope_deg_per_mm"] == pytest.approx(900)
    assert result["branch_turns"] == [0, 0, 1]


def test_radius_unwrap_can_anchor_negative_nominal_branch():
    result = radius_phase_response(np.array([1, 2, 3]), np.array([179, -172, -163]), 2)
    assert result["unwrapped_phase_deg"] == pytest.approx([-181, -172, -163])
    assert result["nominal_central_slope_deg_per_mm"] == pytest.approx(9)


def test_radius_requires_a_centered_nominal_point():
    with pytest.raises(ValueError, match="interior"):
        radius_phase_response(np.array([1, 2, 3]), np.array([0, 5, 10]), 1)


def test_nominal_geometry_link_can_record_different_project_names_explicitly():
    first, second = _metadata(2, 37.908), _metadata(2, 37.908)
    second["project"] = "radius_scan.cst"
    result = validate_pair_metadata(first, second, [], require_same_project=False)
    assert result["consistent"]
    assert result["different_project_names"] == ["case.cst", "radius_scan.cst"]


def test_port_transform_source_map_locates_and_verifies_the_raw_export(tmp_path):
    import hashlib

    from scripts.build_paper_cst_bundle import original_export_path

    raw = tmp_path / "data/sim/sim_sweep_000000_batch/r01-run.s2p"
    raw.parent.mkdir(parents=True)
    raw.write_text("raw", encoding="utf-8")
    delivery = tmp_path / "data/sim/sim_sweep_000000_delivery"
    delivery.mkdir()
    digest = hashlib.sha256(b"raw").hexdigest()
    (delivery / "source_map.csv").write_text(
        "delivered_file,source_file,source_sha256\n"
        f"run_1.s2p,data/sim/sim_sweep_000000_batch/r01-run.s2p,{digest}\n", encoding="utf-8")
    transform = {"source_map": "source_map.csv", "column_permutation": [3, 2, 1, 0]}
    assert original_export_path(tmp_path, delivery, transform, delivery / "run_1.s2p") == raw.resolve()
    raw.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        original_export_path(tmp_path, delivery, transform, delivery / "run_1.s2p")


def test_periodic_diagnostic_source_comes_from_the_configured_case():
    import json
    from pathlib import Path

    from scripts.build_paper_periodic_diagnostic import DEFAULT_CONFIG, source_case

    config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    case = source_case(config)
    assert case["id"] == config["periodic_diagnostic"]["case"]
    assert case["dataset"].startswith("data/sim/")


@pytest.mark.parametrize("unit,scale", [("Hz", 1e9), ("kHz", 1e6), ("MHz", 1e3), ("GHz", 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_complex_sampling_respects_units_and_input_order(unit, scale, reverse):
    from deflector_tuning.analysis.complex_sampling import sample_complex_frequency
    grid, values = np.array([2., 3.]) * scale, np.array([1., 1j])
    if reverse:
        grid, values = grid[::-1], values[::-1]
    actual = sample_complex_frequency(grid, values, [2.5], frequency_unit=unit)
    assert actual == pytest.approx([.5 + .5j])


@pytest.mark.parametrize("grid,values,target", [([2,3], [1,1j], [4]), ([2,2], [1,1j], [2]),
                                               ([2,3], [1,np.nan], [2.5]), ([2,3], [1,1j], [np.nan])])
def test_complex_sampling_rejects_ambiguous_or_invalid_samples(grid, values, target):
    from deflector_tuning.analysis.complex_sampling import sample_complex_frequency
    with pytest.raises(ValueError):
        sample_complex_frequency(grid, values, target, frequency_unit="GHz")


def test_width_series_cannot_silently_keep_last_trace():
    import pandas as pd
    from scripts.plot_wc_absolute_s11_phase import build_wc_absolute_phase_table
    rows = pd.DataFrame({"marker_name": ["f_2pi3"] * 2, "sim_NumDepth": [2.] * 2,
                         "sim_r_c": [56.] * 2, "sim_w_c": [19.] * 2,
                         "s_phase_deg": [10., 80.], "source_file": ["a", "b"]})
    for table in (rows, rows.iloc[::-1]):
        with pytest.raises(ValueError, match="Multiple traces"):
            build_wc_absolute_phase_table(table, fixed_r_c_mm=56)
        selected = build_wc_absolute_phase_table(table, fixed_r_c_mm=56, series_filters={"source_file": "a"})
        assert selected.s_phase_deg.tolist() == [10.]


def test_periodic_parameters_use_selected_config_and_explicit_overrides(monkeypatch):
    import scripts.build_paper_periodic_diagnostic as diagnostic
    config = {"markers": [{"name": n} for n in diagnostic.MARKER_NAMES],
              "cases": [{"id": "custom", "expected_parameters": {"NumTune": 7}}],
              "periodic_diagnostic": {"case": "custom"}}
    monkeypatch.setattr(diagnostic, "resolve_markers", lambda *args: ([{"source_derived_GHz": f} for f in [1., 2., 3.]], {}))
    assert diagnostic.resolve_diagnostic_settings(config, None, None) == ([1., 2., 3.], 7.)
    assert diagnostic.resolve_diagnostic_settings(config, [4., 5., 6.], 9) == ([4., 5., 6.], 9.)
    config["markers"].reverse()
    with pytest.raises(ValueError, match="marker order"):
        diagnostic.resolve_diagnostic_settings(config, None, None)


def test_load_case_rejects_renormalized_source(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace
    import scripts.build_paper_cst_bundle as bundle
    folder = tmp_path / "data/sim/example"
    folder.mkdir(parents=True)
    (folder / "one.s1p").touch()
    monkeypatch.setattr(bundle, "read_cst_metadata", lambda p: {"parameters": {"NumDepth": 1.}})
    monkeypatch.setattr(bundle, "read_touchstone", lambda p: SimpleNamespace(header=SimpleNamespace(parameter="S", reference_ohm=50)))
    with pytest.raises(ValueError, match="R 0"):
        bundle.load_case(tmp_path, {"id": "example", "dataset": "data/sim/example", "expected_parameters": {}}, [1.])


def test_paper_marker_labels_follow_marker_identity():
    from deflector_tuning.workflows.paper_plot_contracts import marker_labels, MARKER_LABELS
    names = ["f_pi2", "f_2pi3", "f_mean"]
    assert marker_labels({"markers": [{"name": n} for n in names]}) == [MARKER_LABELS[n] for n in names]
    with pytest.raises(ValueError):
        marker_labels({"markers": [{"name": "f_mean"}]})


def test_publication_source_check_survives_optimized_python(tmp_path):
    # Execute the actual hash-validation branch extracted from the script under -O.
    # No publication files or rendering are touched by this contract probe.
    import ast
    import subprocess
    import sys
    from pathlib import Path
    source = Path("scripts/build_paper_polar_comparison.py").read_text()
    tree = ast.parse(source)
    check = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and "source_hashes[path.relative_to(ROOT).as_posix()]" in ast.unparse(n.test))
    probe = "from pathlib import Path\nROOT=Path('.')\npath=Path('trace.s1p')\nsource_hashes={'trace.s1p':'changed'}\npoint={'first_sha256':'original'}\nstate='first'\n" + ast.unparse(check)
    result = subprocess.run([sys.executable, "-B", "-O", "-c", probe], capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr


@pytest.mark.parametrize("unit,scale", [("Hz", 1e9), ("GHz", 1.)])
def test_experiment_plot_sampling_uses_touchstone_units(tmp_path, unit, scale):
    import pandas as pd
    from scripts.plot_tuning_radius_equivalence import _sample_s11
    from scripts.plot_tuning_position_on_sim_rc_line import _sample_s11_phase
    path = tmp_path / "trace.s1p"
    path.write_text(f"# {unit} S RI R 50\n{2*scale} 1 0\n{3*scale} 0 1\n")
    markers = pd.DataFrame({"marker_name": ["custom"], "freq_ghz": [2.5]})
    assert _sample_s11(path, markers).s11.iloc[0] == pytest.approx(.5+.5j)
    assert _sample_s11_phase(path, markers).s11_phase_deg.iloc[0] == pytest.approx(45)
    markers["freq_ghz"] = 4.
    for sample in (_sample_s11, _sample_s11_phase):
        with pytest.raises(ValueError, match="outside"):
            sample(path, markers)


def test_width_plot_cli_selects_canonical_source_file(tmp_path, monkeypatch):
    import pandas as pd
    import sys
    import scripts.plot_wc_absolute_s11_phase as script
    source = tmp_path / "markers.csv"
    pd.DataFrame({"marker_name": ["f_2pi3"] * 2, "sim_NumDepth": [2.] * 2,
                  "sim_r_c": [56.] * 2, "sim_w_c": [19.] * 2,
                  "s_phase_deg": [10., 80.], "source_file": ["a.s2p", "b.s2p"]}).to_csv(source, index=False)
    captured = []
    monkeypatch.setattr(script, "plot_wc_absolute_phase", lambda table, *args, **kwargs: captured.append(table) or tmp_path / "plot.png")
    monkeypatch.setattr(sys, "argv", ["plot_wc_absolute_s11_phase", "--input-csv", str(source),
                                    "--output", str(tmp_path / "plot.png"), "--r-c-mm", "56", "--source", "a.s2p"])
    script.main()
    assert captured[0].source_file.tolist() == ["a.s2p"]
    assert captured[0].s_phase_deg.tolist() == [10.]


def test_configured_paper_display_labels_preserve_publication_names():
    import json
    from pathlib import Path
    from deflector_tuning.workflows.paper_plot_contracts import case_label
    config = json.loads(Path("config/paper_cst_20261003.json").read_text())
    assert {case["id"]: case_label(case) for case in config["cases"]} == {"full": "Extended", "half": "Reduced"}


@pytest.mark.parametrize("bounds", [(2.9, 3.0), (3.3, 3.4)])
@pytest.mark.parametrize("unit,scale", [("Hz", 1e9), ("GHz", 1.)])
def test_complex_sampling_endpoints_allow_only_unit_roundoff(bounds, unit, scale):
    from deflector_tuning.analysis.complex_sampling import sample_complex_frequency
    grid = np.array(bounds) * scale
    actual = sample_complex_frequency(grid, [1, 1j], bounds, frequency_unit=unit)
    assert actual == pytest.approx([1, 1j])
    for target in (bounds[0] - 1e-12, bounds[1] + 1e-12):
        with pytest.raises(ValueError, match="outside"):
            sample_complex_frequency(grid, [1, 1j], [target], frequency_unit=unit)


@pytest.mark.parametrize("names", [("f_2pi3", "f_mean", "f_pi2"), ("f_pi2", "f_2pi3", "f_mean")])
def test_dispersion_marker_validation_uses_identity_before_rendering(tmp_path, monkeypatch, names):
    import pandas as pd
    import scripts.build_paper_field_comparison as script
    data = pd.DataFrame({"mode_index": [1] * 181, "phase_deg": np.arange(181),
                         "freq_GHz": 3.0 - np.arange(181) * .001})
    monkeypatch.setattr(script, "load_cst_dispersion_txt", lambda source: data)
    class RenderingReached(Exception):
        pass
    def stop_before_rendering(*args, **kwargs):
        raise RenderingReached()
    monkeypatch.setattr(script.plt, "subplots", stop_before_rendering)
    derived = {"f_2pi3": 2.88, "f_mean": 2.895, "f_pi2": 2.91}
    markers = [{"name": name, "requested_GHz": derived[name]} for name in names]
    config = {"dispersion_source": "synthetic.txt", "mode_index": 1}
    with pytest.raises(RenderingReached):
        script.dispersion_figure(tmp_path, tmp_path, config, markers)
    markers[0]["requested_GHz"] += .01
    with pytest.raises(AssertionError):
        script.dispersion_figure(tmp_path, tmp_path, config, markers)


@pytest.mark.parametrize("module_name,arguments,options", [
    ("scripts.plot_tuning_position_on_sim_rc_line", ["line.csv", "markers.csv", "before", "after", "--output-dir", "output"],
     [("--baseline-r-c-mm", "baseline_r_c_mm", 60.), ("--target-r-c-mm", "target_r_c_mm", 60.), ("--fixed-w-c-mm", "fixed_w_c_mm", 21.)]),
    ("scripts.plot_tuning_radius_equivalence", ["line.csv", "markers.csv", "before", "after", "--output-dir", "output"],
     [("--baseline-r-c-mm", "baseline_r_c_mm", 60.), ("--fixed-w-c-mm", "fixed_w_c_mm", 21.)]),
    ("scripts.plot_wc_absolute_s11_phase", ["--input-csv", "markers.csv", "--output", "plot.png"],
     [("--r-c-mm", "r_c_mm", 60.)]),
])
def test_scientific_cli_defaults_follow_project_and_allow_overrides(monkeypatch, module_name, arguments, options):
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "DEFAULT_PROJECT_DEFAULTS", SimpleNamespace(design_point_by_axis={"sim_r_c": 60., "sim_w_c": 21.}))
    parser = module.build_parser()
    parsed = parser.parse_args(arguments)
    for flag, attribute, expected in options:
        assert getattr(parsed, attribute) == expected
        explicit = parser.parse_args([*arguments, flag, "22.5"])
        assert getattr(explicit, attribute) == 22.5
    if module_name.endswith("position_on_sim_rc_line"):
        assert parsed.state_label == "Measured state"
        assert parser.parse_args([*arguments, "--state-label", "Custom state"]).state_label == "Custom state"
