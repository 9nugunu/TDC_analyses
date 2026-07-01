from pathlib import Path
import json

import pandas as pd

import deflector_tuning.runner as runner


def _tables() -> dict[str, pd.DataFrame]:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_mean",
                "freq_ghz": 2.866,
                "s_db": -2.0,
                "s_phase_deg": 60.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run1.s2p",
                "marker_name": "f_pi2",
                "freq_ghz": 2.876,
                "s_db": -3.0,
                "s_phase_deg": 120.0,
                "tune_position": 0.5,
                "sim_r_c": 1.0,
                "sim_w_c": 2.0,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_mean",
                "freq_ghz": 2.866,
                "s_db": -2.0,
                "s_phase_deg": 70.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "sim",
                "source_file": "run2.s2p",
                "marker_name": "f_pi2",
                "freq_ghz": 2.876,
                "s_db": -3.0,
                "s_phase_deg": 140.0,
                "tune_position": 0.5,
                "sim_r_c": 1.5,
                "sim_w_c": 2.5,
            },
        ]
    )
    return {
        "markers": pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
        "marker_points": marker_points,
        "phase_advance": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "from_tune_position": 0.5,
                    "to_tune_position": 1.5,
                    "phase_advance_0to360_deg": 240.0,
                    "phase_error_from_240_deg": 0.0,
                }
            ]
        ),
        "phase_summary": pd.DataFrame([{"marker_name": "f_2pi3", "transition_count": 1}]),
        "nodal_shift": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "from_tune_position": 0.5,
                    "to_tune_position": 1.5,
                    "position_family": "cell",
                    "phase_error_from_target_deg": 0.0,
                    "abs_phase_error_from_target_deg": 0.0,
                }
            ]
        ),
    }


def _cell_iris_response_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "transition_pair_index": 1,
                "cell_from_tune_position": 0.5,
                "cell_to_tune_position": 1.5,
                "iris_from_tune_position": 1.0,
                "iris_to_tune_position": 2.0,
                "cell_operation_scaled_admittance_delta_abs": 5.0,
                "iris_operation_scaled_admittance_delta_abs": 10.0,
                "operation_scaled_admittance_response_ratio_iris_over_cell": 2.0,
                "cell_signed_phase_step_deg": -100.0,
                "iris_signed_phase_step_deg": -150.0,
                "phase_step_response_ratio_iris_over_cell": 1.5,
                "cell_phase_residual_from_target_deg": 20.0,
                "iris_phase_residual_from_target_deg": 5.0,
                "cell_abs_operation_axis_error_deg": 15.0,
                "iris_abs_operation_axis_error_deg": 5.0,
            }
        ]
    )


def _coupler_cavity_parameter_estimates_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "coupler_frequency_ghz": 2.866,
                "external_quality_factor": 50.0,
                "coupling_beta": 0.95,
                "target_external_quality_factor": 57.7,
                "is_valid": True,
            }
        ]
    )


def test_run_folder_analysis_saves_tables_figures_sim_260526_grid_scan_and_manifest(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        runner,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result.output_dir == output_dir
    assert "grid_scan_spacing" in result.analysis_modes
    assert result.manifest_path.exists()
    assert "sparameter_data" not in result.tables
    assert not (output_dir / "tables" / "sparameter_data.csv").exists()
    assert result.figures["s11"]["overview"].exists()
    assert result.figures["phase_advance"]["overview"].exists()
    assert result.figures["nodal_shift"]["overview"].exists()
    assert result.figures["polar"]["overview"].exists()
    assert result.figures["grid_scan_spacing"]["overview"].exists()
    assert result.figures["grid_scan_sparameter_phase_r_c_line_scan"]["overview"].exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"grid_scan_spacing"' in manifest
    assert '"sparameter_data"' not in manifest
    assert '"enabled": true' in manifest


def test_detect_analysis_modes_skips_sim_260526_grid_scan_for_experiment_marker_points() -> None:
    tables = _tables()
    tables["marker_points"] = tables["marker_points"].assign(data_kind="experiment")

    detected = runner.detect_analysis_modes(tables)

    assert "marker_analysis" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_uses_dataset_category_as_grid_gate() -> None:
    assert "grid_scan_spacing" in runner.detect_analysis_modes(_tables(), dataset_category="grid")
    assert "grid_scan_spacing" not in runner.detect_analysis_modes(_tables(), dataset_category="sweep")


def test_detect_analysis_modes_enables_geometry_phase_response_when_table_has_rows() -> None:
    tables = _tables()
    tables["geometry_phase_response"] = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "sweep_axis": "sim_offset_cell_03",
                "sweep_value": 0.0,
                "phase_delta_shift_from_baseline_deg": 0.0,
            }
        ]
    )

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "geometry_phase_response" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_enables_cell_iris_response_when_comparison_table_has_rows() -> None:
    tables = _tables()
    tables["cell_iris_response_comparison"] = _cell_iris_response_comparison_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "cell_iris_response" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_enables_coupler_cavity_parameters_when_table_has_rows() -> None:
    tables = _tables()
    tables["coupler_cavity_parameter_estimates"] = _coupler_cavity_parameter_estimates_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "coupler_cavity_parameters" in detected
    assert "grid_scan_spacing" not in detected


def test_run_folder_analysis_renders_coupler_cavity_parameters_when_available(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["coupler_cavity_parameter_estimates"] = _coupler_cavity_parameter_estimates_table()
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_coupler_cavity_parameters", fake_plot("coupler_cavity_parameters"))

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_sweep_260620_case",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "coupler_cavity_parameters" in result.analysis_modes
    assert result.figures["coupler_cavity_parameters"]["overview"].exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"coupler_cavity_parameters"' in manifest


def test_run_folder_analysis_renders_cell_iris_response_when_available(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["cell_iris_response_comparison"] = _cell_iris_response_comparison_table()
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_cell_iris_response_comparison", fake_plot("cell_iris_response"))

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_sweep_260620_case",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "cell_iris_response" in result.analysis_modes
    assert result.figures["cell_iris_response"]["overview"].exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"cell_iris_response"' in manifest


def test_resolve_input_paths_uses_data_root_and_default_dispersion(tmp_path: Path) -> None:
    sparameter_path, dispersion_path = runner.resolve_input_paths(
        "prepro/prepro_sweep_260415_sample_prepro",
        data_root=tmp_path / "data",
    )

    assert sparameter_path == tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    assert dispersion_path == tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"


def test_run_folder_analysis_uses_short_dispersion_figure_names(tmp_path: Path, monkeypatch) -> None:
    dispersion_folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_case"
    dispersion_folder.mkdir(parents=True)
    (dispersion_folder / "phase_sweep.txt").write_text(
        "\n".join(
            [
                "#",
                '#"phase"\t"Mode 1 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.10",
                "90\t2.90",
                "120\t2.86",
                "180\t2.84",
                "#",
                '#"phase"\t"Mode 2 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.30",
                "90\t3.10",
                "120\t3.05",
                "180\t3.00",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("dispersion input must not use S-parameter marker analysis")

    monkeypatch.setattr(runner, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_s11_with_markers", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_phase_advance", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_nodal_shift", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("dispersion",)
    assert "dispersion" in result.figures
    assert result.figures["dispersion"]["all_modes"].exists()
    assert set(result.figures["dispersion"]) == {
        "all_modes",
        "mode_01",
        "mode_02",
    }
    assert result.figures["dispersion"]["all_modes"].name == "all_modes.png"
    assert result.figures["dispersion"]["mode_01"].name == "mode_01.png"
    assert result.figures["dispersion"]["mode_02"].name == "mode_02.png"
    assert set(result.tables) == {"phase_sweep_long", "phase_sweep_wide", "phase_sweep_summary"}
    summary = pd.read_csv(result.tables["phase_sweep_summary"])
    assert summary.loc[0, "freq_120_GHz"] == 2.86
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"dispersion"' in manifest
    assert '"s11"' not in manifest
    assert '"phase_advance"' not in manifest


def test_run_folder_analysis_uses_dispersion_only_lane_for_single_mode_cst_exports(
    tmp_path: Path, monkeypatch
) -> None:
    dispersion_folder = tmp_path / "data" / "sim" / "dispersion_case"
    dispersion_folder.mkdir(parents=True)
    (dispersion_folder / "phase_sweep.txt").write_text(
        "\n".join(
            [
                "#",
                '#"phase"\t"Mode 1 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.10",
                "90\t2.90",
                "120\t2.86",
                "180\t2.84",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("dispersion input must not use S-parameter marker analysis")

    monkeypatch.setattr(runner, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_s11_with_markers", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_phase_advance", fail_sparameter_lane)
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("dispersion",)
    assert "dispersion" in result.figures
    assert result.figures["dispersion"]["all_modes"].exists()
    assert set(result.figures["dispersion"]) == {"all_modes", "mode_01"}
    assert result.figures["dispersion"]["all_modes"].name == "all_modes.png"
    assert result.figures["dispersion"]["mode_01"].name == "mode_01.png"
    assert set(result.tables) == {"phase_sweep_long", "phase_sweep_wide", "phase_sweep_summary"}
    summary = pd.read_csv(result.tables["phase_sweep_summary"])
    assert summary.loc[0, "freq_120_GHz"] == 2.86
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"dispersion"' in manifest
    assert '"s11"' not in manifest
    assert '"phase_advance"' not in manifest


def test_run_folder_analysis_uses_profile_only_lane_for_cst_z_profiles(tmp_path: Path, monkeypatch) -> None:
    profile_folder = tmp_path / "data" / "sim" / "sim_profile_260618_hem11_eh_zcut"
    profile_folder.mkdir(parents=True)
    (profile_folder / "E_fieldDist.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z) [Real]"',
                "#----------------------------------------------",
                "0.0\t1.0",
                "2.92\t3.0",
                "20.414\t0.5",
            ]
        ),
        encoding="utf-8",
    )
    (profile_folder / "EM_fieldPhase.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z)_phase [Real]"',
                "#----------------------------------------------",
                "0.0\t0.0",
                "2.92\t60.0",
                "20.414\t120.0",
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z)_phase [Real]"',
                "#----------------------------------------------",
                "0.0\t-20.0",
                "2.92\t40.0",
                "20.414\t100.0",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("profile input must not use S-parameter marker analysis")

    monkeypatch.setattr(runner, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("profile",)
    assert set(result.tables) == {
        "profile_summary",
        "field_energy_ratio_pairs",
        "field_energy_ratio_summary",
    }
    assert set(result.figures) == {"profile"}
    assert set(result.figures["profile"]) == {"E_fieldDist", "EM_fieldPhase", "E_fieldPhase", "H_fieldPhase"}
    assert result.figures["profile"]["E_fieldDist"].exists()
    assert result.figures["profile"]["E_fieldPhase"].exists()
    assert result.figures["profile"]["H_fieldPhase"].exists()
    summary = pd.read_csv(result.tables["profile_summary"])
    assert summary["source_file"].tolist() == ["E_fieldDist.txt", "EM_fieldPhase.txt", "EM_fieldPhase.txt"]
    assert summary["value_kind"].tolist() == ["real", "phase", "phase"]
    field_ratios = pd.read_csv(result.tables["field_energy_ratio_pairs"])
    assert "adjacent_average_iris_to_cell_energy_ratio" in field_ratios.columns
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"profile"' in manifest
    assert '"s11"' not in manifest


def test_run_folder_analysis_detects_legacy_named_cst_z_profiles(tmp_path: Path, monkeypatch) -> None:
    profile_folder = tmp_path / "data" / "sim" / "sim_EMfield_260618_PhaseDistribution"
    profile_folder.mkdir(parents=True)
    (profile_folder / "H_fieldDist.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z) [Real]"',
                "#----------------------------------------------",
                "0.0\t1.0",
                "2.92\t3.0",
                "20.414\t0.5",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("parseable profile input must not use S-parameter marker analysis")

    monkeypatch.setattr(runner, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("profile",)
    assert set(result.figures["profile"]) == {"H_fieldDist"}


def test_run_folder_analysis_skips_phase_plot_when_phase_table_is_empty(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["phase_advance"] = tables["phase_advance"].iloc[0:0]
    tables["nodal_shift"] = tables["nodal_shift"].iloc[0:0]
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    def fail_phase_plot(*args, **kwargs):
        raise AssertionError("empty phase_advance should not be plotted")

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fail_phase_plot)
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        runner,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "phase_advance" not in result.figures
    assert "nodal_shift" not in result.figures
    assert result.figures["s11"]["overview"].exists()
    assert result.figures["polar"]["overview"].exists()


def test_run_folder_analysis_reuses_existing_grid_s11_figures_from_manifest(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    output_dir = tmp_path / "out"
    cached_s11 = output_dir / "figures" / "s11" / "s11_cached.png"
    cached_s11.parent.mkdir(parents=True)
    cached_s11.write_text("cached", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({"outputs": {"figures": {"s11": {"cached": str(cached_s11)}}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)

    def fail_early_s11_load(self, path):
        raise AssertionError("cached grid S11 figures should be checked before loading S-parameter data")

    monkeypatch.setattr(runner.DataLoader, "load", fail_early_s11_load)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fail_s11_plot(*args, **kwargs):
        raise AssertionError("cached grid S11 figures should be reused")

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fail_s11_plot)
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        runner,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result.figures["s11"]["cached"] == cached_s11


def test_run_folder_analysis_logs_progress_steps(tmp_path: Path, monkeypatch, caplog) -> None:
    tables = _tables()
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        runner,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    with caplog.at_level("INFO", logger="deflector_tuning.runner"):
        runner.run_folder_analysis(
            sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
            dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
            output_dir=output_dir,
            marker_role="sim",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting folder analysis" in message for message in messages)
    assert any("Loading S-parameter table" in message for message in messages)
    assert any("Building marker analysis tables" in message for message in messages)
    assert any("Rendering S11 figures" in message for message in messages)
    assert any("Writing manifest" in message for message in messages)
    assert any("Folder analysis completed successfully" in message for message in messages)


def test_run_folder_analysis_does_not_save_sparameter_data_csv(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "s_name": "S11",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            },
            {
                "source_file": "run1.s2p",
                "s_name": "S21",
                "tune_position": 0.5,
                "freq_ghz": 2.856,
                "s_db": -20.0,
                "s_phase_deg": 30.0,
            },
        ]
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            folder = Path(args[2] if name == "s11" else args[1])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        runner,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "sparameter_data" not in result.tables
    assert not (output_dir / "tables" / "sparameter_data.csv").exists()
