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
        "marker_pts": marker_points,
        "phase_adv": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "pos_from": 0.5,
                    "pos_to": 1.5,
                    "phase_adv_deg": 240.0,
                    "phase_err_240_deg": 0.0,
                }
            ]
        ),
        "phase_stats": pd.DataFrame([{"marker_name": "f_2pi3", "n_steps": 1}]),
        "nodal_shift": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "pos_from": 0.5,
                    "pos_to": 1.5,
                    "position_family": "cell",
                    "phase_err_deg": 0.0,
                    "phase_err_abs_deg": 0.0,
                }
            ]
        ),
    }


def test_run_folder_analysis_loads_sparameter_folder_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sparameter_path = tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    dispersion_path = tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"
    sparameter_path.mkdir(parents=True)
    for tune_position, phases in [
        (0.5, [10.0, 20.0, 30.0]),
        (1.5, [-110.0, -100.0, -90.0]),
    ]:
        (sparameter_path / f"{tune_position}_processed.csv").write_text(
            "freq[Hz],Magnitude,Phase_deg\n"
            f"2855880000,-1.0,{phases[0]}\n"
            f"2866050000,-2.0,{phases[1]}\n"
            f"2876210000,-3.0,{phases[2]}\n",
            encoding="utf-8",
        )

    processed = dispersion_path / "processed"
    processed.mkdir(parents=True)
    (processed / "dispersion_summary.csv").write_text(
        "mode_index,freq_90_GHz,freq_120_GHz\n1,2.8778045932946,2.8574021866048\n",
        encoding="utf-8",
    )

    loader = runner.DataLoader()
    original_load = loader.load
    loaded_paths: list[Path] = []

    def tracked_load(path: str | Path) -> pd.DataFrame:
        loaded_paths.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(loader, "load", tracked_load)

    runner.run_folder_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        output_dir=tmp_path / "out",
        marker_role="exp",
        loader=loader,
    )

    assert loaded_paths == [sparameter_path], (
        f"expected one S-parameter folder load, got {len(loaded_paths)}: {loaded_paths}"
    )


def _cell_iris_response_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "pair_index": 1,
                "cell_pos_from": 0.5,
                "cell_pos_to": 1.5,
                "iris_pos_from": 1.0,
                "iris_pos_to": 2.0,
                "cell_admit_delta_mag": 5.0,
                "iris_admit_delta_mag": 10.0,
                "admit_ratio_iris_cell": 2.0,
                "cell_phase_step_deg": -100.0,
                "iris_phase_step_deg": -150.0,
                "phase_ratio_iris_cell": 1.5,
                "cell_phase_err_deg": 20.0,
                "iris_phase_err_deg": 5.0,
                "cell_admit_axis_err_abs_deg": 15.0,
                "iris_admit_axis_err_abs_deg": 5.0,
            }
        ]
    )


def _coupler_cavity_parameter_estimates_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file": "run1.s2p",
                "coupler_freq_ghz": 2.866,
                "q_ext": 50.0,
                "beta": 0.95,
                "q_ext_target": 57.7,
                "is_valid": True,
            }
        ]
    )


def _kyhl_admittance_points_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "op_admit_re": 1.0,
                "op_admit_im": 0.5,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 60.0,
                "op_admit_axis_deg": 60.0,
                "op_admit_axis_err_deg": 0.0,
            }
        ]
    )


def _kyhl_f2pi3_normalized_admittance_audit_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "source_file": "run1.s2p",
                "tune_position": 0.5,
                "mode_admit_re": 0.2,
                "mode_admit_im": -1.7,
                "mode_gamma_re": -0.5,
                "mode_gamma_im": -0.8660254,
                "mode_gamma_mag": 1.0,
                "mode_gamma_ang_deg": 240.0,
            }
        ]
    )


def _kyhl_admittance_transitions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "position_family": "cell",
                "pos_from": 0.5,
                "pos_to": 1.5,
                "op_admit_delta_re": 0.5,
                "op_admit_delta_im": -1.0,
                "op_mode_deg": 120.0,
                "op_admit_axes_deg": "60;180;300",
                "op_admit_ang_deg": 60.0,
                "op_admit_axis_deg": 60.0,
                "op_admit_axis_err_deg": 0.0,
            }
        ]
    )


def test_run_folder_analysis_saves_tables_figures_sim_260526_grid_scan_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
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
    s11_plot_kwargs: dict[str, object] = {}

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_plot(name):
        def _plot(*args, **kwargs):
            if name == "s11":
                s11_plot_kwargs.update(kwargs)
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
        plot_workers=3,
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
    assert s11_plot_kwargs["render_workers"] == 3
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"grid_scan_spacing"' in manifest
    assert '"sparameter_data"' not in manifest
    assert '"enabled": true' in manifest
    manifest_data = json.loads(manifest)
    assert manifest_data["table_schema_version"] == 2
    assert manifest_data["table_contract"] == "standard"
    assert manifest_data["table_constants"] == {}


def test_run_folder_analysis_tables_only_skips_plots_and_preserves_existing_figures(
    tmp_path: Path, monkeypatch
) -> None:
    tables = _tables()
    output_dir = tmp_path / "out"
    valid_figure = output_dir / "figures" / "s11" / "overview.png"
    valid_figure.parent.mkdir(parents=True)
    valid_figure.write_text("existing", encoding="utf-8")
    missing_figure = output_dir / "figures" / "polar" / "missing.png"
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "figures": {
                        "s11": {"overview": str(valid_figure)},
                        "polar": {"missing": str(missing_figure)},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {
            name: Path(output) / f"{name}.csv" for name in analysis_tables
        },
    )

    def fail_plot(*args, **kwargs):
        raise AssertionError("tables-only mode must not render figures")

    monkeypatch.setattr(runner, "_load_s11_table_for_figures", fail_plot)
    monkeypatch.setattr(runner, "plot_s11_with_markers", fail_plot)
    monkeypatch.setattr(runner, "plot_phase_advance", fail_plot)
    monkeypatch.setattr(runner, "plot_nodal_shift", fail_plot)
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fail_plot)

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
        tables_only=True,
    )

    assert result.figures == {"s11": {"overview": valid_figure}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["figures"] == {
        "s11": {"overview": str(valid_figure)}
    }
    assert manifest["table_schema_version"] == 2
    assert manifest["table_contract"] == "standard"


def test_detect_analysis_modes_skips_sim_260526_grid_scan_for_experiment_marker_points() -> None:
    tables = _tables()
    tables["marker_pts"] = tables["marker_pts"].assign(data_kind="experiment")

    detected = runner.detect_analysis_modes(tables)

    assert "marker_analysis" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_uses_dataset_category_as_grid_gate() -> None:
    assert "grid_scan_spacing" in runner.detect_analysis_modes(_tables(), dataset_category="grid")
    assert "grid_scan_spacing" not in runner.detect_analysis_modes(_tables(), dataset_category="sweep")


def test_detect_analysis_modes_enables_geometry_phase_response_when_table_has_rows() -> None:
    tables = _tables()
    tables["geom_phase"] = pd.DataFrame(
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
    tables["cell_iris_cmp"] = _cell_iris_response_comparison_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "cell_iris_response" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_enables_coupler_cavity_parameters_when_table_has_rows() -> None:
    tables = _tables()
    tables["coupler_params"] = _coupler_cavity_parameter_estimates_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "coupler_cavity_parameters" in detected
    assert "grid_scan_spacing" not in detected


def test_run_folder_analysis_renders_coupler_cavity_parameters_when_available(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["coupler_params"] = _coupler_cavity_parameter_estimates_table()
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
    tables["cell_iris_cmp"] = _cell_iris_response_comparison_table()
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


def test_run_folder_analysis_renders_f2pi3_normalized_admittance_when_available(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["kyhl_admit_audit"] = _kyhl_f2pi3_normalized_admittance_audit_table()
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

    def fake_f2pi3_normalized_admittance(*args, **kwargs):
        folder = Path(args[1])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "f_2pi3_normalized_admittance.png"
        path.write_text("f2pi3_normalized_admittance", encoding="utf-8")
        return {"f_2pi3": path}

    monkeypatch.setattr(runner, "plot_s11_with_markers", fake_plot("s11"))
    monkeypatch.setattr(runner, "plot_phase_advance", fake_plot("phase_advance"))
    monkeypatch.setattr(runner, "plot_nodal_shift", fake_plot("nodal_shift"))
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(
        runner,
        "plot_f2pi3_normalized_admittance_view",
        fake_f2pi3_normalized_admittance,
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_sweep_260620_case",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result.figures["kyhl_normalized_admittance"]["f_2pi3"].exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"kyhl_normalized_admittance"' in manifest


def test_resolve_input_paths_uses_data_root_and_default_dispersion(
    tmp_path: Path,
) -> None:
    sparameter_path, dispersion_path = runner.resolve_input_paths(
        "prepro/prepro_sweep_260415_sample_prepro",
        data_root=tmp_path / "data",
    )

    assert sparameter_path == tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    assert dispersion_path == tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"


def test_resolve_input_paths_accepts_configured_default_dispersion(tmp_path: Path) -> None:
    _, dispersion_path = runner.resolve_input_paths(
        "sim/sim_sweep_260519_scan_dataset",
        data_root=tmp_path / "data",
        default_dispersion_subpath=Path("sim") / "custom_dispersion",
    )

    assert dispersion_path == tmp_path / "data" / "sim" / "custom_dispersion"


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
    assert set(result.tables) == {
        "phase_sweep_long",
        "phase_sweep_wide",
        "phase_sweep_summary",
    }
    summary = pd.read_csv(result.tables["phase_sweep_summary"])
    assert summary.loc[0, "freq_120_GHz"] == 2.86
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"dispersion"' in manifest
    assert '"s11"' not in manifest
    assert '"phase_advance"' not in manifest


def test_run_folder_analysis_uses_dispersion_only_lane_for_single_mode_cst_exports(tmp_path: Path, monkeypatch) -> None:
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
    assert set(result.tables) == {
        "phase_sweep_long",
        "phase_sweep_wide",
        "phase_sweep_summary",
    }
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
    assert set(result.figures["profile"]) == {
        "E_fieldDist",
        "EM_fieldPhase",
        "E_fieldPhase",
        "H_fieldPhase",
    }
    assert result.figures["profile"]["E_fieldDist"].exists()
    assert result.figures["profile"]["E_fieldPhase"].exists()
    assert result.figures["profile"]["H_fieldPhase"].exists()
    summary = pd.read_csv(result.tables["profile_summary"])
    assert summary["source_file"].tolist() == [
        "E_fieldDist.txt",
        "EM_fieldPhase.txt",
        "EM_fieldPhase.txt",
    ]
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
    tables["phase_adv"] = tables["phase_adv"].iloc[0:0]
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


def test_run_folder_analysis_reuses_loaded_table_when_grid_cache_becomes_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tables = _tables()
    tables["marker_pts"] = tables["marker_pts"].iloc[0:0]
    tables["phase_adv"] = tables["phase_adv"].iloc[0:0]
    tables["nodal_shift"] = tables["nodal_shift"].iloc[0:0]
    output_dir = tmp_path / "out"
    cached_s11 = output_dir / "figures" / "s11" / "s11_cached.png"
    cached_s11.parent.mkdir(parents=True)
    cached_s11.write_text("cached", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({"outputs": {"figures": {"s11": {"cached": str(cached_s11)}}}}),
        encoding="utf-8",
    )

    sparameter_path = tmp_path / "data" / "sim" / "sim_grid_260526_scan"
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run.s2p",
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 0.0,
            }
        ]
    )
    loader = runner.DataLoader()
    loaded_paths: list[Path] = []

    def tracked_load(path: str | Path) -> pd.DataFrame:
        loaded_paths.append(Path(path))
        return sparameter_table

    monkeypatch.setattr(loader, "load", tracked_load)

    def fake_build_marker_analysis(**kwargs):
        kwargs["loader"].load(kwargs["sparameter_path"])
        return tables

    monkeypatch.setattr(runner, "build_marker_analysis", fake_build_marker_analysis)
    monkeypatch.setattr(
        runner,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )
    monkeypatch.setattr(
        runner,
        "plot_s11_with_markers",
        lambda *args, **kwargs: {"overview": output_dir / "figures" / "s11" / "rebuilt.png"},
    )

    runner.run_folder_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
        loader=loader,
    )

    assert loaded_paths == [sparameter_path]


def test_runner_preserves_profile_and_dispersion_patch_seams(tmp_path: Path, monkeypatch) -> None:
    profile_path = tmp_path / "profile.txt"
    dispersion_path = tmp_path / "dispersion.txt"
    profile_path.write_text("profile", encoding="utf-8")
    dispersion_path.write_text("dispersion", encoding="utf-8")
    profile_calls: list[Path] = []
    dispersion_calls: list[Path] = []

    monkeypatch.setattr(
        runner,
        "load_field_profile_export",
        lambda path: profile_calls.append(Path(path)),
    )
    monkeypatch.setattr(
        runner,
        "load_cst_dispersion_txt",
        lambda path: dispersion_calls.append(Path(path)),
    )

    assert runner.find_cst_profile_inputs(profile_path) == (profile_path,)
    assert runner.find_cst_dispersion_inputs(dispersion_path) == (dispersion_path,)
    assert profile_calls == [profile_path]
    assert dispersion_calls == [dispersion_path]
    assert runner.BASE_ANALYSIS_MODES[0] == "marker_analysis"
    assert "sim_r_c" in runner.GRID_SCAN_REQUIRED_COLUMNS
    assert "f_2pi3" in runner.GRID_SCAN_REQUIRED_MARKERS


def test_run_folder_analysis_routes_3d_field_headers_before_1d_profile_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_folder = (
        tmp_path
        / "data"
        / "sim"
        / "sim_profile_260723_field3d_auto"
    )
    profile_folder.mkdir(parents=True)
    for kind, component, units in (
        ("e", "E", "V/m"),
        ("h", "H", "A/m"),
    ):
        header = (
            f"x [mm] y [mm] z [mm] "
            f"{component}xRe [{units}] {component}xIm [{units}] "
            f"{component}yRe [{units}] {component}yIm [{units}] "
            f"{component}zRe [{units}] {component}zIm [{units}]"
        )
        (profile_folder / f"{kind}-field [1]_NoPlunger.txt").write_text(
            header + "\n" + "-" * len(header) + "\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "fig" / "analyses" / profile_folder.name
    expected = runner.RunResult(
        output_dir=output_dir,
        tables=runner.AnalysisPaths(),
        figures=runner.FigurePaths(),
        analysis_modes=("field3d",),
        manifest_path=output_dir / "manifest.json",
    )
    recorded_pairs = []

    def fake_field3d_run(pairs, **kwargs):
        recorded_pairs.extend(pairs)
        return expected

    monkeypatch.setattr(
        runner,
        "_run_field3d_only_analysis_from_runner",
        fake_field3d_run,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "find_cst_profile_inputs",
        lambda path: (_ for _ in ()).throw(
            AssertionError("3D headers must bypass the 1D profile loader")
        ),
    )

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        dispersion_path=tmp_path / "dispersion",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result == expected
    assert [pair.case_id for pair in recorded_pairs] == ["NoPlunger"]


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
