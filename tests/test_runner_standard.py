from pathlib import Path
import json

import pandas as pd

import deflector_tuning.runner as runner
import deflector_tuning.workflows.marker_analysis as marker_workflow
import deflector_tuning.workflows.marker_figures as marker_figures
from runner_helpers import (
    _tables,
    _cell_iris_response_comparison_table,
    _coupler_cavity_parameter_estimates_table,
    _kyhl_f2pi3_normalized_admittance_audit_table,
    fake_figure_plot,
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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11", recorded_kwargs=s11_plot_kwargs))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_grid_scan_spacing_error_maps", fake_figure_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        marker_figures,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_figure_plot("grid_scan_sparameter_phase_r_c_line_scan"),
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


def test_run_folder_analysis_prefers_port_e_rows_for_position_phase_bars(
    tmp_path: Path, monkeypatch
) -> None:
    tables = _tables()
    tables["marker_pts"] = pd.DataFrame(
        [
            {
                "dataset_id": "raw_sweep_260701_iris_portE",
                "source_file": f"{position:.0f}_portE.S2P",
                "tune_position": position,
                "marker_name": marker_name,
                "freq_target_ghz": target_frequency,
                "freq_ghz": target_frequency,
                "s_phase_deg": phase,
            }
            for position, phases in {
                1.0: {"f_2pi3": 175.0, "f_mean": 180.0, "f_pi2": 170.0},
                2.0: {"f_2pi3": 65.0, "f_mean": 5.0, "f_pi2": 295.0},
            }.items()
            for marker_name, target_frequency, phase in (
                ("f_2pi3", 2.856, phases["f_2pi3"]),
                ("f_mean", 2.866, phases["f_mean"]),
                ("f_pi2", 2.876, phases["f_pi2"]),
            )
        ]
    )
    no_port_extension_rows = tables["marker_pts"].loc[
        tables["marker_pts"]["tune_position"].eq(1.0)
    ].copy()
    no_port_extension_rows["source_file"] = "1_noportE.S2P"
    no_port_extension_rows["s_phase_deg"] = 70.0
    tables["marker_pts"] = pd.concat(
        [tables["marker_pts"], no_port_extension_rows],
        ignore_index=True,
    )
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "1_portE.S2P",
                "tune_position": 1.0,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 175.0,
            }
        ]
    )
    output_dir = tmp_path / "out"
    rendered: list[tuple[str, Path, tuple[float, ...]]] = []
    rendered_sources: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {
            name: Path(output) / f"{name}.csv" for name in analysis_tables
        },
    )

    def fake_s11_plot(*args, **kwargs):
        output_folder = Path(args[2])
        output_folder.mkdir(parents=True, exist_ok=True)
        path = output_folder / "s11.png"
        path.write_text("s11", encoding="utf-8")
        return {"overview": path}

    def fake_mapping_plot(name):
        def _plot(*args, **kwargs):
            output_folder = Path(args[1])
            output_folder.mkdir(parents=True, exist_ok=True)
            path = output_folder / f"{name}.png"
            path.write_text(name, encoding="utf-8")
            return {"overview": path}

        return _plot

    def fake_phase_bar_plot(name):
        def _plot(marker_points, output_path, *, positions, config):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            rendered.append((name, path, tuple(positions)))
            rendered_sources.append(
                (name, tuple(marker_points["source_file"].unique()))
            )
            return path

        return _plot

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_s11_plot)
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_mapping_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_mapping_plot("nodal_shift"))
    monkeypatch.setattr(
        marker_figures, "plot_marker_phase_polar_views", fake_mapping_plot("polar")
    )
    monkeypatch.setattr(
        marker_figures,
        "plot_position_phase_bars",
        fake_phase_bar_plot("position_phase"),
        raising=False,
    )
    monkeypatch.setattr(
        marker_figures,
        "plot_position_phase_advance_bars",
        fake_phase_bar_plot("phase_advance"),
        raising=False,
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "raw" / "raw_sweep_260701_iris_portE",
        output_dir=output_dir,
        marker_role="raw",
    )

    expected_folder = output_dir / "figures" / "phase_bar"
    assert result.figures["phase_bar"]["position_phase"] == (
        expected_folder / "position_1p0_vs_2p0_phase_bars.png"
    )
    assert result.figures["phase_bar"]["phase_advance"] == (
        expected_folder / "position_1p0_to_2p0_phase_advance_bars.png"
    )
    assert rendered == [
        (
            "position_phase",
            expected_folder / "position_1p0_vs_2p0_phase_bars.png",
            (1.0, 2.0),
        ),
        (
            "phase_advance",
            expected_folder / "position_1p0_to_2p0_phase_advance_bars.png",
            (1.0, 2.0),
        ),
    ]
    assert rendered_sources == [
        (
            "position_phase",
            ("1_portE.S2P", "2_portE.S2P"),
        ),
        (
            "phase_advance",
            ("1_portE.S2P", "2_portE.S2P"),
        ),
    ]


def test_run_folder_analysis_propagates_geometry_options_and_records_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    tables = _tables()

    def fake_build_marker_analysis(**kwargs):
        captured.update(kwargs)
        return tables

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fake_build_marker_analysis)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {
            name: Path(output) / f"{name}.csv" for name in analysis_tables
        },
    )

    output_dir = tmp_path / "analysis"
    result = runner.run_folder_analysis(
        sparameter_path=tmp_path
        / "data"
        / "sim"
        / "sim_sweep_260728_zlen",
        output_dir=output_dir,
        marker_role="sim",
        tables_only=True,
        geometry_sweep_axis="sim_L_c",
        geometry_sweep_base=29.148,
    )

    assert captured["geometry_sweep_axis"] == "sim_L_c"
    assert captured["geometry_sweep_base"] == 29.148
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["geometry_phase_response"] == {
        "sweep_axis": "sim_L_c",
        "sweep_base": 29.148,
    }


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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_coupler_cavity_parameters", fake_figure_plot("coupler_cavity_parameters"))

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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_cell_iris_response_comparison", fake_figure_plot("cell_iris_response"))

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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fake_f2pi3_normalized_admittance(*args, **kwargs):
        folder = Path(args[1])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "f_2pi3_normalized_admittance.png"
        path.write_text("f2pi3_normalized_admittance", encoding="utf-8")
        return {"f_2pi3": path}

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(
        marker_figures,
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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def fail_phase_plot(*args, **kwargs):
        raise AssertionError("empty phase_advance should not be plotted")

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fail_phase_plot)
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_grid_scan_spacing_error_maps", fake_figure_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        marker_figures,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_figure_plot("grid_scan_sparameter_phase_r_c_line_scan"),
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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_grid_scan_spacing_error_maps", fake_figure_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        marker_figures,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_figure_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    with caplog.at_level("INFO", logger="deflector_tuning"):
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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(runner.DataLoader, "load", lambda self, path: sparameter_table)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fake_figure_plot("s11"))
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_grid_scan_spacing_error_maps", fake_figure_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        marker_figures,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_figure_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "sparameter_data" not in result.tables
    assert not (output_dir / "tables" / "sparameter_data.csv").exists()
