from pathlib import Path

import pandas as pd

import deflector_tuning.runner as runner


def _tables() -> dict[str, pd.DataFrame]:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "scan",
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
                "dataset_id": "scan",
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
                "dataset_id": "scan",
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
                "dataset_id": "scan",
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
                "dataset_id": "scan",
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
                "dataset_id": "scan",
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
    }


def test_run_folder_analysis_saves_tables_figures_grid_scan_and_manifest(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "scan",
        dispersion_path=tmp_path / "data" / "sim" / "dispersion",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result.output_dir == output_dir
    assert "grid_scan_spacing" in result.analysis_modes
    assert result.manifest_path.exists()
    assert result.figures["s11"]["overview"].exists()
    assert result.figures["phase_advance"]["overview"].exists()
    assert result.figures["polar"]["overview"].exists()
    assert result.figures["grid_scan_spacing"]["overview"].exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"grid_scan_spacing"' in manifest
    assert '"enabled": true' in manifest


def test_detect_analysis_modes_skips_grid_scan_for_experiment_marker_points() -> None:
    tables = _tables()
    tables["marker_points"] = tables["marker_points"].assign(data_kind="experiment")

    detected = runner.detect_analysis_modes(tables)

    assert "marker_analysis" in detected
    assert "grid_scan_spacing" not in detected


def test_run_folder_analysis_skips_phase_plot_when_phase_table_is_empty(tmp_path: Path, monkeypatch) -> None:
    tables = _tables()
    tables["phase_advance"] = tables["phase_advance"].iloc[0:0]
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
    monkeypatch.setattr(runner, "plot_marker_phase_polar_views", fake_plot("polar"))
    monkeypatch.setattr(runner, "plot_grid_scan_spacing_error_maps", fake_plot("grid_scan_spacing"))

    result = runner.run_folder_analysis(
        sparameter_path=tmp_path / "data" / "sim" / "scan",
        dispersion_path=tmp_path / "data" / "sim" / "dispersion",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert "phase_advance" not in result.figures
    assert result.figures["s11"]["overview"].exists()
    assert result.figures["polar"]["overview"].exists()
