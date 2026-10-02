from pathlib import Path
import json
from dataclasses import replace

import pandas as pd
import pytest

import deflector_tuning.runner as runner
import deflector_tuning.workflows.marker_analysis as marker_workflow
import deflector_tuning.workflows.marker_figures as marker_figures
from runner_helpers import (
    _tables,
    fake_figure_plot,
)


def test_run_folder_analysis_loads_sparameter_folder_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sparameter_path = tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    dispersion_path = tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"
    sparameter_path.mkdir(parents=True)
    for tune_position, phases in [
        (0.5, [10.0, 20.0, 30.0]),
        (1.5, [-110.0, -160.0, 150.0]),
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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", lambda **_: tables)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {
            name: Path(output) / f"{name}.csv" for name in analysis_tables
        },
    )

    def fail_plot(*args, **kwargs):
        raise AssertionError("tables-only mode must not render figures")

    monkeypatch.setattr(marker_workflow, "load_s11_table_for_figures", fail_plot)
    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fail_plot)
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fail_plot)
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fail_plot)
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fail_plot)

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


@pytest.mark.parametrize("change", ["trace", "marker", "style", "missing", "legacy", "tables_only", "tables_only_missing"])
def test_grid_s11_cache_reuses_only_unchanged_render_inputs(tmp_path: Path, monkeypatch, change) -> None:
    tables = _tables()
    source_table = tables["marker_pts"].copy()
    output_dir = tmp_path / "out"
    load_calls = []
    plot_calls = []
    loader = runner.DataLoader()

    def load(path):
        load_calls.append(path)
        return source_table

    def build(**kwargs):
        kwargs["loader"].load(kwargs["sparameter_path"])
        return tables

    monkeypatch.setattr(loader, "load", load)
    monkeypatch.setattr(marker_workflow, "build_marker_analysis", build)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )

    def plot_s11(*args, **kwargs):
        plot_calls.append(1)
        figures = fake_figure_plot("s11")(*args, **kwargs)
        second = figures["overview"].with_name("second_s11.png")
        second.write_text("second image")
        return {**figures, "second": second}

    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", plot_s11)
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fake_figure_plot("phase_advance"))
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fake_figure_plot("nodal_shift"))
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fake_figure_plot("polar"))
    monkeypatch.setattr(marker_figures, "plot_grid_scan_spacing_error_maps", fake_figure_plot("grid_scan_spacing"))
    monkeypatch.setattr(
        marker_figures,
        "plot_grid_scan_sparameter_phase_r_c_line_scan",
        fake_figure_plot("grid_scan_sparameter_phase_r_c_line_scan"),
    )

    options = dict(
        sparameter_path=tmp_path / "data" / "sim" / "sim_grid_260526_scan",
        dispersion_path=tmp_path / "data" / "sim" / "sim_dispersion_260505_case",
        output_dir=output_dir,
        marker_role="sim",
        loader=loader,
    )
    first = runner.run_folder_analysis(**options)
    second = runner.run_folder_analysis(**options)
    assert first.figures == second.figures
    assert len(plot_calls) == 1
    assert len(load_calls) == 2  # One load per run, even with analysis and drawing.

    manifest_path = output_dir / "manifest.json"
    old_signature = json.loads(manifest_path.read_text())["figure_cache"]["s11"]
    if change in {"trace", "tables_only"}:
        source_table.loc[0, "s_db"] = -8.0
    elif change == "marker":
        tables["marker_pts"].loc[0, "freq_ghz"] = 2.858
    elif change == "style":
        options["project_defaults"] = replace(
            runner.DEFAULT_PROJECT_DEFAULTS, ideal_phase_guide_angles_deg=(0.0, 90.0)
        )
    elif change in {"missing", "tables_only_missing"}:
        first.figures["s11"]["overview"].unlink()
    else:
        payload = json.loads(manifest_path.read_text())
        payload.pop("figure_cache")
        manifest_path.write_text(json.dumps(payload))

    if change.startswith("tables_only"):
        runner.run_folder_analysis(**options, tables_only=True)
        assert len(plot_calls) == 1
        cached = json.loads(manifest_path.read_text())["figure_cache"]
        if change == "tables_only_missing":
            assert "s11" not in cached
        else:
            # Table-only refresh must not certify old figures against new inputs.
            assert cached["s11"] == old_signature
    runner.run_folder_analysis(**options)
    assert len(plot_calls) == 2
    assert len(load_calls) == (4 if change.startswith("tables_only") else 3)


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

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fake_build_marker_analysis)
    monkeypatch.setattr(
        marker_workflow,
        "save_marker_analysis",
        lambda analysis_tables, output: {name: Path(output) / f"{name}.csv" for name in analysis_tables},
    )
    monkeypatch.setattr(
        marker_figures,
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
