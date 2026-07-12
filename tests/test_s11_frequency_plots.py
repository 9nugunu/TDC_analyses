from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest

import deflector_tuning.visualization.s11_frequency_plots as s11_frequency_plots
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.s11_frequency_plots import (
    build_s11_plot_plans,
    plot_s11_with_markers,
)


def _sparameter_table() -> pd.DataFrame:
    rows = []
    for source_file, tune_position, phase_offset in [
        ("0.5_processed.csv", 0.5, 0.0),
        ("1.5_processed.csv", 1.5, -20.0),
    ]:
        for freq_ghz, s_db, phase in [
            (2.84, -1.0, 10.0 + phase_offset),
            (2.856, -3.0, 20.0 + phase_offset),
            (2.872, -2.0, 30.0 + phase_offset),
        ]:
            rows.append(
                {
                    "dataset_id": "dataset",
                    "data_layer": "prepro",
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "s_name": "S11",
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "source_format": "processed_csv_db_phase",
                }
            )
    return pd.DataFrame(rows)


def _marker_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 20.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "dataset",
                "data_layer": "prepro",
                "source_file": "1.5_processed.csv",
                "tune_position": 1.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 0.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )


def test_thin_trace_for_plot_limits_dense_frequency_traces() -> None:
    table = pd.DataFrame({"freq_ghz": range(100_011), "s_db": range(100_011)})

    result = s11_frequency_plots._thin_trace_for_plot(table)

    assert len(result) == s11_frequency_plots.MAX_TRACE_POINTS_PER_SOURCE
    assert result.iloc[0]["freq_ghz"] == 0
    assert result.iloc[-1]["freq_ghz"] == 100_010


def test_build_s11_plot_plans_describes_outputs_before_rendering(tmp_path: Path) -> None:
    plans = build_s11_plot_plans(
        _sparameter_table(),
        _marker_points(),
        tmp_path,
        split_by_position=True,
    )

    assert [plan.key for plan in plans] == ["overview", "cell_0p5", "cell_1p5"]
    assert [plan.kind for plan in plans] == ["overview", "position", "position"]
    assert [plan.output_path.name for plan in plans] == [
        "with_markers.png",
        "cell_0p5.png",
        "cell_1p5.png",
    ]
    assert all(not plan.output_path.exists() for plan in plans)


def test_build_s11_plot_plans_splits_one_dimensional_navigator_points(tmp_path: Path) -> None:
    s_table = _sparameter_table().assign(
        source_file=["run_001.s1p"] * 3 + ["run_002.s1p"] * 3,
        tune_position=pd.NA,
        sim_r_c=[56.09] * 3 + [56.10] * 3,
    )
    marker_points = _marker_points().assign(
        source_file=["run_001.s1p", "run_002.s1p"],
        tune_position=pd.NA,
        sim_r_c=[56.09, 56.10],
    )

    plans = build_s11_plot_plans(s_table, marker_points, tmp_path)

    assert [plan.key for plan in plans] == ["r_c_56p09", "r_c_56p1"]
    assert [plan.output_path.name for plan in plans] == ["r_c_56p09.png", "r_c_56p1.png"]
    assert [plan.title for plan in plans] == [
        "S11 magnitude | r_c = 56.09 mm",
        "S11 magnitude | r_c = 56.1 mm",
    ]


def test_build_s11_plot_plans_uses_navigator_values_in_two_dimensional_titles(tmp_path: Path) -> None:
    s_table = _sparameter_table().assign(
        source_file=["run_001.s1p"] * 3 + ["run_002.s1p"] * 3,
        tune_position=pd.NA,
        sim_r_c=[56.09] * 3 + [56.10] * 3,
        sim_w_c=[19.0224] * 3 + [19.1224] * 3,
    )
    marker_points = _marker_points().assign(
        source_file=["run_001.s1p", "run_002.s1p"],
        tune_position=pd.NA,
        sim_r_c=[56.09, 56.10],
        sim_w_c=[19.0224, 19.1224],
    )

    plan = build_s11_plot_plans(s_table, marker_points, tmp_path)[0]

    assert plan.output_path.name == "r_c_56p09_w_c_19p0224.png"
    assert plan.title == "S11 magnitude | r_c = 56.09 mm; w_c = 19.0224 mm"


def _port_side_sparameter_table() -> pd.DataFrame:
    rows = []
    for source_file, port_side, tune_position, phase_offset in [
        ("in_0.5cell.csv", "in", 0.5, 0.0),
        ("out_0.5cell.csv", "out", 0.5, -5.0),
        ("in_1.5cell.csv", "in", 1.5, -20.0),
        ("out_1.5cell.csv", "out", 1.5, -25.0),
    ]:
        for freq_ghz, s_db, phase in [
            (2.84, -1.0, 10.0 + phase_offset),
            (2.856, -3.0, 20.0 + phase_offset),
            (2.872, -2.0, 30.0 + phase_offset),
        ]:
            rows.append(
                {
                    "dataset_id": "raw_sweep_250609_sparams_beforebrazing",
                    "data_layer": "raw",
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "port_side": port_side,
                    "s_name": "S11",
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "source_format": "raw_csv_ri",
                }
            )
    return pd.DataFrame(rows)


def _port_side_marker_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "raw_sweep_250609_sparams_beforebrazing",
                "data_layer": "raw",
                "source_file": "in_0.5cell.csv",
                "tune_position": 0.5,
                "port_side": "in",
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 20.0,
                "source_format": "raw_csv_ri",
            },
            {
                "dataset_id": "raw_sweep_250609_sparams_beforebrazing",
                "data_layer": "raw",
                "source_file": "out_0.5cell.csv",
                "tune_position": 0.5,
                "port_side": "out",
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 15.0,
                "source_format": "raw_csv_ri",
            },
            {
                "dataset_id": "raw_sweep_250609_sparams_beforebrazing",
                "data_layer": "raw",
                "source_file": "in_1.5cell.csv",
                "tune_position": 1.5,
                "port_side": "in",
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 0.0,
                "source_format": "raw_csv_ri",
            },
            {
                "dataset_id": "raw_sweep_250609_sparams_beforebrazing",
                "data_layer": "raw",
                "source_file": "out_1.5cell.csv",
                "tune_position": 1.5,
                "port_side": "out",
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": -5.0,
                "source_format": "raw_csv_ri",
            },
        ]
    )


def test_plot_s11_with_markers_writes_individual_position_pngs_without_overview(tmp_path: Path) -> None:
    paths = plot_s11_with_markers(
        _sparameter_table(),
        _marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    assert "overview" in paths
    assert "cell_0p5" in paths
    assert "cell_1p5" in paths
    assert paths["overview"].name == "with_markers.png"
    assert paths["cell_0p5"].name == "cell_0p5.png"
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_plot_s11_with_markers_names_num_depth_positions_by_depth(tmp_path: Path) -> None:
    sparameter_table = _sparameter_table().copy()
    marker_points = _marker_points().copy()
    sparameter_table["sim_NumDepth"] = sparameter_table["tune_position"].map({0.5: 1, 1.5: 2})
    marker_points["sim_NumDepth"] = marker_points["tune_position"].map({0.5: 1, 1.5: 2})

    paths = plot_s11_with_markers(sparameter_table, marker_points, tmp_path, config=PlotConfig(dpi=120))

    assert "overview" in paths
    assert "depth_1p0" in paths
    assert "depth_2p0" in paths
    assert paths["depth_1p0"].name == "depth_1p0.png"
    assert paths["depth_2p0"].name == "depth_2p0.png"


def test_plot_s11_with_markers_skips_overview_when_port_sides_exist(tmp_path: Path) -> None:
    paths = plot_s11_with_markers(
        _port_side_sparameter_table(),
        _port_side_marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    assert "overview" not in paths
    assert set(paths) == {"cell_0p5", "cell_1p5"}
    assert paths["cell_0p5"].name == "cell_0p5.png"


def test_plot_s11_with_markers_uses_distinct_port_side_styles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []
    saved_paths: list[Path] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures.append(fig)
        saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)

    plot_s11_with_markers(
        _port_side_sparameter_table(),
        _port_side_marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "cell_0p5.png")
    ax = next(axis for axis in saved_figures[target_index].axes if axis.get_visible())
    trace_lines = [line for line in ax.lines if line.get_label() in {"in", "out"}]
    trace_styles = {(line.get_label(), line.get_color(), line.get_linestyle()) for line in trace_lines}
    assert ("in", s11_frequency_plots.PORT_SIDE_STYLES["in"]["color"], s11_frequency_plots.PORT_SIDE_STYLES["in"]["linestyle"]) in trace_styles
    assert ("out", s11_frequency_plots.PORT_SIDE_STYLES["out"]["color"], s11_frequency_plots.PORT_SIDE_STYLES["out"]["linestyle"]) in trace_styles


def test_plot_s11_with_markers_filters_touchstone_table_to_s11_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures.append(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)
    sparameter_table = pd.concat(
        [
            _sparameter_table(),
            _sparameter_table().assign(s_name="S21", s_db=-60.0),
        ],
        ignore_index=True,
    )
    marker_points = pd.concat(
        [
            _marker_points(),
            _marker_points().assign(s_name="S21", s_db=-60.0),
        ],
        ignore_index=True,
    )

    plot_s11_with_markers(sparameter_table, marker_points, tmp_path, split_by_position=True, config=PlotConfig(dpi=120))

    ax = next(axis for axis in saved_figures[0].axes if axis.get_visible())
    trace_lines = [line for line in ax.lines if line.get_label() != "_nolegend_"]
    assert all(min(line.get_ydata()) > -10.0 for line in trace_lines)


def test_plot_s11_with_markers_collapses_unlabelled_duplicate_position_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []
    saved_paths: list[Path] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures.append(fig)
        saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)
    s1p = _sparameter_table().query("tune_position == 0.5").assign(source_file="0.5.S1P", port_side=pd.NA)
    s2p = s1p.assign(source_file="0.5.S2P", s_db=s1p["s_db"] - 1.0)
    marker_s1p = _marker_points().query("tune_position == 0.5").assign(source_file="0.5.S1P", port_side=pd.NA)
    marker_s2p = marker_s1p.assign(source_file="0.5.S2P", s_db=marker_s1p["s_db"] - 1.0)

    plot_s11_with_markers(
        pd.concat([s1p, s2p], ignore_index=True),
        pd.concat([marker_s1p, marker_s2p], ignore_index=True),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "cell_0p5.png")
    ax = next(axis for axis in saved_figures[target_index].axes if axis.get_visible())
    trace_lines = [line for line in ax.lines if line.get_label() == "0.5"]
    marker_texts = [text for text in ax.texts if "f_{" in text.get_text()]
    assert len(trace_lines) == 1
    assert len(marker_texts) == 1


def test_plot_s11_with_markers_separates_port_side_marker_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []
    saved_paths: list[Path] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures.append(fig)
        saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)

    plot_s11_with_markers(
        _port_side_sparameter_table(),
        _port_side_marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "cell_0p5.png")
    ax = next(axis for axis in saved_figures[target_index].axes if axis.get_visible())
    marker_texts = [text for text in ax.texts if "f_{" in text.get_text()]
    assert len(marker_texts) == 2
    alignments = {text.get_ha() for text in marker_texts}
    assert alignments == {"left", "right"}


def test_plot_s11_with_markers_accepts_sim_260526_grid_scan_without_tune_position(tmp_path: Path) -> None:
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s1p",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            },
            {
                "source_file": "run1.s1p",
                "freq_ghz": 2.86,
                "s_db": -2.0,
                "s_phase_deg": 20.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            },
        ]
    )
    marker_points = pd.DataFrame(
        [
            {
                "source_file": "run1.s1p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            }
        ]
    )

    paths = plot_s11_with_markers(sparameter_table, marker_points, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["grid_r_c_56p59_w_c_19p32"]
    assert paths["grid_r_c_56p59_w_c_19p32"].name == "r_c_56p59_w_c_19p32.png"
    assert paths["grid_r_c_56p59_w_c_19p32"].exists()


def test_plot_s11_with_markers_omits_unreadable_large_legends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    for index in range(s11_frequency_plots.MAX_LEGEND_ENTRIES + 1):
        source_file = f"run_{index:03d}.s2p"
        rows.extend(
            [
                {
                    "source_file": source_file,
                    "freq_ghz": 2.60,
                    "s_name": "S11",
                    "s_db": -10.0 - index * 0.01,
                    "s_phase_deg": 0.0,
                },
                {
                    "source_file": source_file,
                    "freq_ghz": 2.70,
                    "s_name": "S11",
                    "s_db": -8.0 - index * 0.01,
                    "s_phase_deg": 10.0,
                },
            ]
        )
        marker_rows.append(
            {
                "source_file": source_file,
                "marker_name": "f_mean",
                "freq_ghz": 2.65,
                "s_name": "S11",
                "s_db": -9.0,
                "s_phase_deg": 5.0,
            }
        )
    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)

    plot_s11_with_markers(pd.DataFrame(rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    assert saved_figures
    assert saved_figures[0].axes[0].get_legend() is None


def test_plot_s11_with_markers_labels_source_traces_by_run_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    sparameter_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    for source_file, offset in [
        ("run_001.s2p", 0.0),
        ("06_3-5_TDS-Half-Plunger-Iris2Dsweep-260527_99.s1p", -1.0),
    ]:
        for freq_ghz, s_db in [(2.85, -1.0 + offset), (2.86, -2.0 + offset)]:
            sparameter_rows.append(
                {
                    "source_file": source_file,
                    "freq_ghz": freq_ghz,
                    "s_name": "S11",
                    "s_db": s_db,
                    "s_phase_deg": 10.0,
                }
            )
        marker_rows.append(
            {
                "source_file": source_file,
                "marker_name": "f_mean",
                "freq_ghz": 2.855,
                "s_name": "S11",
                "s_db": -1.5 + offset,
                "s_phase_deg": 10.0,
            }
        )

    monkeypatch.setattr(s11_frequency_plots, "save_figure", _capture_figure)

    plot_s11_with_markers(pd.DataFrame(sparameter_rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    legend = saved_figures[0].axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["RUN 001", "RUN 99"]


def test_plot_s11_with_markers_does_not_write_position_nan_when_positions_are_missing(tmp_path: Path) -> None:
    sparameter_table = _sparameter_table().assign(tune_position=pd.NA)
    marker_points = _marker_points().assign(tune_position=pd.NA)

    paths = plot_s11_with_markers(sparameter_table, marker_points, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["overview"]
    assert paths["overview"].name == "with_markers.png"


def test_plot_s11_with_markers_ignores_unpositioned_reference_trace_for_position_split(tmp_path: Path) -> None:
    reference_trace = _sparameter_table().query("tune_position == 0.5").assign(
        source_file="Fullstructure.S2P",
        tune_position=float("nan"),
    )
    sparameter_table = pd.concat([_sparameter_table(), reference_trace], ignore_index=True)

    paths = plot_s11_with_markers(sparameter_table, _marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert "overview" in paths
    assert "cell_0p5" in paths
    assert "cell_1p5" in paths
    assert all("nan" not in key.lower() for key in paths)


def test_plot_s11_with_markers_names_sim_260526_grid_scan_by_cell_or_iris_when_tune_position_exists(tmp_path: Path) -> None:
    sparameter_rows = []
    marker_rows = []
    for source_file, tune_position, sim_r_c, sim_w_c in [
        ("cell_run.s2p", 0.5, 56.59, 19.32),
        ("iris_run.s2p", 1.0, 56.59, 19.32),
    ]:
        for freq_ghz, s_db, phase in [(2.85, -1.0, 10.0), (2.86, -2.0, 20.0)]:
            sparameter_rows.append(
                {
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "sim_r_c": sim_r_c,
                    "sim_w_c": sim_w_c,
                }
            )
        marker_rows.append(
            {
                "source_file": source_file,
                "tune_position": tune_position,
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": sim_r_c,
                "sim_w_c": sim_w_c,
            }
        )

    paths = plot_s11_with_markers(pd.DataFrame(sparameter_rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["cell_r_c_56p59_w_c_19p32", "iris_r_c_56p59_w_c_19p32"]
    assert paths["cell_r_c_56p59_w_c_19p32"].name == "cell_r_c_56p59_w_c_19p32.png"
    assert paths["iris_r_c_56p59_w_c_19p32"].name == "iris_r_c_56p59_w_c_19p32.png"


def test_plot_s11_with_markers_keeps_same_tune_position_grid_points_separate(tmp_path: Path) -> None:
    sparameter_rows = []
    marker_rows = []
    for source_file, sim_r_c, sim_w_c in [
        ("run_001.s2p", 56.59, 19.3224),
        ("run_002.s2p", 56.84, 19.3224),
    ]:
        for freq_ghz, s_db, phase in [(2.85, -1.0, 10.0), (2.86, -2.0, 20.0)]:
            sparameter_rows.append(
                {
                    "source_file": source_file,
                    "tune_position": 0.5,
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "sim_r_c": sim_r_c,
                    "sim_w_c": sim_w_c,
                }
            )
        marker_rows.append(
            {
                "source_file": source_file,
                "tune_position": 0.5,
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": sim_r_c,
                "sim_w_c": sim_w_c,
            }
        )

    paths = plot_s11_with_markers(pd.DataFrame(sparameter_rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["cell_r_c_56p59_w_c_19p3224", "cell_r_c_56p84_w_c_19p3224"]
    assert paths["cell_r_c_56p59_w_c_19p3224"].name == "cell_r_c_56p59_w_c_19p3224.png"
    assert paths["cell_r_c_56p84_w_c_19p3224"].name == "cell_r_c_56p84_w_c_19p3224.png"


def test_plot_s11_with_markers_keeps_same_grid_point_num_depths_separate(tmp_path: Path) -> None:
    sparameter_rows = []
    marker_rows = []
    for source_file, num_depth, tune_position in [
        ("run_001.s2p", 1, 0.5),
        ("run_002.s2p", 2, 1.5),
    ]:
        for freq_ghz, s_db, phase in [(2.85, -1.0, 10.0), (2.86, -2.0, 20.0)]:
            sparameter_rows.append(
                {
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "sim_NumDepth": num_depth,
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "sim_r_c": 56.59,
                    "sim_w_c": 19.3224,
                }
            )
        marker_rows.append(
            {
                "source_file": source_file,
                "tune_position": tune_position,
                "sim_NumDepth": num_depth,
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.3224,
            }
        )

    paths = plot_s11_with_markers(pd.DataFrame(sparameter_rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["cell_depth_1p0_r_c_56p59_w_c_19p3224", "cell_depth_2p0_r_c_56p59_w_c_19p3224"]
    assert paths["cell_depth_1p0_r_c_56p59_w_c_19p3224"].name == "cell_depth_1p0_r_c_56p59_w_c_19p3224.png"
    assert paths["cell_depth_2p0_r_c_56p59_w_c_19p3224"].name == "cell_depth_2p0_r_c_56p59_w_c_19p3224.png"


def test_plot_s11_with_markers_splits_sim_sweeps_by_depth_and_offset(tmp_path: Path) -> None:
    sparameter_rows = []
    marker_rows = []
    for source_file, num_depth, tune_position, offset in [
        ("run_001.s2p", 4.5, 4.5, -3),
        ("run_002.s2p", 4.5, 4.5, 0),
        ("run_003.s2p", 4.5, 4.5, 3),
        ("run_004.s2p", 5.0, 5.0, -3),
        ("run_002.s2p", 5.0, 5.0, 0),
    ]:
        for freq_ghz, s_db, phase in [(2.85, -1.0, 10.0), (2.86, -2.0, 20.0)]:
            sparameter_rows.append(
                {
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "sim_NumDepth": num_depth,
                    "sim_offset_cell_03": offset,
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                }
            )
        marker_rows.append(
            {
                "source_file": source_file,
                "tune_position": tune_position,
                "sim_NumDepth": num_depth,
                "sim_offset_cell_03": offset,
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
            }
        )

    paths = plot_s11_with_markers(pd.DataFrame(sparameter_rows), pd.DataFrame(marker_rows), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "cell_depth_4p5_offset_cell_03_m3",
        "cell_depth_4p5_offset_cell_03_0",
        "cell_depth_4p5_offset_cell_03_3",
        "iris_depth_5p0_offset_cell_03_m3",
        "iris_depth_5p0_offset_cell_03_0",
    ]
    assert paths["cell_depth_4p5_offset_cell_03_m3"].name == "cell_depth_4p5_offset_cell_03_m3.png"
    assert paths["iris_depth_5p0_offset_cell_03_0"].name == "iris_depth_5p0_offset_cell_03_0.png"


def test_plot_s11_with_markers_rejects_empty_sparameter_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sparameter_table is empty"):
        plot_s11_with_markers(pd.DataFrame(), _marker_points(), tmp_path)


def test_plot_s11_with_markers_reports_non_finite_marker_values(tmp_path: Path) -> None:
    marker_points = _marker_points().copy()
    marker_points.loc[0, "s_db"] = float("-inf")

    with pytest.raises(ValueError, match=r"Non-finite plotting values in S11 marker_points"):
        plot_s11_with_markers(_sparameter_table(), marker_points, tmp_path)
