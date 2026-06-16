from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest

import deflector_tuning.visualization.s11_frequency_plots as s11_frequency_plots
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers


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
                    "dataset_id": "250609_beforebrazing",
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
                "dataset_id": "250609_beforebrazing",
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
                "dataset_id": "250609_beforebrazing",
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
                "dataset_id": "250609_beforebrazing",
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
                "dataset_id": "250609_beforebrazing",
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

    assert "overview" not in paths
    assert "position_0p5" in paths
    assert "position_1p5" in paths
    assert paths["position_0p5"].name == "s11_position_0p5.png"
    assert not (tmp_path / "s11_with_markers.png").exists()
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_plot_s11_with_markers_skips_overview_when_port_sides_exist(tmp_path: Path) -> None:
    paths = plot_s11_with_markers(
        _port_side_sparameter_table(),
        _port_side_marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    assert "overview" not in paths
    assert set(paths) == {"position_0p5", "position_1p5"}
    assert paths["position_0p5"].name == "s11_position_0p5.png"


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

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "s11_position_0p5.png")
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

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "s11_position_0p5.png")
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

    target_index = next(index for index, path in enumerate(saved_paths) if path.name == "s11_position_0p5.png")
    ax = next(axis for axis in saved_figures[target_index].axes if axis.get_visible())
    marker_texts = [text for text in ax.texts if "f_{" in text.get_text()]
    assert len(marker_texts) == 2
    alignments = {text.get_ha() for text in marker_texts}
    assert alignments == {"left", "right"}


def test_plot_s11_with_markers_accepts_grid_scan_without_tune_position(tmp_path: Path) -> None:
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
    assert paths["grid_r_c_56p59_w_c_19p32"].name == "s11_r_c_56p59_w_c_19p32.png"
    assert paths["grid_r_c_56p59_w_c_19p32"].exists()


def test_plot_s11_with_markers_rejects_empty_sparameter_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sparameter_table is empty"):
        plot_s11_with_markers(pd.DataFrame(), _marker_points(), tmp_path)


def test_plot_s11_with_markers_reports_non_finite_marker_values(tmp_path: Path) -> None:
    marker_points = _marker_points().copy()
    marker_points.loc[0, "s_db"] = float("-inf")

    with pytest.raises(ValueError, match=r"Non-finite plotting values in S11 marker_points"):
        plot_s11_with_markers(_sparameter_table(), marker_points, tmp_path)
