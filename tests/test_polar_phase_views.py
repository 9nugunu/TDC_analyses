from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import deflector_tuning.visualization.polar_phase_views as polar_phase_views
from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.plot_config import PlotConfig


def _marker_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "target_freq_ghz": 2.866,
                "freq_ghz": 2.866,
                "freq_error_ghz": 0.0,
                "s_db": -2.0,
                "s_phase_deg": -110.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_pi2",
                "marker_role": "exp",
                "target_freq_ghz": 2.876,
                "freq_ghz": 2.876,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 130.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.5,
                "s_phase_deg": 20.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "target_freq_ghz": 2.866,
                "freq_ghz": 2.866,
                "freq_error_ghz": 0.0,
                "s_db": -2.5,
                "s_phase_deg": -100.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_pi2",
                "marker_role": "exp",
                "target_freq_ghz": 2.876,
                "freq_ghz": 2.876,
                "freq_error_ghz": 0.0,
                "s_db": -3.5,
                "s_phase_deg": 140.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )


def _grid_scan_marker_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, sim_r_c, sim_w_c, phases in [
        ("run03.s2p", 54.59, 18.3224, (10.0, -110.0, 130.0)),
        ("run10.s2p", 55.59, 19.3224, (25.0, -95.0, 145.0)),
    ]:
        for marker_name, target_freq_ghz, s_db, phase_deg in [
            ("f_2pi3", 2.856, -1.0, phases[0]),
            ("f_mean", 2.866, -2.0, phases[1]),
            ("f_pi2", 2.876, -3.0, phases[2]),
        ]:
            rows.append(
                {
                    "dataset_id": "grid_scan",
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": source_file,
                    "tune_position": 0.5,
                    "sim_r_c": sim_r_c,
                    "sim_w_c": sim_w_c,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": marker_name,
                    "marker_role": "sim",
                    "target_freq_ghz": target_freq_ghz,
                    "freq_ghz": target_freq_ghz,
                    "freq_error_ghz": 0.0,
                    "s_db": s_db,
                    "s_phase_deg": phase_deg,
                    "source_format": "touchstone_ri",
                }
            )
    return pd.DataFrame(rows)


def test_plot_marker_phase_polar_views_writes_per_position_and_overview_pngs(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "0.5",
        "1.0",
        "overview",
        "cell_overlay",
        "cell_f_2pi3_overlay",
        "iris_overlay",
        "iris_f_2pi3_overlay",
    ]
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    assert paths["0.5"].name == "cell_0p5.png"
    assert paths["1.0"].name == "iris_1p0.png"
    assert paths["overview"].name == "all_positions.png"
    assert paths["cell_overlay"].name == "cell_overlay.png"
    assert paths["cell_f_2pi3_overlay"].name == "cell_f_2pi3_overlay.png"
    assert paths["iris_overlay"].name == "iris_overlay.png"
    assert paths["iris_f_2pi3_overlay"].name == "iris_f_2pi3_overlay.png"
    assert plt.rcParams["font.sans-serif"][:4] == ["Pretendard", "Noto Sans", "Malgun Gothic", "DejaVu Sans"]


def test_plot_marker_phase_polar_views_uses_full_typography_for_per_position_and_overview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    config = PlotConfig(dpi=120, title_size=21, compact_title_size=7, label_size=17, compact_label_size=6)
    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_marker_points(), tmp_path, config=config)

    assert saved_figures
    for fig in saved_figures:
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            assert ax.title.get_fontsize() == pytest.approx(config.title_size)
            marker_labels = [text for text in ax.texts if "f_{" in text.get_text()]
            if marker_labels:
                assert {text.get_fontsize() for text in marker_labels} == {float(config.label_size)}
                continue
            legend = ax.get_legend()
            if legend is not None:
                legend_labels = [text for text in legend.get_texts() if "f_{" in text.get_text()]
                if legend_labels:
                    assert {text.get_fontsize() for text in legend_labels} == {float(config.label_size)}


def test_plot_marker_phase_polar_views_uses_marker_specific_colors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert saved_figures
    first_axis = next(ax for ax in saved_figures[0].axes if ax.get_visible())
    line_colors = {line.get_color() for line in first_axis.lines}
    assert polar_phase_views.MARKER_COLORS["f_2pi3"] in line_colors
    assert polar_phase_views.MARKER_COLORS["f_mean"] in line_colors
    assert polar_phase_views.MARKER_COLORS["f_pi2"] in line_colors


def test_plot_marker_phase_polar_views_adds_ideal_guides_to_f_2pi3_family_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_paths: list[Path] = []
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_paths.append(path)
        saved_figures.append(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    guide_targets = {"cell_f_2pi3_overlay.png", "iris_f_2pi3_overlay.png"}
    matched = [
        fig
        for fig, path in zip(saved_figures, saved_paths, strict=True)
        if path.name in guide_targets
    ]
    assert len(matched) == 2
    for fig in matched:
        ax = next(axis for axis in fig.axes if axis.get_visible())
        guide_labels = {text.get_text() for text in ax.texts if text.get_text() in {"0°", "120°", "240°"}}
        assert guide_labels == {"0°", "120°", "240°"}


def test_plot_marker_phase_polar_views_adds_ideal_guides_to_per_position_tune_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_paths: list[Path] = []
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_paths.append(path)
        saved_figures.append(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    guide_targets = {"cell_0p5.png", "iris_1p0.png"}
    matched = [
        fig
        for fig, path in zip(saved_figures, saved_paths, strict=True)
        if path.name in guide_targets
    ]
    assert len(matched) == 2
    for fig in matched:
        ax = next(axis for axis in fig.axes if axis.get_visible())
        guide_labels = {text.get_text() for text in ax.texts if text.get_text() in {"0°", "120°", "240°"}}
        assert guide_labels == {"0°", "120°", "240°"}


def test_plot_marker_phase_polar_views_uses_overview_only_when_tune_positions_are_missing(tmp_path: Path) -> None:
    table = _marker_points().assign(tune_position=pd.NA)

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["overview"]
    assert paths["overview"].name == "all_positions.png"
    assert paths["overview"].exists()


def test_plot_marker_phase_polar_views_uses_grid_points_when_tune_position_does_not_vary(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_grid_scan_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["r_c=54.59, w_c=18.3224", "r_c=55.59, w_c=19.3224"]
    assert paths["r_c=54.59, w_c=18.3224"].name == "position_r_c_54p59_w_c_18p3224.png"
    assert paths["r_c=55.59, w_c=19.3224"].name == "position_r_c_55p59_w_c_19p3224.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_marker_phase_polar_views_rejects_empty_marker_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="marker_points is empty"):
        plot_marker_phase_polar_views(pd.DataFrame(), tmp_path)


def test_plot_marker_phase_polar_views_skips_family_overlay_when_family_has_single_position(tmp_path: Path) -> None:
    table = _marker_points().iloc[:3].copy()

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["overview"]
