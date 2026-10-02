from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import deflector_tuning.visualization.polar_phase_views as polar_phase_views
from deflector_tuning.visualization.polar_phase_views import (
    build_polar_plot_plans,
    plot_marker_phase_polar_views,
)
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


def _kyhl_pair_marker_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, tune_position, phases in [
        ("0.5_processed.csv", 0.5, (10.0, -110.0, 130.0)),
        ("1.0_processed.csv", 1.0, (20.0, -100.0, 140.0)),
        ("1.5_processed.csv", 1.5, (35.0, -70.0, 160.0)),
        ("2.0_processed.csv", 2.0, (45.0, -60.0, 170.0)),
    ]:
        for marker_name, target_freq_ghz, s_db, phase_deg in [
            ("f_2pi3", 2.856, -1.0, phases[0]),
            ("f_mean", 2.866, -2.0, phases[1]),
            ("f_pi2", 2.876, -3.0, phases[2]),
        ]:
            rows.append(
                {
                    "dataset_id": "sample_dataset",
                    "data_kind": "experiment",
                    "data_layer": "prepro",
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": marker_name,
                    "marker_role": "exp",
                    "target_freq_ghz": target_freq_ghz,
                    "freq_ghz": target_freq_ghz,
                    "freq_error_ghz": 0.0,
                    "s_db": s_db,
                    "s_phase_deg": phase_deg,
                    "source_format": "processed_csv_db_phase",
                }
            )
    return pd.DataFrame(rows)


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
                    "dataset_id": "sim_grid_260526_scan",
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


def _grid_scan_marker_points_with_tune_positions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, tune_position, sim_r_c, sim_w_c, phases in [
        ("run03_cell.s2p", 0.5, 54.59, 18.3224, (10.0, -110.0, 130.0)),
        ("run03_iris.s2p", 1.0, 54.59, 18.3224, (20.0, -100.0, 140.0)),
        ("run10_cell.s2p", 0.5, 55.59, 19.3224, (25.0, -95.0, 145.0)),
        ("run10_iris.s2p", 1.0, 55.59, 19.3224, (35.0, -85.0, 155.0)),
    ]:
        for marker_name, target_freq_ghz, s_db, phase_deg in [
            ("f_2pi3", 2.856, -1.0, phases[0]),
            ("f_mean", 2.866, -2.0, phases[1]),
            ("f_pi2", 2.876, -3.0, phases[2]),
        ]:
            rows.append(
                {
                    "dataset_id": "sim_grid_260526_scan",
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": source_file,
                    "tune_position": tune_position,
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


def _plunger_offset_marker_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, offset, num_depth, phases in [
        ("06_3-4_TDS-Half-Plunger-IrisOffset-260526_1.s1p", -3.0, 1, (10.0, -110.0, 130.0)),
        ("06_3-4_TDS-Half-Plunger-IrisOffset-260526_4.s1p", 0.0, 1, (25.0, -95.0, 145.0)),
    ]:
        for marker_name, target_freq_ghz, s_db, phase_deg in [
            ("f_2pi3", 2.856, -1.0, phases[0]),
            ("f_mean", 2.866, -2.0, phases[1]),
            ("f_pi2", 2.876, -3.0, phases[2]),
        ]:
            rows.append(
                {
                    "dataset_id": "sim_sweep_260526_iris_plunger_offset",
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": source_file,
                    "tune_position": pd.NA,
                    "sim_DepthPlunger_offset": offset,
                    "sim_NumDepth": num_depth,
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


def test_build_polar_plot_plans_describes_outputs_before_rendering(tmp_path: Path) -> None:
    plans = build_polar_plot_plans(_marker_points(), tmp_path)

    assert [plan.key for plan in plans] == [
        "0.5",
        "1.0",
        "overview",
        "cell_overlay",
        "cell_f_2pi3_overlay",
        "iris_overlay",
        "iris_f_2pi3_overlay",
    ]
    assert [plan.kind for plan in plans] == [
        "position",
        "position",
        "overview",
        "family_overlay",
        "family_overlay",
        "family_overlay",
        "family_overlay",
    ]
    assert all(not plan.output_path.exists() for plan in plans)


@pytest.mark.parametrize(
    "marker_points_factory",
    [
        _marker_points,
        _kyhl_pair_marker_points,
        _grid_scan_marker_points,
        _grid_scan_marker_points_with_tune_positions,
        _plunger_offset_marker_points,
    ],
)
def test_build_polar_plot_plans_preserves_input_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_points_factory
) -> None:
    marker_points = marker_points_factory()
    original = marker_points.copy(deep=True)
    output_dir = tmp_path / "planned_outputs"

    def reject_rendering(*args, **kwargs):
        raise AssertionError("planning must not create figures")

    monkeypatch.setattr(plt, "subplots", reject_rendering)

    plans = build_polar_plot_plans(marker_points, output_dir)

    assert plans
    assert not output_dir.exists()
    pd.testing.assert_frame_equal(marker_points, original)


def test_build_polar_plot_plans_places_no_port_extension_positions_in_subfolder(
    tmp_path: Path,
) -> None:
    marker_points = _marker_points().copy()
    marker_points.loc[marker_points["tune_position"] == 0.5, "source_file"] = "0.5_noportE.s2p"
    marker_points.loc[marker_points["tune_position"] == 1.0, "source_file"] = "1_portE.s2p"

    plans = build_polar_plot_plans(marker_points, tmp_path)
    paths = {plan.key: plan.output_path for plan in plans}

    assert paths["0.5"] == tmp_path / "No_portExtension" / "cell_0p5.png"
    assert paths["1.0"] == tmp_path / "iris_1p0.png"


def test_plot_marker_phase_polar_views_writes_separate_kyhl_phase_pair_arc_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: dict[str, matplotlib.figure.Figure] = {}

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures[path.name] = fig
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    paths = plot_marker_phase_polar_views(_kyhl_pair_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert paths["kyhl_phase_cell_overlay"].name == "kyhl_phase_cell_overlay.png"
    assert paths["kyhl_phase_iris_overlay"].name == "kyhl_phase_iris_overlay.png"
    assert "kyhl_phase_pair_overlay" not in paths

    cell_ax = saved_figures["kyhl_phase_cell_overlay.png"].axes[0]
    iris_ax = saved_figures["kyhl_phase_iris_overlay.png"].axes[0]
    assert cell_ax.get_title() == "KYHL cell phase overlay: 0.5->1.5"
    assert iris_ax.get_title() == "KYHL iris phase overlay: 1.0->2.0"
    assert "cell 0.5->1.5" in [text.get_text() for text in cell_ax.texts]
    assert "iris 1.0->2.0" in [text.get_text() for text in iris_ax.texts]
    assert cell_ax.get_legend() is None
    assert iris_ax.get_legend() is None

    for ax in [cell_ax, iris_ax]:
        text_labels = [text.get_text() for text in ax.texts]
        assert any(r"\Delta" in label and "deg" in label for label in text_labels)
        line_colors = {line.get_color() for line in ax.lines}
        assert polar_phase_views.MARKER_COLORS["f_2pi3"] in line_colors
        assert polar_phase_views.MARKER_COLORS["f_mean"] in line_colors
        assert polar_phase_views.MARKER_COLORS["f_pi2"] in line_colors
        marker_arc_lines = [line for line in ax.lines if line.get_gid() == "kyhl_phase_rotation_arc"]
        assert len(marker_arc_lines) == 3
        assert all(len(line.get_xdata()) > 2 for line in marker_arc_lines)
        start_radial_lines = [line for line in ax.lines if line.get_gid() == "kyhl_phase_radial_start"]
        end_radial_lines = [line for line in ax.lines if line.get_gid() == "kyhl_phase_radial_end"]
        assert len(start_radial_lines) == 3
        assert len(end_radial_lines) == 3
        for line in start_radial_lines + end_radial_lines:
            assert len(line.get_xdata()) == 2
            assert len(set(line.get_xdata())) == 1
            assert line.get_ydata()[0] == pytest.approx(0.0)
            assert line.get_ydata()[1] == pytest.approx(1.0)
        for line in start_radial_lines:
            assert line.get_alpha() == pytest.approx(0.50)
            assert line.get_linestyle() == ":"
        for line in end_radial_lines:
            assert line.get_alpha() == pytest.approx(0.50)
            assert line.get_linestyle() == "-"


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


def test_plot_marker_phase_polar_views_uses_concise_sweep_titles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    titles = [ax.get_title() for fig in saved_figures for ax in fig.axes if ax.get_visible()]
    assert "0.5: Polar phase" in titles
    assert "1.0: Polar phase" in titles
    assert "Cell overlay: Polar phase" in titles
    assert "Cell $f_{2\\pi/3}$ overlay: Polar phase" in titles


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


def test_plot_marker_phase_polar_views_fans_out_clustered_marker_labels(
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
    clustered = _marker_points().copy()
    clustered.loc[clustered["tune_position"] == 0.5, "s_phase_deg"] = [-147.0, -157.0, -166.0]

    plot_marker_phase_polar_views(clustered, tmp_path, config=PlotConfig(dpi=120))

    fig = next(fig for fig, path in zip(saved_figures, saved_paths, strict=True) if path.name == "cell_0p5.png")
    ax = next(axis for axis in fig.axes if axis.get_visible())
    marker_labels = [text for text in ax.texts if "f_{" in text.get_text()]
    assert len(marker_labels) == 3
    marker_positions = [text.get_position() for text in marker_labels]
    assert len({round(radius, 3) for _, radius in marker_positions}) > 1
    assert max(theta for theta, _ in marker_positions) - min(theta for theta, _ in marker_positions) > 0.5
    delta_32_label = next(text for text in ax.texts if r"\Delta\phi_{32}" in text.get_text())
    assert delta_32_label.get_ha() == "left"
    assert delta_32_label.get_position()[1] < 0.70


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
        guide_labels = {text.get_text() for text in ax.texts if text.get_text() in {"0°", "180°", "60°", "-60°"}}
        assert guide_labels == {"0°", "180°", "60°", "-60°"}


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
        guide_labels = {text.get_text() for text in ax.texts if text.get_text() in {"0°", "180°", "60°", "-60°"}}
        assert guide_labels == {"0°", "180°", "60°", "-60°"}


def test_plot_marker_phase_polar_views_uses_configured_guide_angles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_figures: dict[str, matplotlib.figure.Figure] = {}

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        path = Path(output_path)
        saved_figures[path.name] = fig
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(
        _marker_points(),
        tmp_path,
        config=PlotConfig(dpi=120, ideal_phase_guide_angles_deg=(0.0, 90.0)),
    )

    axis = next(axis for axis in saved_figures["cell_0p5.png"].axes if axis.get_visible())
    guide_labels = {text.get_text() for text in axis.texts if text.get_text() in {"0°", "90°"}}
    assert guide_labels == {"0°", "90°"}


def test_plot_marker_phase_polar_views_adds_ideal_guides_to_sim_260526_grid_scan_plots(
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

    plot_marker_phase_polar_views(_grid_scan_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    guide_targets = {"cell_r_c_54p59_w_c_18p3224.png", "cell_r_c_55p59_w_c_19p3224.png"}
    matched = [fig for fig, path in zip(saved_figures, saved_paths, strict=True) if path.name in guide_targets]
    assert len(matched) == 2
    for fig in matched:
        ax = next(axis for axis in fig.axes if axis.get_visible())
        guide_labels = {text.get_text() for text in ax.texts if text.get_text() in {"0°", "180°", "60°", "-60°"}}
        assert guide_labels == {"0°", "180°", "60°", "-60°"}


def test_plot_marker_phase_polar_views_splits_by_source_file_when_tune_positions_are_missing(tmp_path: Path) -> None:
    table = _marker_points().assign(tune_position=pd.NA)

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["0.5_processed", "1.0_processed", "overview"]
    assert paths["0.5_processed"].name == "position_0p5_processed.png"
    assert paths["1.0_processed"].name == "position_1p0_processed.png"
    assert paths["overview"].name == "all_positions.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_marker_phase_polar_views_uses_grid_points_when_tune_position_does_not_vary(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_grid_scan_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["cell r_c=54.59, w_c=18.3224", "cell r_c=55.59, w_c=19.3224"]
    assert paths["cell r_c=54.59, w_c=18.3224"].name == "cell_r_c_54p59_w_c_18p3224.png"
    assert paths["cell r_c=55.59, w_c=19.3224"].name == "cell_r_c_55p59_w_c_19p3224.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_marker_phase_polar_views_keeps_same_grid_point_num_depths_separate(tmp_path: Path) -> None:
    depth_1 = _grid_scan_marker_points().copy()
    depth_1["source_file"] = depth_1["source_file"].str.replace("run03", "run001").str.replace("run10", "run003")
    depth_1["sim_NumDepth"] = 1
    depth_1["tune_position"] = 0.5
    depth_2 = depth_1.copy()
    depth_2["source_file"] = depth_2["source_file"].str.replace("run001", "run002").str.replace("run003", "run004")
    depth_2["sim_NumDepth"] = 2
    depth_2["tune_position"] = 1.5
    table = pd.concat([depth_1, depth_2], ignore_index=True)

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "cell depth=1p0 r_c=54.59, w_c=18.3224",
        "cell depth=1p0 r_c=55.59, w_c=19.3224",
        "cell depth=2p0 r_c=54.59, w_c=18.3224",
        "cell depth=2p0 r_c=55.59, w_c=19.3224",
    ]
    assert (
        paths["cell depth=1p0 r_c=54.59, w_c=18.3224"].name
        == "cell_depth_1p0_r_c_54p59_w_c_18p3224.png"
    )
    assert (
        paths["cell depth=2p0 r_c=54.59, w_c=18.3224"].name
        == "cell_depth_2p0_r_c_54p59_w_c_18p3224.png"
    )


def test_plot_marker_phase_polar_views_uses_result_navigator_sweep_columns(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_plunger_offset_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["plunger_offset=-3", "plunger_offset=0"]
    assert paths["plunger_offset=-3"].name == "iris_plunger_offset_m3.png"
    assert paths["plunger_offset=0"].name == "iris_plunger_offset_0.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_marker_phase_polar_views_includes_num_depth_in_sweep_filenames(tmp_path: Path) -> None:
    depth_1 = _plunger_offset_marker_points()
    depth_1["tune_position"] = 0.5
    depth_2 = depth_1.copy()
    depth_2["sim_NumDepth"] = 2
    depth_2["tune_position"] = 1.5
    depth_2["source_file"] = depth_2["source_file"].str.replace("_1.s1p", "_10.s1p").str.replace("_4.s1p", "_13.s1p")
    table = pd.concat([depth_1, depth_2], ignore_index=True)

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "depth=1p0_plunger_offset=-3",
        "depth=1p0_plunger_offset=0",
        "depth=2p0_plunger_offset=-3",
        "depth=2p0_plunger_offset=0",
    ]
    assert paths["depth=1p0_plunger_offset=-3"].name == "iris_depth_1p0_plunger_offset_m3.png"
    assert paths["depth=1p0_plunger_offset=0"].name == "iris_depth_1p0_plunger_offset_0.png"
    assert paths["depth=2p0_plunger_offset=-3"].name == "iris_depth_2p0_plunger_offset_m3.png"
    assert paths["depth=2p0_plunger_offset=0"].name == "iris_depth_2p0_plunger_offset_0.png"


def test_plot_marker_phase_polar_views_splits_duplicate_positions_by_source_file(tmp_path: Path) -> None:
    table = pd.concat(
        [
            _marker_points().iloc[:3],
            _marker_points().iloc[:3].assign(
                source_file="repeat_processed.csv",
                s_phase_deg=[40.0, -80.0, 160.0],
            ),
        ],
        ignore_index=True,
    )

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["0.5_processed", "repeat_processed", "overview"]
    assert paths["0.5_processed"].name == "position_0p5_processed.png"
    assert paths["repeat_processed"].name == "position_repeat_processed.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_marker_phase_polar_views_uses_grid_points_before_tune_positions(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_grid_scan_marker_points_with_tune_positions(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "cell r_c=54.59, w_c=18.3224",
        "cell r_c=55.59, w_c=19.3224",
        "iris r_c=54.59, w_c=18.3224",
        "iris r_c=55.59, w_c=19.3224",
    ]
    assert paths["cell r_c=54.59, w_c=18.3224"].name == "cell_r_c_54p59_w_c_18p3224.png"
    assert paths["cell r_c=55.59, w_c=19.3224"].name == "cell_r_c_55p59_w_c_19p3224.png"
    assert paths["iris r_c=54.59, w_c=18.3224"].name == "iris_r_c_54p59_w_c_18p3224.png"
    assert paths["iris r_c=55.59, w_c=19.3224"].name == "iris_r_c_55p59_w_c_19p3224.png"


def test_plot_marker_phase_polar_views_uses_concise_grid_titles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def _capture_figure(fig: matplotlib.figure.Figure, output_path: str | Path, config: PlotConfig | None = None) -> Path:
        saved_figures.append(fig)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(polar_phase_views, "save_figure", _capture_figure)

    plot_marker_phase_polar_views(_grid_scan_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    titles = [ax.get_title() for fig in saved_figures for ax in fig.axes if ax.get_visible()]
    assert titles == [
        "Cell polar: r_c=54.59, w_c=18.3224",
        "Cell polar: r_c=55.59, w_c=19.3224",
    ]


def test_plot_marker_phase_polar_views_rejects_empty_marker_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="marker_points is empty"):
        plot_marker_phase_polar_views(pd.DataFrame(), tmp_path)


def test_plot_marker_phase_polar_views_reports_non_finite_phase_values(tmp_path: Path) -> None:
    table = _marker_points().copy()
    table.loc[0, "s_phase_deg"] = float("nan")

    with pytest.raises(ValueError, match=r"Non-finite plotting values in polar marker_points"):
        plot_marker_phase_polar_views(table, tmp_path)


def test_plot_marker_phase_polar_views_skips_family_overlay_when_family_has_single_position(tmp_path: Path) -> None:
    table = _marker_points().iloc[:3].copy()

    paths = plot_marker_phase_polar_views(table, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["overview"]
