from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import deflector_tuning.visualization.grid_scan_phase_line_plots as phase_line_plots
from deflector_tuning.visualization.grid_scan_phase_line_plots import (
    plot_grid_scan_sparameter_phase_r_c_line_scan,
    plot_phase_rc_map,
)
from deflector_tuning.visualization.plot_config import PlotConfig


def test_plot_grid_scan_sparameter_phase_r_c_line_scan_writes_named_phase_plot(tmp_path: Path) -> None:
    marker_points = pd.DataFrame(
        [
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -20.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -30.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 18.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -40.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_mean",
                "s_phase_deg": 160.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_mean",
                "s_phase_deg": 170.0,
            },
            {
                "sim_r_c": 55.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_pi2",
                "s_phase_deg": 30.0,
            },
            {
                "sim_r_c": 54.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_pi2",
                "s_phase_deg": 20.0,
            },
        ]
    )

    paths = plot_grid_scan_sparameter_phase_r_c_line_scan(marker_points, tmp_path)

    assert list(paths) == ["sparameter_phase_vs_r_c_at_w_c_19p3224"]
    assert paths["sparameter_phase_vs_r_c_at_w_c_19p3224"].name == "sparameter_phase_vs_r_c_at_w_c_19p3224.png"
    assert paths["sparameter_phase_vs_r_c_at_w_c_19p3224"].stat().st_size > 0
    csv = pd.read_csv(tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224.csv")
    assert csv["sim_w_c"].tolist() == [19.3224] * 6
    assert csv["sim_r_c"].tolist() == [54.59, 55.59, 54.59, 55.59, 54.59, 55.59]
    assert csv["marker_name"].tolist() == ["f_2pi3", "f_2pi3", "f_mean", "f_mean", "f_pi2", "f_pi2"]
    assert csv["s_phase_deg"].tolist() == [-30.0, -20.0, 170.0, 160.0, 20.0, 30.0]
    assert "s_phase_deg_0_360" not in csv
    assert "s_phase_deg_unwrapped" not in csv
    f_mean_csv = pd.read_csv(tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224__f_mean.csv")
    assert f_mean_csv["marker_name"].tolist() == ["f_mean", "f_mean"]
    assert f_mean_csv["s_phase_deg"].tolist() == [170.0, 160.0]
    assert "s_phase_deg_0_360" not in f_mean_csv
    assert "s_phase_deg_unwrapped" not in f_mean_csv
    assert (tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224__f_2pi3.csv").exists()
    assert (tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224__f_pi2.csv").exists()


def test_plot_phase_rc_map_writes_baseline_current_and_design_guides(tmp_path: Path) -> None:
    line_scan = pd.DataFrame(
        [
            {"sim_r_c": 56.20, "sim_w_c": 19.3224, "marker_name": "f_mean", "s_phase_deg": 12.0},
            {"sim_r_c": 56.59, "sim_w_c": 19.3224, "marker_name": "f_mean", "s_phase_deg": -3.0},
        ]
    )
    fit = pd.DataFrame(
        [
            {"state": "baseline", "r_c_mm": 56.179},
            {"state": "current", "r_c_mm": 56.237},
            {"state": "design", "r_c_mm": 56.590},
        ]
    )
    output = tmp_path / "phase_rc_map.png"

    path = plot_phase_rc_map(line_scan, fit, output)

    assert path == output
    assert path.stat().st_size > 0


def test_plot_phase_rc_map_uses_initial_current_ideal_labels_and_ideal_annotations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    line_scan = pd.DataFrame(
        [
            {"sim_r_c": 56.20, "sim_w_c": 19.3224, "marker_name": "f_mean", "s_phase_deg": 12.0},
            {"sim_r_c": 56.59, "sim_w_c": 19.3224, "marker_name": "f_mean", "s_phase_deg": -3.0},
        ]
    )
    rc_fit = pd.DataFrame(
        [
            {"state": "baseline", "r_c_mm": 56.179},
            {"state": "current", "r_c_mm": 56.237},
            {"state": "design", "r_c_mm": 56.590},
        ]
    )
    captured: dict[str, object] = {}

    def capture_plot(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "phase_rc_map.png"

    monkeypatch.setattr(phase_line_plots, "_plot_sparameter_phase_line_scan", capture_plot)

    phase_line_plots.plot_phase_rc_map(line_scan, rc_fit, tmp_path / "phase_rc_map.png")

    assert captured["candidate_label"] == "current $r_c=56.237$ mm"
    assert captured["before_label"] == "initial $r_c=56.179$ mm"
    assert captured["target_label"] == "ideal $r_c=56.590$ mm"
    assert captured["annotate_design_phase_values"] is True
    assert captured["design_r_c"] == 56.59


def test_plot_phase_rc_map_uses_sparse_marker_specific_line_styles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    line_scan = pd.DataFrame(
        [
            {
                "sim_r_c": 56.10 + 0.01 * point,
                "sim_w_c": 19.3224,
                "marker_name": marker_name,
                "s_phase_deg": phase - point,
            }
            for marker_name, phase in (("f_2pi3", 100.0), ("f_mean", 50.0), ("f_pi2", -20.0))
            for point in range(24)
        ]
    )
    rc_fit = pd.DataFrame(
        [
            {"state": "baseline", "r_c_mm": 56.20},
            {"state": "current", "r_c_mm": 56.24},
            {"state": "design", "r_c_mm": 56.59},
        ]
    )
    calls: list[dict[str, object]] = []
    original_plot = plt.Axes.plot

    def capture_plot(self, *args, **kwargs):
        calls.append(kwargs)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", capture_plot)

    plot_phase_rc_map(line_scan, rc_fit, tmp_path / "phase_rc_map.png")

    styles = {str(call["label"]): call for call in calls}
    assert styles[r"$f_{2\pi/3}$"]["marker"] == "o"
    assert styles[r"$f_{mean}$"]["marker"] == "s"
    assert styles[r"$f_{\pi/2}$"]["marker"] == "^"
    assert all(int(call["markevery"]) > 1 for call in calls)


def test_plot_grid_scan_sparameter_phase_r_c_line_scan_keeps_one_point_per_radius(tmp_path: Path) -> None:
    marker_points = pd.DataFrame(
        [
            {
                "sim_r_c": 56.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": -176.0,
                "source_file": "run_1.s1p",
                "run_id": 1,
            },
            {
                "sim_r_c": 56.59,
                "sim_w_c": 19.3224,
                "marker_name": "f_2pi3",
                "s_phase_deg": 62.0,
                "source_file": "run_2.s1p",
                "run_id": 2,
            },
        ]
    )

    plot_grid_scan_sparameter_phase_r_c_line_scan(marker_points, tmp_path)

    csv = pd.read_csv(tmp_path / "sparameter_phase_vs_r_c_at_w_c_19p3224.csv")
    assert csv["s_phase_deg"].tolist() == [62.0]
    assert "s_phase_deg_0_360" not in csv
    assert "s_phase_deg_unwrapped" not in csv
    assert csv["source_file"].tolist() == ["run_2.s1p"]


def test_design_phase_annotations_mask_the_line_and_show_the_anchor() -> None:
    figure, axis = plt.subplots()
    line_scan = pd.DataFrame(
        [
            {"marker_name": "f_2pi3", "sim_r_c": 56.59, "s_phase_deg": 62.0},
            {"marker_name": "f_mean", "sim_r_c": 56.59, "s_phase_deg": -3.3},
            {"marker_name": "f_pi2", "sim_r_c": 56.59, "s_phase_deg": -66.0},
        ]
    )

    phase_line_plots._annotate_design_phase_values(
        axis,
        line_scan,
        design_r_c=56.59,
        config=PlotConfig(),
    )

    assert len(axis.texts) == 3
    assert all(annotation.get_bbox_patch() is not None for annotation in axis.texts)
    assert all(annotation.arrow_patch is not None for annotation in axis.texts)
    annotations = {annotation.get_text(): annotation for annotation in axis.texts}
    assert annotations["62.0 deg"].get_position() == (10, -22)
    plt.close(figure)
