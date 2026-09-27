from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import pytest

from deflector_tuning.analysis.geometry_phase_response import compute_geometry_phase_response
from deflector_tuning.visualization import geometry_phase_response_plots
from deflector_tuning.visualization.geometry_phase_response_plots import plot_geometry_phase_response
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import PlotConfig


def _marker_points() -> pd.DataFrame:
    rows = []
    phases = {
        (-1.0, 4.5): {"f_2pi3": 10.0, "f_mean": 40.0},
        (-1.0, 5.0): {"f_2pi3": 110.0, "f_mean": 130.0},
        (0.0, 4.5): {"f_2pi3": 20.0, "f_mean": 45.0},
        (0.0, 5.0): {"f_2pi3": 140.0, "f_mean": 135.0},
        (1.0, 4.5): {"f_2pi3": 30.0, "f_mean": 50.0},
        (1.0, 5.0): {"f_2pi3": 170.0, "f_mean": 150.0},
    }
    for (offset, tune_position), marker_phases in phases.items():
        for marker_name, phase in marker_phases.items():
            rows.append(
                {
                    "dataset_id": "sim_sweep",
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": f"run_{offset}_{tune_position}.s2p",
                    "marker_name": marker_name,
                    "marker_role": "sim",
                    "s_name": "S11",
                    "tune_position": tune_position,
                    "sim_NumDepth": tune_position,
                    "sim_offset_cell_03": offset,
                    "s_phase_deg": phase,
                }
            )
    return pd.DataFrame(rows)


def test_compute_geometry_phase_response_pairs_cell_and_iris_by_sweep_axis() -> None:
    response = compute_geometry_phase_response(_marker_points())

    f_2pi3 = response[response["marker_name"] == "f_2pi3"].sort_values("sweep_value")

    assert f_2pi3["sweep_axis"].unique().tolist() == ["sim_offset_cell_03"]
    assert f_2pi3["sweep_value"].tolist() == [-1.0, 0.0, 1.0]
    assert f_2pi3["sweep_base"].unique().tolist() == [0.0]
    assert f_2pi3["cell_iris_phase_delta_deg"].tolist() == pytest.approx([100.0, 120.0, 140.0])
    assert f_2pi3["cell_iris_delta_shift_deg"].tolist() == pytest.approx([-20.0, 0.0, 20.0])
    assert f_2pi3["cell_phase_shift_deg"].tolist() == pytest.approx([-10.0, 0.0, 10.0])
    assert f_2pi3["iris_phase_shift_deg"].tolist() == pytest.approx([-30.0, 0.0, 30.0])


def test_compute_geometry_phase_response_returns_empty_when_sweep_axis_is_ambiguous() -> None:
    marker_points = _marker_points()
    marker_points["sim_other_offset"] = marker_points["sim_offset_cell_03"] * 2.0

    response = compute_geometry_phase_response(marker_points)

    assert response.empty


def test_compute_geometry_phase_response_uses_requested_axis_and_sampled_reference() -> None:
    marker_points = _marker_points().drop(columns="sim_offset_cell_03")
    marker_points["sim_L_c"] = marker_points["source_file"].map(
        {
            "run_-1.0_4.5.s2p": 28.148,
            "run_-1.0_5.0.s2p": 28.148,
            "run_0.0_4.5.s2p": 29.148,
            "run_0.0_5.0.s2p": 29.148,
            "run_1.0_4.5.s2p": 30.148,
            "run_1.0_5.0.s2p": 30.148,
        }
    )
    marker_points["sim_DepthPlunger"] = marker_points["tune_position"].map(
        {4.5: 49.562, 5.0: 67.056}
    )

    response = compute_geometry_phase_response(
        marker_points,
        sweep_axis="sim_L_c",
        sweep_base=29.148,
    )

    f_2pi3 = response[response["marker_name"] == "f_2pi3"].sort_values(
        "sweep_value"
    )
    assert f_2pi3["sweep_axis"].unique().tolist() == ["sim_L_c"]
    assert f_2pi3["sweep_base"].unique().tolist() == [29.148]
    assert f_2pi3["cell_phase_shift_deg"].tolist() == pytest.approx(
        [-10.0, 0.0, 10.0]
    )
    assert f_2pi3["iris_phase_shift_deg"].tolist() == pytest.approx(
        [-30.0, 0.0, 30.0]
    )


@pytest.mark.parametrize(
    ("sweep_axis", "sweep_base", "message"),
    [
        ("sim_missing", 29.148, "missing"),
        ("sim_constant", 29.148, "must vary"),
        ("sim_L_c", 29.0, "sampled"),
    ],
)
def test_compute_geometry_phase_response_rejects_invalid_explicit_axis_or_reference(
    sweep_axis: str,
    sweep_base: float,
    message: str,
) -> None:
    marker_points = _marker_points().drop(columns="sim_offset_cell_03")
    marker_points["sim_L_c"] = marker_points["source_file"].map(
        {
            "run_-1.0_4.5.s2p": 28.148,
            "run_-1.0_5.0.s2p": 28.148,
            "run_0.0_4.5.s2p": 29.148,
            "run_0.0_5.0.s2p": 29.148,
            "run_1.0_4.5.s2p": 30.148,
            "run_1.0_5.0.s2p": 30.148,
        }
    )
    marker_points["sim_constant"] = 1.0

    with pytest.raises(ValueError, match=message):
        compute_geometry_phase_response(
            marker_points,
            sweep_axis=sweep_axis,
            sweep_base=sweep_base,
        )


def test_plot_geometry_phase_response_writes_absolute_and_family_pickup_figures(tmp_path: Path) -> None:
    response = compute_geometry_phase_response(_marker_points())

    paths = plot_geometry_phase_response(response, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "cell_iris_absolute_phase",
        "cell_iris_phase_pickup",
        "f_2pi3_absolute_phase",
        "f_2pi3_phase_pickup",
        "f_mean_absolute_phase",
        "f_mean_phase_pickup",
    ]
    assert paths["cell_iris_absolute_phase"].name == "absolute_phase.png"
    assert paths["cell_iris_phase_pickup"].name == "phase_pickup.png"
    assert paths["f_2pi3_absolute_phase"].name == "f_2pi3_absolute_phase.png"
    assert paths["f_2pi3_phase_pickup"].name == "f_2pi3_phase_pickup.png"
    assert paths["cell_iris_absolute_phase"].exists()
    assert paths["cell_iris_phase_pickup"].exists()
    assert paths["f_2pi3_absolute_phase"].exists()
    assert paths["f_2pi3_phase_pickup"].exists()


def test_plot_geometry_phase_response_labels_l_c_as_length_and_marker_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = compute_geometry_phase_response(
        _marker_points().rename(columns={"sim_offset_cell_03": "sim_L_c"}),
        sweep_axis="sim_L_c",
        sweep_base=0.0,
    )
    axis_text_calls: list[dict[str, object]] = []
    original = geometry_phase_response_plots.apply_axis_text_style

    def capture_axis_text(*args: object, **kwargs: object) -> None:
        axis_text_calls.append(kwargs)
        original(*args, **kwargs)

    monkeypatch.setattr(
        geometry_phase_response_plots,
        "apply_axis_text_style",
        capture_axis_text,
    )

    plot_geometry_phase_response(response, tmp_path, config=PlotConfig(dpi=72))

    assert axis_text_calls[0]["xlabel"] == r"Coupler-cell length $L_c$ [mm]"
    assert axis_text_calls[0]["ylabel"] == r"$\phi$ [deg]"
    assert axis_text_calls[0]["title"] == "Cell and iris marker phase"


def test_plot_geometry_phase_response_aligns_square_iris_markers_in_second_legend_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_labels: list[list[str]] = []

    def capture_legend(fig: object, output_path: str | Path, config: PlotConfig) -> Path:
        legend = fig.axes[0].get_legend()
        captured_labels.append([text.get_text() for text in legend.get_texts()])
        return Path(output_path)

    monkeypatch.setattr(
        geometry_phase_response_plots,
        "save_figure",
        capture_legend,
    )

    plot_geometry_phase_response(
        compute_geometry_phase_response(_marker_points()),
        tmp_path,
        config=PlotConfig(dpi=72),
    )

    combined_labels = captured_labels[0]
    assert [label.rsplit(" ", 1)[1] for label in combined_labels] == [
        "cell",
        "cell",
        "iris",
        "iris",
    ]
    assert combined_labels[0].removesuffix(" cell") == combined_labels[2].removesuffix(" iris")
    assert combined_labels[1].removesuffix(" cell") == combined_labels[3].removesuffix(" iris")


def test_plot_geometry_phase_response_uses_light_cell_and_saturated_iris_colors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_axes: list[object] = []

    def capture_figure(fig: object, output_path: str | Path, config: PlotConfig) -> Path:
        captured_axes.append(fig.axes[0])
        return Path(output_path)

    monkeypatch.setattr(
        geometry_phase_response_plots,
        "save_figure",
        capture_figure,
    )

    plot_geometry_phase_response(
        compute_geometry_phase_response(_marker_points()),
        tmp_path,
        config=PlotConfig(dpi=72),
    )

    lines = {line.get_label(): line for line in captured_axes[0].lines}
    for marker_name in ("f_2pi3", "f_mean"):
        marker_label = MARKER_LABELS[marker_name]
        cell_rgb = mcolors.to_rgb(lines[f"{marker_label} cell"].get_color())
        iris_rgb = mcolors.to_rgb(lines[f"{marker_label} iris"].get_color())
        cell_hsv = mcolors.rgb_to_hsv(cell_rgb)
        iris_hsv = mcolors.rgb_to_hsv(iris_rgb)

        assert lines[f"{marker_label} iris"].get_color() == MARKER_COLORS[marker_name]
        assert cell_rgb != iris_rgb
        assert cell_hsv[1] < iris_hsv[1]
        assert cell_hsv[2] > iris_hsv[2]
        hue_difference = abs(float(cell_hsv[0] - iris_hsv[0]))
        assert min(hue_difference, 1.0 - hue_difference) < 0.01


def test_plot_geometry_phase_response_marks_baseline_and_uses_phase_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_points = _marker_points()
    marker_points["sim_L_c"] = marker_points["sim_offset_cell_03"] + 29.148
    marker_points = marker_points.drop(columns="sim_offset_cell_03")
    response = compute_geometry_phase_response(
        marker_points,
        sweep_axis="sim_L_c",
        sweep_base=29.148,
    )
    captured_axes: list[object] = []
    captured_axis_text: list[dict[str, object]] = []
    original_axis_text = geometry_phase_response_plots.apply_axis_text_style

    def capture_figure(fig: object, output_path: str | Path, config: PlotConfig) -> Path:
        captured_axes.append(fig.axes[0])
        return Path(output_path)

    def capture_axis_text(*args: object, **kwargs: object) -> None:
        captured_axis_text.append(kwargs)
        original_axis_text(*args, **kwargs)

    monkeypatch.setattr(
        geometry_phase_response_plots,
        "save_figure",
        capture_figure,
    )
    monkeypatch.setattr(
        geometry_phase_response_plots,
        "apply_axis_text_style",
        capture_axis_text,
    )

    plot_geometry_phase_response(response, tmp_path, config=PlotConfig(dpi=72))

    absolute_ax, pickup_ax = captured_axes[:2]
    assert captured_axis_text[0]["ylabel"] == r"$\phi$ [deg]"
    assert captured_axis_text[1]["ylabel"] == r"$\Delta\phi$ [deg]"
    for ax in (absolute_ax, pickup_ax):
        reference_lines = [
            line for line in ax.lines if line.get_label() == "_sweep_reference"
        ]
        assert len(reference_lines) == 1
        assert list(reference_lines[0].get_xdata()) == pytest.approx(
            [29.148, 29.148]
        )
        assert "_sweep_reference" not in [
            text.get_text() for text in ax.get_legend().get_texts()
        ]


def test_plot_geometry_phase_response_aligns_x_ticks_with_sweep_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_axes: list[object] = []

    def capture_figure(fig: object, output_path: str | Path, config: PlotConfig) -> Path:
        captured_axes.append(fig.axes[0])
        return Path(output_path)

    monkeypatch.setattr(
        geometry_phase_response_plots,
        "save_figure",
        capture_figure,
    )

    response = compute_geometry_phase_response(_marker_points())
    plot_geometry_phase_response(response, tmp_path, config=PlotConfig(dpi=72))

    assert list(captured_axes[0].get_xticks()) == pytest.approx([-1.0, 0.0, 1.0])
