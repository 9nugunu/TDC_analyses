from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd

from deflector_tuning.visualization import field3d_plots
from deflector_tuning.visualization.plot_config import PlotConfig


def test_plot_slater_positions_groups_cell_and_iris_positions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    table = pd.DataFrame(
        {
            "case_ids": [
                "NumDepth0.5",
                "NumDepth1.0",
                "NumDepth1.5",
                "NumDepth2.0",
            ],
            "e_j": [0.35e-12, 0.77e-12, 0.25e-12, 0.69e-12],
            "h_j": [0.17e-12, 0.18e-12, 0.33e-12, 0.17e-12],
            "k_e_minus_h_j": [0.18e-12, 0.59e-12, -0.08e-12, 0.52e-12],
            "k_over_u": [1.0e-6, 3.0e-6, -1.0e-6, 2.0e-6],
            "inserted_k_e_minus_h_j": [
                0.12e-12,
                0.42e-12,
                -0.03e-12,
                0.31e-12,
            ],
            "inserted_k_over_u": [2.0e-6, 1.5e-6, 0.5e-6, 2.5e-6],
        }
    )
    captured: list[matplotlib.figure.Figure] = []

    def capture_figure(
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        config: PlotConfig,
    ) -> Path:
        captured.append(fig)
        return Path(output_path)

    monkeypatch.setattr(field3d_plots, "save_figure", capture_figure)

    field3d_plots.plot_slater_positions(
        table,
        tmp_path / "slater_pos.png",
        config=PlotConfig(dpi=120),
    )

    ax = captured[0].axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "0.5",
        "1.5",
        "1.0",
        "2.0",
    ]
    group_labels = ax.texts
    assert [text.get_text() for text in group_labels] == ["Cell", "Iris"]
    assert all(
        text.get_fontsize() == ax.xaxis.label.get_fontsize()
        for text in group_labels
    )
    tick_positions = ax.get_xticks()
    assert tick_positions[2] - tick_positions[1] > tick_positions[1] - tick_positions[0]
    _, legend_labels = ax.get_legend_handles_labels()
    assert r"$K/U$ (NoPlunger field)" in legend_labels
    assert r"$K/U$ (inserted-state field)" in legend_labels
    assert ax.get_ylabel() == (
        r"Normalized local term $\mathbf{K}/\mathbf{U}$ [ppm]"
    )
    assert not ax.containers
    labeled_lines = {
        line.get_label(): line
        for line in ax.lines
        if not line.get_label().startswith("_")
    }
    assert labeled_lines[r"$K/U$ (NoPlunger field)"].get_ydata().tolist() == [
        1.0,
        -1.0,
    ]
    assert labeled_lines[r"$K/U$ (inserted-state field)"].get_ydata().tolist() == [
        2.0,
        0.5,
    ]


def test_plot_slater_volumes_uses_cumulative_energy_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    table = pd.DataFrame(
        {
            "case_ids": [
                "NumDepth0.5",
                "NumDepth1.0",
                "NumDepth1.5",
                "NumDepth2.0",
            ],
            "e_j": [4.0e-12, 3.0e-12, 2.0e-12, 1.0e-12],
            "h_j": [1.0e-12, 0.8e-12, 0.6e-12, 0.4e-12],
            "k_e_minus_h_j": [3.0e-12, 2.2e-12, 1.4e-12, 0.6e-12],
        }
    )
    captured: list[matplotlib.figure.Figure] = []

    def capture_figure(
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        config: PlotConfig,
    ) -> Path:
        captured.append(fig)
        return Path(output_path)

    monkeypatch.setattr(field3d_plots, "save_figure", capture_figure)

    field3d_plots.plot_slater_volumes(
        table,
        tmp_path / "slater_vol.png",
        config=PlotConfig(dpi=120),
    )

    ax = captured[0].axes[0]
    assert ax.get_title() == "Cumulative Slater overlap"
    assert ax.get_ylabel() == "Integrated energy [pJ]"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "0.5",
        "1.5",
        "1.0",
        "2.0",
    ]
