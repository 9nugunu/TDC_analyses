"""Plots for one-dimensional geometry phase-response sweeps."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS

REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "sweep_axis",
    "sweep_value",
    "cell_phase_deg",
    "iris_phase_deg",
    "cell_phase_shift_from_baseline_deg",
    "iris_phase_shift_from_baseline_deg",
)
def plot_geometry_phase_response(
    response: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write phase-response plots for a one-dimensional geometry sweep."""

    if response.empty:
        raise ValueError("geometry_phase_response is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in response]
    if missing:
        raise ValueError(f"geometry_phase_response is missing required columns: {missing}")
    require_finite_plot_columns(
        response,
        columns=(
            "sweep_value",
            "cell_phase_deg",
            "iris_phase_deg",
            "cell_phase_shift_from_baseline_deg",
            "iris_phase_shift_from_baseline_deg",
        ),
        context="geometry_phase_response",
    )

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = response.sort_values(["marker_name", "sweep_value"], kind="mergesort")
    sweep_axis = str(table["sweep_axis"].dropna().iloc[0])
    xlabel = _axis_label(sweep_axis)

    paths: OrderedDict[str, Path] = OrderedDict()
    paths["cell_iris_absolute_phase"] = _plot_family_absolute_phase(
        table,
        folder / "absolute_phase.png",
        xlabel=xlabel,
        config=config,
    )
    paths["cell_iris_phase_pickup"] = _plot_family_phase_pickup(
        table,
        folder / "phase_pickup.png",
        xlabel=xlabel,
        config=config,
    )
    for marker_name, marker_table in table.groupby("marker_name", sort=False):
        slug = _marker_slug(str(marker_name))
        paths[f"{slug}_absolute_phase"] = _plot_family_absolute_phase(
            marker_table,
            folder / f"{slug}_absolute_phase.png",
            xlabel=xlabel,
            config=config,
        )
        paths[f"{slug}_phase_pickup"] = _plot_family_phase_pickup(
            marker_table,
            folder / f"{slug}_phase_pickup.png",
            xlabel=xlabel,
            config=config,
        )
    return paths


def _plot_family_absolute_phase(table: pd.DataFrame, output_path: Path, *, xlabel: str, config: PlotConfig) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    _plot_family_lines(
        ax,
        table,
        value_columns=("cell_phase_deg", "iris_phase_deg"),
        value_labels=("cell", "iris"),
    )
    apply_axis_text_style(
        ax,
        xlabel=xlabel,
        ylabel="Phase advance Phi [deg]",
        title="Cell and iris phase advance Phi",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size, ncol=2)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_family_phase_pickup(table: pd.DataFrame, output_path: Path, *, xlabel: str, config: PlotConfig) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    _plot_family_lines(
        ax,
        table,
        value_columns=("cell_phase_shift_from_baseline_deg", "iris_phase_shift_from_baseline_deg"),
        value_labels=("cell", "iris"),
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    apply_axis_text_style(
        ax,
        xlabel=xlabel,
        ylabel="Marker phase shift from baseline [deg]",
        title="Cell and iris phase pickup",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size, ncol=2)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_family_lines(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    value_columns: tuple[str, str],
    value_labels: tuple[str, str],
) -> None:
    for marker_name, marker_table in table.groupby("marker_name", sort=False):
        marker_table = marker_table.sort_values("sweep_value", kind="mergesort")
        color = MARKER_COLORS.get(str(marker_name), "#333333")
        label = MARKER_LABELS.get(str(marker_name), str(marker_name))
        ax.plot(
            marker_table["sweep_value"],
            marker_table[value_columns[0]],
            marker="o",
            linewidth=2.0,
            linestyle="-",
            color=color,
            label=f"{label} {value_labels[0]}",
        )
        ax.plot(
            marker_table["sweep_value"],
            marker_table[value_columns[1]],
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color=color,
            label=f"{label} {value_labels[1]}",
        )


def _axis_label(sweep_axis: str) -> str:
    label = sweep_axis.removeprefix("sim_").replace("_", " ")
    return f"{label} [CST parameter]"


def _marker_slug(marker_name: str) -> str:
    return marker_name.lower().replace("/", "_").replace(" ", "_")
