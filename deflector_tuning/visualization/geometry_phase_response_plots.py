"""Plots for one-dimensional geometry phase-response sweeps."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
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
    "sweep_base",
    "cell_phase_deg",
    "iris_phase_deg",
    "cell_phase_shift_deg",
    "iris_phase_shift_deg",
)

SIMULATION_AXIS_LABELS: dict[str, str] = {
    "sim_L_c": r"Coupler-cell length $L_c$ [mm]",
}


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
            "cell_phase_shift_deg",
            "iris_phase_shift_deg",
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
    _draw_sweep_reference(ax, table)
    _set_sweep_ticks(ax, table)
    apply_axis_text_style(
        ax,
        xlabel=xlabel,
        ylabel=r"$\phi$ [deg]",
        title="Cell and iris marker phase",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = _cell_iris_legend(ax, config)
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
        value_columns=("cell_phase_shift_deg", "iris_phase_shift_deg"),
        value_labels=("cell", "iris"),
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    _draw_sweep_reference(ax, table)
    _set_sweep_ticks(ax, table)
    apply_axis_text_style(
        ax,
        xlabel=xlabel,
        ylabel=r"$\Delta\phi$ [deg]",
        title="Cell and iris phase pickup",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = _cell_iris_legend(ax, config)
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
        iris_color = MARKER_COLORS.get(str(marker_name), "#333333")
        cell_color = _cell_color(iris_color)
        label = MARKER_LABELS.get(str(marker_name), str(marker_name))
        ax.plot(
            marker_table["sweep_value"],
            marker_table[value_columns[0]],
            marker="o",
            linewidth=2.0,
            linestyle="-",
            color=cell_color,
            label=f"{label} {value_labels[0]}",
        )
        ax.plot(
            marker_table["sweep_value"],
            marker_table[value_columns[1]],
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color=iris_color,
            label=f"{label} {value_labels[1]}",
        )


def _cell_color(base_color: str) -> str:
    white_fraction = 0.35
    rgb = mcolors.to_rgb(base_color)
    return mcolors.to_hex(
        tuple(
            channel + (1.0 - channel) * white_fraction
            for channel in rgb
        )
    )


def _draw_sweep_reference(ax: plt.Axes, table: pd.DataFrame) -> None:
    sweep_bases = pd.to_numeric(table["sweep_base"], errors="coerce").dropna().unique()
    if len(sweep_bases) != 1:
        raise ValueError(
            "geometry_phase_response must contain exactly one finite sweep_base"
        )
    ax.axvline(
        float(sweep_bases[0]),
        color="0.48",
        linestyle=":",
        linewidth=1.4,
        label="_sweep_reference",
        zorder=0,
    )


def _set_sweep_ticks(ax: plt.Axes, table: pd.DataFrame) -> None:
    sweep_values = (
        pd.to_numeric(table["sweep_value"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    ax.set_xticks(sweep_values.to_numpy())


def _cell_iris_legend(ax: plt.Axes, config: PlotConfig) -> plt.Legend:
    handles, labels = ax.get_legend_handles_labels()
    cell_indices = [index for index, label in enumerate(labels) if label.endswith(" cell")]
    iris_indices = [index for index, label in enumerate(labels) if label.endswith(" iris")]
    paired_indices = set(cell_indices + iris_indices)
    order = cell_indices + iris_indices + [
        index for index in range(len(labels)) if index not in paired_indices
    ]
    return ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=True,
        loc="best",
        fontsize=config.compact_legend_size,
        ncol=2,
    )


def _axis_label(sweep_axis: str) -> str:
    if sweep_axis in SIMULATION_AXIS_LABELS:
        return SIMULATION_AXIS_LABELS[sweep_axis]
    label = sweep_axis.removeprefix("sim_").replace("_", " ")
    return f"{label} [CST parameter]"


def _marker_slug(marker_name: str) -> str:
    return marker_name.lower().replace("/", "_").replace(" ", "_")
