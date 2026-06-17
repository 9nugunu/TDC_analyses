"""Dispersion curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_plot_style,
    save_figure,
)

REQUIRED_COLUMNS: tuple[str, ...] = ("mode_index", "phase_deg", "freq_GHz")


def plot_dispersion_curves(
    dispersion_table: pd.DataFrame,
    output_path: str | Path,
    *,
    mode_indices: tuple[int, ...] | None = None,
    title: str = "CST dispersion: frequency vs phase advance",
    config: PlotConfig | None = None,
) -> Path:
    """Write a frequency-vs-phase dispersion curve PNG."""

    if dispersion_table.empty:
        raise ValueError("dispersion_table is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in dispersion_table]
    if missing:
        raise ValueError(f"dispersion_table is missing required columns: {missing}")

    config = _dispersion_plot_config(config)
    apply_plot_style(config)
    table = dispersion_table.copy()
    if mode_indices is not None:
        table = table[table["mode_index"].isin(mode_indices)]
    if table.empty:
        raise ValueError("No dispersion rows remain after mode filtering")
    require_finite_plot_columns(table, columns=("phase_deg", "freq_GHz"), context="dispersion_table")

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    line_width = max(config.line_width * 1.55, 2.6)
    marker_size = max(config.marker_size**0.5 * 0.95, 6.2)
    for mode_index, group in table.groupby("mode_index", sort=True):
        group = group.sort_values("phase_deg")
        line = ax.plot(
            group["phase_deg"],
            group["freq_GHz"],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            label="_nolegend_",
        )[0]
        _label_curve_end(ax, group, f"Mode {int(mode_index)}", color=line.get_color(), config=config)

    apply_axis_text_style(
        ax,
        xlabel="Phase advance (deg)",
        ylabel="Frequency (GHz)",
        title=title,
        config=config,
    )
    ax.grid(True, which="major", color="0.76", linewidth=0.9, alpha=0.7)
    ax.grid(True, which="minor", color="0.88", linestyle=":", linewidth=0.75, alpha=0.7)
    ax.minorticks_on()
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _label_curve_end(ax, group: pd.DataFrame, label: str, *, color: str, config: PlotConfig) -> None:
    """Place compact direct labels at curve endpoints instead of a separate legend."""

    if group.empty:
        return
    point = group.sort_values("phase_deg").iloc[-1]
    ax.annotate(
        label,
        xy=(float(point["phase_deg"]), float(point["freq_GHz"])),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        fontsize=config.annotation_size,
        fontweight=config.legend_weight,
        va="center",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        clip_on=False,
    )


def _dispersion_plot_config(config: PlotConfig | None) -> PlotConfig:
    if config is not None:
        return config
    return PlotConfig(
        title_size=22,
        label_size=18,
        tick_size=15,
        annotation_size=14,
        legend_size=13,
        line_width=2.2,
        marker_size=72,
    )
