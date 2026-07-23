"""Plots for experimental phase response to plunger offset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)


def plot_phase_offset_response(
    response: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Plot per-marker phase change versus experimental plunger offset."""

    required = ("marker_name", "plunger_offset_mm", "phase_delta_deg")
    missing = [column for column in required if column not in response]
    if missing:
        raise ValueError(f"phase offset response is missing required columns: {missing}")
    if response.empty:
        raise ValueError("phase offset response is empty")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for marker_name, marker_rows in response.groupby("marker_name", sort=False, dropna=False):
        marker_key = str(marker_name)
        means = (
            marker_rows.groupby("plunger_offset_mm", as_index=False, sort=True)["phase_delta_deg"]
            .mean()
        )
        color = MARKER_COLORS.get(marker_key, "#444444")
        ax.plot(
            means["plunger_offset_mm"],
            means["phase_delta_deg"],
            color=color,
            linewidth=config.line_width,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=MARKER_LABELS.get(marker_key, marker_key),
            zorder=3,
        )
        ax.scatter(
            marker_rows["plunger_offset_mm"],
            marker_rows["phase_delta_deg"],
            color=color,
            s=24,
            alpha=0.45,
            zorder=4,
        )

    ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=1)
    ax.axvline(0.0, color="0.75", linewidth=0.9, zorder=1)
    apply_axis_text_style(
        ax,
        xlabel="Plunger offset [mm]",
        ylabel="S-parameter phase change [deg]",
        title="Phase response to plunger offset",
        config=config,
        compact=True,
    )
    ax.grid(True, color="0.88", linewidth=0.8)
    legend = ax.legend(frameon=False, fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, folder / "phase_vs_plunger_offset.png", config)
    plt.close(fig)
    return path
