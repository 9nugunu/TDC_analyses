"""Dispersion curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import PlotConfig, apply_axis_text_style, apply_plot_style, save_figure

REQUIRED_COLUMNS: tuple[str, ...] = ("mode_index", "phase_deg", "freq_GHz")


def plot_dispersion_curves(
    dispersion_table: pd.DataFrame,
    output_path: str | Path,
    *,
    mode_indices: tuple[int, ...] | None = None,
    title: str = "CST dispersion curves",
    config: PlotConfig | None = None,
) -> Path:
    """Write a frequency-vs-phase dispersion curve PNG."""

    if dispersion_table.empty:
        raise ValueError("dispersion_table is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in dispersion_table]
    if missing:
        raise ValueError(f"dispersion_table is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    table = dispersion_table.copy()
    if mode_indices is not None:
        table = table[table["mode_index"].isin(mode_indices)]
    if table.empty:
        raise ValueError("No dispersion rows remain after mode filtering")
    require_finite_plot_columns(table, columns=("phase_deg", "freq_GHz"), context="dispersion_table")

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    for mode_index, group in table.groupby("mode_index", sort=True):
        group = group.sort_values("phase_deg")
        ax.plot(
            group["phase_deg"],
            group["freq_GHz"],
            marker="o",
            markersize=3.2,
            linewidth=1.8,
            label=f"Mode {int(mode_index)}",
        )

    apply_axis_text_style(
        ax,
        xlabel="Phase advance (deg)",
        ylabel="Frequency (GHz)",
        title=title,
        config=config,
    )
    ax.grid(True, which="major", color="0.78", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="0.90", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.legend(frameon=True, loc="best", ncol=2 if table["mode_index"].nunique() > 6 else 1)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path
