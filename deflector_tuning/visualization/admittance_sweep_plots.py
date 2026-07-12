"""Actual complex-Y11 sweep plots for direct CST admittance exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from deflector_tuning.markers.frequency_markers import MARKER_ORDER
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style, save_figure


REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "source_file",
    "y_real_siemens",
    "y_imag_siemens",
)


def plot_y11_marker_sweep(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Draw direct CST Y11 trajectories in the actual complex-admittance plane."""

    missing = set(REQUIRED_COLUMNS).difference(marker_points.columns)
    if missing:
        raise ValueError(f"Y11 marker points are missing required columns: {sorted(missing)}")
    if marker_points.empty:
        raise ValueError("Y11 marker points must not be empty")

    config = config or PlotConfig()
    apply_plot_style(config)
    fig, ax = plt.subplots(figsize=(7.6, 6.1), constrained_layout=True)
    ax.axhline(0.0, color="0.70", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.70", linewidth=0.8, zorder=0)

    for marker_name in MARKER_ORDER:
        group = marker_points[marker_points["marker_name"] == marker_name].copy()
        if group.empty:
            continue
        group = _sort_sweep(group)
        color = MARKER_COLORS.get(marker_name, "#444444")
        real_ms = group["y_real_siemens"].astype(float) * 1e3
        imag_ms = group["y_imag_siemens"].astype(float) * 1e3
        label = MARKER_LABELS.get(marker_name, marker_name)
        ax.plot(real_ms, imag_ms, color=color, linewidth=config.line_width, label=label, zorder=2)
        ax.scatter(real_ms.iloc[0], imag_ms.iloc[0], marker="s", s=config.marker_size, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        if len(group) > 1:
            ax.scatter(real_ms.iloc[-1], imag_ms.iloc[-1], marker="^", s=config.marker_size, color=color, edgecolors="white", linewidths=0.8, zorder=3)

    ax.set_xlabel(r"Re($Y_{11}$) [mS]")
    ax.set_ylabel(r"Im($Y_{11}$) [mS]")
    ax.set_title("CST direct $Y_{11}$ sweep at dispersion markers", fontsize=config.title_size, fontweight=config.title_weight)
    marker_handles, marker_labels = ax.get_legend_handles_labels()
    sweep_handles = [
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="0.35", markeredgecolor="white", label="first $r_c$"),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="0.35", markeredgecolor="white", label="last $r_c$"),
    ]
    ax.legend([*marker_handles, *sweep_handles], [*marker_labels, "first $r_c$", "last $r_c$"], loc="best", frameon=True)
    ax.grid(True, color="0.90", linewidth=0.7)
    save_figure(fig, output_path, config)
    plt.close(fig)
    return Path(output_path)


def _sort_sweep(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if "sim_r_c" in output:
        output["_sweep_sort"] = pd.to_numeric(output["sim_r_c"], errors="coerce")
    elif "run_id" in output:
        output["_sweep_sort"] = pd.to_numeric(output["run_id"], errors="coerce")
    else:
        output["_sweep_sort"] = range(len(output))
    return output.sort_values(["_sweep_sort", "source_file"], kind="mergesort")
