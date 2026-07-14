"""Complex-plane sweep plots for direct CST one-port Y11/Z11 exports."""

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


def plot_y11_marker_sweep(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Draw direct CST Y11 trajectories in the actual complex-admittance plane."""

    return _plot_one_port_marker_sweep(
        marker_points,
        output_path,
        real_column="y_re_siemens",
        imag_column="y_im_siemens",
        scale=1e3,
        parameter_label="Y_{11}",
        unit="mS",
        title="CST direct $Y_{11}$ sweep at dispersion markers",
        config=config,
    )


def plot_z11_marker_sweep(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Draw direct CST Z11 trajectories in the complex-impedance plane."""

    return _plot_one_port_marker_sweep(
        marker_points,
        output_path,
        real_column="z_re_ohm",
        imag_column="z_im_ohm",
        scale=1.0,
        parameter_label="Z_{11}",
        unit=r"$\Omega$",
        title="CST direct $Z_{11}$ sweep at dispersion markers",
        config=config,
    )


def _plot_one_port_marker_sweep(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    real_column: str,
    imag_column: str,
    scale: float,
    parameter_label: str,
    unit: str,
    title: str,
    config: PlotConfig | None,
) -> Path:
    required_columns = {"marker_name", "source_file", real_column, imag_column}
    missing = required_columns.difference(marker_points.columns)
    if missing:
        raise ValueError(
            f"One-port marker points are missing required columns: {sorted(missing)}"
        )
    if marker_points.empty:
        raise ValueError("One-port marker points must not be empty")

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
        real_values = group[real_column].astype(float) * scale
        imag_values = group[imag_column].astype(float) * scale
        label = MARKER_LABELS.get(marker_name, marker_name)
        ax.plot(real_values, imag_values, color=color, linewidth=config.line_width, label=label, zorder=2)
        ax.scatter(real_values.iloc[0], imag_values.iloc[0], marker="s", s=config.marker_size, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        if len(group) > 1:
            ax.scatter(real_values.iloc[-1], imag_values.iloc[-1], marker="^", s=config.marker_size, color=color, edgecolors="white", linewidths=0.8, zorder=3)

    ax.set_xlabel(rf"Re(${parameter_label}$) [{unit}]")
    ax.set_ylabel(rf"Im(${parameter_label}$) [{unit}]")
    ax.set_title(title, fontsize=config.title_size, fontweight=config.title_weight)
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
