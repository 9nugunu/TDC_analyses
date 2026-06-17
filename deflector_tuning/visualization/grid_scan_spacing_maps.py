"""Contour maps for simulation grid-scan marker-spacing errors."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import PlotConfig, apply_axis_text_style, apply_plot_style, save_figure

ERROR_METRICS: OrderedDict[str, str] = OrderedDict(
    [
        ("spacing_60deg_target_error_deg", "60 deg target error [deg]"),
        ("spacing_equality_error_deg", "Spacing equality error [deg]"),
    ]
)
BEST_MARKER_COLOR = "#c51b7d"
DESIGN_CROSSHAIR_COLOR = "0.45"
DESIGN_CROSSHAIR_LINESTYLE = "--"
DESIGN_CROSSHAIR_LINEWIDTH = 0.9
DEFAULT_DESIGN_POINT_BY_AXIS = {"sim_r_c": 56.59, "sim_w_c": 19.3224}


def plot_grid_scan_spacing_error_maps(
    spacing_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    x_column: str = "sim_r_c",
    y_column: str = "sim_w_c",
    design_point: tuple[float, float] | None = None,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write the two requested r_c x w_c spacing-error maps."""

    if spacing_summary.empty:
        raise ValueError("spacing_summary is empty")
    required = [x_column, y_column, *ERROR_METRICS.keys()]
    missing = [column for column in required if column not in spacing_summary]
    if missing:
        raise ValueError(f"spacing_summary is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    require_finite_plot_columns(
        spacing_summary,
        columns=("sim_r_c", "sim_w_c", *ERROR_METRICS.keys()),
        context="grid_scan spacing_summary",
        id_columns=("dataset_id", "source_file", "run_id", "sim_r_c", "sim_w_c"),
    )
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    spacing_summary.to_csv(folder / "grid_scan_spacing_summary.csv", index=False)

    paths: OrderedDict[str, Path] = OrderedDict()
    for column, label in ERROR_METRICS.items():
        key = column.removesuffix("_deg")
        paths[key] = _plot_error_map(
            spacing_summary,
            folder / f"{key}.png",
            x_column,
            y_column,
            column,
            label,
            config,
            design_point=design_point or _default_design_point(x_column, y_column),
        )
    return paths


def _plot_error_map(
    table: pd.DataFrame,
    output_path: Path,
    x_column: str,
    y_column: str,
    value_column: str,
    value_label: str,
    config: PlotConfig,
    *,
    design_point: tuple[float, float] | None,
) -> Path:
    x_values = np.array(sorted(table[x_column].dropna().unique()), dtype=float)
    y_values = np.array(sorted(table[y_column].dropna().unique()), dtype=float)
    pivot = table.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean").reindex(index=y_values, columns=x_values)
    X, Y = np.meshgrid(x_values, y_values)
    Z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    is_contour_map = len(x_values) >= 2 and len(y_values) >= 2
    if is_contour_map:
        cf = ax.contourf(X, Y, Z, levels=12, cmap="RdYlGn_r")
        cs = ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.7, alpha=0.75)
        ax.clabel(cs, inline=True, fontsize=max(7, config.annotation_size - 2), fmt="%.1f")
    else:
        cf = ax.scatter(table[x_column], table[y_column], c=table[value_column], cmap="RdYlGn_r", s=90, edgecolor="black")
    best = table.loc[table[value_column].idxmin()]
    ax.scatter([best[x_column]], [best[y_column]], marker="*", s=190, c=BEST_MARKER_COLOR, edgecolor="black", linewidth=0.8, zorder=6)
    if design_point is not None:
        _draw_design_crosshair(ax, design_point)
    apply_axis_text_style(
        ax,
        xlabel=r"$r_c$ [mm]",
        ylabel=r"$w_c$ [mm]",
        title=value_label,
        config=config,
    )
    colorbar = fig.colorbar(cf, ax=ax, shrink=0.92)
    colorbar.set_label(value_label, fontsize=config.label_size, fontweight=config.label_weight, rotation=-90, labelpad=30)
    colorbar.ax.tick_params(labelsize=config.tick_size)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight(config.tick_weight)
    ax.grid(True, color="0.85", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _default_design_point(x_column: str, y_column: str) -> tuple[float, float] | None:
    if x_column not in DEFAULT_DESIGN_POINT_BY_AXIS or y_column not in DEFAULT_DESIGN_POINT_BY_AXIS:
        return None
    return (DEFAULT_DESIGN_POINT_BY_AXIS[x_column], DEFAULT_DESIGN_POINT_BY_AXIS[y_column])


def _draw_design_crosshair(ax: plt.Axes, design_point: tuple[float, float]) -> None:
    design_x, design_y = design_point
    ax.axvline(
        design_x,
        color=DESIGN_CROSSHAIR_COLOR,
        linestyle=DESIGN_CROSSHAIR_LINESTYLE,
        linewidth=DESIGN_CROSSHAIR_LINEWIDTH,
        alpha=0.85,
        zorder=5,
    )
    ax.axhline(
        design_y,
        color=DESIGN_CROSSHAIR_COLOR,
        linestyle=DESIGN_CROSSHAIR_LINESTYLE,
        linewidth=DESIGN_CROSSHAIR_LINEWIDTH,
        alpha=0.85,
        zorder=5,
    )
