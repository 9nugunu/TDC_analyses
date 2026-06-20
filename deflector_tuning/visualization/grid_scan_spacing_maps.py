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
from deflector_tuning.visualization.plot_config import (
    BEST_MARKER_COLOR,
    DEFAULT_DESIGN_POINT_BY_AXIS,
    DESIGN_REFERENCE_LINEWIDTH,
    REFERENCE_GUIDE_ALPHA,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
    PlotConfig,
    apply_axis_text_style,
    apply_plot_style,
    contour_contrast_color,
    save_figure,
)

ERROR_METRICS: OrderedDict[str, tuple[str, str]] = OrderedDict(
    [
        ("spacing_60deg_target_error_deg", ("60deg_target_error", "60 deg target error [deg]")),
        ("spacing_equality_error_deg", ("equality_error", "Spacing equality error [deg]")),
    ]
)
SENSITIVITY_METRICS: OrderedDict[str, tuple[str, str]] = OrderedDict(
    [
        ("dphase_d_sim_r_c_deg_per_mm", (r"$\partial\phi/\partial r_c$ [deg/mm]", "contour_signed_cmap")),
        ("dphase_d_sim_w_c_deg_per_mm", (r"$\partial\phi/\partial w_c$ [deg/mm]", "contour_signed_cmap")),
        ("gradient_magnitude_deg_per_mm", (r"$|\nabla\phi|$ [deg/mm]", "contour_magnitude_cmap")),
    ]
)
MARKER_LABELS: dict[str, str] = {
    "f_2pi3": r"$f_{2\pi/3}$",
    "f_mean": r"$f_{mean}$",
    "f_pi2": r"$f_{\pi/2}$",
}


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
    for column, (key, label) in ERROR_METRICS.items():
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


def plot_grid_scan_phase_sensitivity_maps(
    sensitivity_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    x_column: str = "sim_r_c",
    y_column: str = "sim_w_c",
    design_point: tuple[float, float] | None = None,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write phase sensitivity maps for each marker and geometry parameter."""

    if sensitivity_summary.empty:
        raise ValueError("sensitivity_summary is empty")
    required = [x_column, y_column, "marker_name", *SENSITIVITY_METRICS.keys()]
    missing = [column for column in required if column not in sensitivity_summary]
    if missing:
        raise ValueError(f"sensitivity_summary is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    require_finite_plot_columns(
        sensitivity_summary,
        columns=(x_column, y_column),
        context="grid_scan sensitivity_summary coordinates",
        id_columns=("dataset_id", "source_file", "run_id", "marker_name", x_column, y_column),
    )
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    sensitivity_summary.to_csv(folder / "grid_scan_phase_sensitivity_summary.csv", index=False)

    paths: OrderedDict[str, Path] = OrderedDict()
    for marker_name, marker_table in sensitivity_summary.groupby("marker_name", sort=False, dropna=False):
        marker_key = _safe_key(marker_name)
        for column, (label, cmap_attribute) in SENSITIVITY_METRICS.items():
            finite_table = marker_table.dropna(subset=[column])
            if finite_table.empty:
                continue
            cmap = str(getattr(config, cmap_attribute))
            key = f"{marker_key}_{column.removesuffix('_deg_per_mm')}"
            paths[key] = _plot_grid_value_map(
                finite_table,
                folder / f"{key}.png",
                x_column,
                y_column,
                column,
                f"{_marker_label(marker_name)} {label}",
                config,
                design_point=design_point or _default_design_point(x_column, y_column),
                cmap=cmap,
                draw_best=False,
            )
    if not paths:
        raise ValueError("no finite grid-scan phase sensitivity values to plot")
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
    return _plot_grid_value_map(
        table,
        output_path,
        x_column,
        y_column,
        value_column,
        value_label,
        config,
        design_point=design_point,
        cmap=config.contour_error_cmap,
        draw_best=True,
    )


def _plot_grid_value_map(
    table: pd.DataFrame,
    output_path: Path,
    x_column: str,
    y_column: str,
    value_column: str,
    value_label: str,
    config: PlotConfig,
    *,
    design_point: tuple[float, float] | None,
    cmap: str,
    draw_best: bool,
) -> Path:
    x_values = np.array(sorted(table[x_column].dropna().unique()), dtype=float)
    y_values = np.array(sorted(table[y_column].dropna().unique()), dtype=float)
    pivot = table.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean").reindex(index=y_values, columns=x_values)
    X, Y = np.meshgrid(x_values, y_values)
    Z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    is_contour_map = len(x_values) >= 2 and len(y_values) >= 2
    if is_contour_map:
        contour_color = contour_contrast_color(cmap, config)
        cf = ax.contourf(X, Y, Z, levels=12, cmap=cmap)
        cs = ax.contour(
            X,
            Y,
            Z,
            levels=8,
            colors=contour_color,
            linewidths=config.contour_line_width,
            alpha=config.contour_line_alpha,
        )
        contour_labels = ax.clabel(cs, inline=True, fontsize=config.contour_label_size, fmt="%.1f", colors=contour_color)
        for label in contour_labels:
            label.set_fontweight(config.contour_label_weight)
    else:
        cf = ax.scatter(table[x_column], table[y_column], c=table[value_column], cmap=cmap, s=90, edgecolor="black")
    if draw_best:
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


def _safe_key(value: object) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _marker_label(value: object) -> str:
    return MARKER_LABELS.get(str(value), str(value).replace("_", r"\_"))


def _default_design_point(x_column: str, y_column: str) -> tuple[float, float] | None:
    if x_column not in DEFAULT_DESIGN_POINT_BY_AXIS or y_column not in DEFAULT_DESIGN_POINT_BY_AXIS:
        return None
    return (DEFAULT_DESIGN_POINT_BY_AXIS[x_column], DEFAULT_DESIGN_POINT_BY_AXIS[y_column])


def _draw_design_crosshair(ax: plt.Axes, design_point: tuple[float, float]) -> None:
    design_x, design_y = design_point
    ax.axvline(
        design_x,
        color=REFERENCE_GUIDE_COLOR,
        linestyle=REFERENCE_GUIDE_LINESTYLE,
        linewidth=DESIGN_REFERENCE_LINEWIDTH,
        alpha=REFERENCE_GUIDE_ALPHA,
        zorder=5,
    )
    ax.axhline(
        design_y,
        color=REFERENCE_GUIDE_COLOR,
        linestyle=REFERENCE_GUIDE_LINESTYLE,
        linewidth=DESIGN_REFERENCE_LINEWIDTH,
        alpha=REFERENCE_GUIDE_ALPHA,
        zorder=5,
    )
