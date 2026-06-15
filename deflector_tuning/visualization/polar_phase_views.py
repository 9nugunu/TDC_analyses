"""Polar phase views for marker-sampled S-parameter phases."""

from __future__ import annotations

from collections import OrderedDict
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style, save_figure

MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
MARKER_LABELS: dict[str, str] = {
    "f_2pi3": r"$f_{2\pi/3}$",
    "f_mean": r"$f_{mean}$",
    "f_pi2": r"$f_{\pi/2}$",
}
REQUIRED_COLUMNS: tuple[str, ...] = ("tune_position", "marker_name", "s_phase_deg")


def plot_marker_phase_polar_views(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_prefix: str = "marker-frequency polar phase view",
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write one unit-circle polar phase view per tune position plus an overview."""

    if marker_points.empty:
        raise ValueError("marker_points is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: OrderedDict[str, Path] = OrderedDict()
    groups = list(_iter_position_groups(marker_points))
    for position_label, position_table in groups:
        fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
        _draw_position(ax, position_table, f"{position_label}: {title_prefix}", config=config)
        output_path = folder / f"position_{_safe_label(position_label)}.png"
        save_figure(fig, output_path, config)
        plt.close(fig)
        paths[position_label] = output_path

    overview_path = folder / "all_positions.png"
    _save_overview(groups, overview_path, title_prefix=title_prefix, config=config)
    paths["overview"] = overview_path
    return paths


def _iter_position_groups(marker_points: pd.DataFrame):
    table = marker_points.copy()
    table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table.sort_values(["_position_sort", "tune_position", "marker_name"], kind="mergesort")
    for position, group in table.groupby("tune_position", sort=False, dropna=False):
        yield _format_position(position), group


def _save_overview(groups: list[tuple[str, pd.DataFrame]], output_path: Path, *, title_prefix: str, config: PlotConfig) -> None:
    column_count = 2 if len(groups) > 1 else 1
    row_count = ceil(len(groups) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(config.overview_panel_size[0] * column_count, config.overview_panel_size[1] * row_count),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, (position_label, position_table) in zip(flat_axes, groups, strict=False):
        _draw_position(ax, position_table, f"{position_label}: {title_prefix}", compact=False, config=config)
    for ax in flat_axes[len(groups) :]:
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.32, hspace=0.44)
    save_figure(fig, output_path, config)
    plt.close(fig)


def _draw_position(ax, position_table: pd.DataFrame, title: str, *, compact: bool = False, config: PlotConfig) -> None:
    title_size = config.compact_title_size if compact else config.title_size
    label_size = config.compact_label_size if compact else config.label_size
    marker_size = config.compact_marker_size if compact else config.marker_size
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.grid(color="0.88", linewidth=0.6)
    ax.spines["polar"].set_color("0.20")
    ax.spines["polar"].set_linewidth(0.8)

    by_marker = {row["marker_name"]: row for _, row in position_table.iterrows()}
    available = [marker for marker in MARKER_ORDER if marker in by_marker]
    angles = {marker: np.deg2rad(float(by_marker[marker]["s_phase_deg"])) for marker in available}

    for marker in available:
        angle = angles[marker]
        ax.plot([angle, angle], [0, 1.0], color="red", alpha=0.55, linewidth=config.line_width)
        ax.scatter([angle], [1.0], marker="s", s=marker_size, facecolors="white", edgecolors="red", linewidths=config.line_width, zorder=3)
        text = ax.text(
            angle,
            1.08,
            MARKER_LABELS.get(marker, marker),
            color="black",
            fontsize=label_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "square,pad=0.12", "edgecolor": "red", "facecolor": "white", "linewidth": 0.6},
        )
        text.set_clip_on(False)

    _draw_spacing_arc(ax, angles, "f_2pi3", "f_mean", r"$\Delta\phi_{21}$", radius=0.70, compact=compact, config=config)
    _draw_spacing_arc(ax, angles, "f_mean", "f_pi2", r"$\Delta\phi_{32}$", radius=0.84, compact=compact, config=config)


def _draw_spacing_arc(ax, angles: dict[str, float], start_marker: str, end_marker: str, label: str, *, radius: float, compact: bool, config: PlotConfig) -> None:
    if start_marker not in angles or end_marker not in angles:
        return
    start_deg = np.rad2deg(angles[start_marker])
    end_deg = np.rad2deg(angles[end_marker])
    delta = _wrap180(end_deg - start_deg)
    theta_deg = np.linspace(start_deg, start_deg + delta, 64)
    theta = np.deg2rad(theta_deg)
    ax.plot(theta, np.full_like(theta, radius), color="red", linewidth=config.line_width)
    mid = np.deg2rad(start_deg + delta / 2.0)
    text = ax.text(
        mid,
        radius + 0.06,
        f"{label} = {delta:+.1f}°",
        color="red",
        fontsize=config.compact_annotation_size if compact else config.annotation_size,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=0,
        rotation_mode="anchor",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
    )
    text.set_clip_on(False)


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _format_position(position: object) -> str:
    try:
        value = float(position)
    except (TypeError, ValueError):
        return str(position)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _safe_label(label: str) -> str:
    return label.replace(".", "p").replace("-", "m").replace(" ", "_")
