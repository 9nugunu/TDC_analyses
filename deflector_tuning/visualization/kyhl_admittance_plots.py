from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style, save_figure

REQUIRED_POINT_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "tune_position",
    "op_admit_ang_deg",
    "op_admit_axis_deg",
    "op_admit_axis_err_deg",
)
REQUIRED_TRANSITION_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "position_family",
    "pos_from",
    "pos_to",
    "op_admit_ang_deg",
    "op_admit_axis_deg",
    "op_admit_axis_err_deg",
)
F2PI3_MARKER = "f_2pi3"
NORMALIZED_POINT_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "source_file",
    "tune_position",
    "mode_admit_re",
    "mode_admit_im",
    "mode_gamma_re",
    "mode_gamma_im",
    "mode_gamma_mag",
    "mode_gamma_ang_deg",
)


def plot_kyhl_operation_polar(
    points: pd.DataFrame,
    transitions: pd.DataFrame,
    output_path: str | Path,
    *,
    marker_name: str = "f_2pi3",
    config: PlotConfig | None = None,
) -> Path:
    """Plot KYHL point and transition angles against operation-mode axes."""

    _require_columns(points, REQUIRED_POINT_COLUMNS, context="KYHL points")
    _require_columns(transitions, REQUIRED_TRANSITION_COLUMNS, context="KYHL transitions")
    point_table = points[points["marker_name"] == marker_name].copy()
    transition_table = transitions[transitions["marker_name"] == marker_name].copy()
    if point_table.empty:
        raise ValueError(f"No KYHL point rows found for marker_name={marker_name!r}")
    if transition_table.empty:
        raise ValueError(f"No KYHL transition rows found for marker_name={marker_name!r}")

    config = config or PlotConfig()
    apply_plot_style(config)
    axes_deg = _operation_axes(point_table, transition_table)
    operation_mode = _first_float(point_table, "op_mode_deg", fallback=_first_float(transition_table, "op_mode_deg"))
    marker_label = MARKER_LABELS.get(marker_name, marker_name)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.4),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    _draw_operation_axes(axes[0], axes_deg, config=config)
    _draw_operation_axes(axes[1], axes_deg, config=config)
    _draw_points(axes[0], point_table, marker_name=marker_name, config=config)
    _draw_transitions(axes[1], transition_table, marker_name=marker_name, config=config)

    title = f"KYHL operation-mode admittance check ({marker_label}, {operation_mode:g} deg)"
    fig.suptitle(title, fontsize=config.title_size, fontweight=config.title_weight)
    save_figure(fig, output_path, config)
    plt.close(fig)
    return Path(output_path)


def plot_f2pi3_normalized_admittance_view(
    points: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return OrderedDict(
        [
            (
                F2PI3_MARKER,
                plot_f2pi3_normalized_admittance(
                    points,
                    folder / "f_2pi3_normalized_admittance.png",
                    config=config,
                ),
            )
        ]
    )


def plot_f2pi3_normalized_admittance(
    points: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    _require_columns(points, NORMALIZED_POINT_COLUMNS, context="KYHL normalized admittance points")
    point_table = points[points["marker_name"] == F2PI3_MARKER].copy()
    if point_table.empty:
        raise ValueError("No normalized admittance point rows found for marker_name='f_2pi3'")

    config = config or PlotConfig()
    apply_plot_style(config)
    point_table["_tune_sort"] = pd.to_numeric(point_table["tune_position"], errors="coerce")
    point_table = point_table.sort_values(
        ["_tune_sort", "source_file"] if "source_file" in point_table else ["_tune_sort"],
        kind="mergesort",
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8), subplot_kw={"projection": "polar"}, constrained_layout=True)
    color = MARKER_COLORS.get(F2PI3_MARKER, "#444444")
    _draw_smith_polar_reference(ax, config=config)
    theta = np.deg2rad(point_table["mode_gamma_ang_deg"].astype(float))
    radius = point_table["mode_gamma_mag"].astype(float)
    ax.scatter(
        theta,
        radius,
        s=config.marker_size,
        marker="s",
        facecolors="white",
        edgecolors=color,
        linewidths=config.line_width,
        zorder=4,
    )
    ax.set_title(
        "f2pi/3 normalized-admittance polar",
        fontsize=min(config.title_size, 18),
        fontweight=config.title_weight,
        pad=18,
    )
    save_figure(fig, output_path, config)
    plt.close(fig)
    return Path(output_path)


def plot_kyhl_operation_polar_views(
    points: pd.DataFrame,
    transitions: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    _require_columns(points, REQUIRED_POINT_COLUMNS, context="KYHL points")
    _require_columns(transitions, REQUIRED_TRANSITION_COLUMNS, context="KYHL transitions")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return OrderedDict(
        [
            (
                F2PI3_MARKER,
                plot_kyhl_operation_polar(
                    points,
                    transitions,
                    folder / "f_2pi3_admittance_polar.png",
                    marker_name=F2PI3_MARKER,
                    config=config,
                ),
            )
        ]
    )


def _draw_operation_axes(ax, axes_deg: tuple[float, ...], *, config: PlotConfig) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 1.18)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0.0, 90.0, 180.0, 270.0]))
    ax.set_xticklabels([])
    ax.grid(color="0.88", linewidth=0.7)
    ax.spines["polar"].set_color("0.20")
    ax.spines["polar"].set_linewidth(0.8)
    for axis_deg in axes_deg:
        theta = np.deg2rad(axis_deg)
        ax.plot([theta, theta], [0.0, 1.03], color="0.35", linestyle="--", linewidth=1.2, alpha=0.80)
        text = ax.text(
            theta,
            1.10,
            f"{axis_deg:g} deg",
            color="0.25",
            fontsize=config.annotation_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        text.set_clip_on(False)


def _draw_smith_polar_reference(ax, *, config: PlotConfig) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad([0.0, 90.0, 180.0, 270.0]))
    ax.set_xticklabels([])
    ax.grid(color="0.88", linewidth=0.6)
    ax.spines["polar"].set_color("0.20")
    ax.spines["polar"].set_linewidth(0.8)
    for angle_deg, label in ((120.0, "pi/2"), (180.0, "mean"), (240.0, "2pi/3")):
        theta = np.deg2rad(angle_deg)
        ax.plot([theta, theta], [0.0, 1.02], color="0.35", linestyle="--", linewidth=max(config.line_width - 0.2, 0.8), alpha=0.55)
        text = ax.text(
            theta,
            1.07,
            f"{angle_deg:.0f} deg\n{label}",
            color="0.25",
            fontsize=config.annotation_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        text.set_clip_on(False)


def _draw_points(ax, table: pd.DataFrame, *, marker_name: str, config: PlotConfig) -> None:
    color = MARKER_COLORS.get(marker_name, "#444444")
    table = table.copy()
    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table.sort_values(["_tune_sort", "source_file"] if "source_file" in table else ["_tune_sort"], kind="mergesort")
    radii = np.linspace(0.55, 0.95, len(table))
    label_offsets = np.linspace(-14.0, 14.0, len(table)) if len(table) > 1 else np.array([8.0])
    for radius, label_offset, (_, row) in zip(radii, label_offsets, table.iterrows(), strict=True):
        angle_deg = float(row["op_admit_ang_deg"])
        angle = np.deg2rad(angle_deg)
        if angle_deg > 180.0 and label_offset > 0.0:
            label_offset *= -1.0
        tune_label = _format_position(row["tune_position"])
        error = float(row["op_admit_axis_err_deg"])
        ax.plot([angle, angle], [0.0, radius], color=color, linewidth=1.4, alpha=0.50)
        ax.scatter([angle], [radius], s=70, color=color, edgecolors="white", linewidths=0.8, zorder=4)
        _annotate(
            ax,
            angle,
            min(radius + 0.08, 1.05),
            f"{tune_label}\n{error:+.1f} deg",
            angle_offset_deg=float(label_offset),
            config=config,
        )
    ax.set_title("sampled points", fontsize=config.compact_title_size, fontweight=config.title_weight, pad=14)


def _draw_transitions(ax, table: pd.DataFrame, *, marker_name: str, config: PlotConfig) -> None:
    marker_color = MARKER_COLORS.get(marker_name, "#444444")
    family_colors = {"cell": "#9467bd", "iris": "#ff7f0e"}
    table = table.copy()
    table["_from_sort"] = pd.to_numeric(table["pos_from"], errors="coerce")
    table = table.sort_values(["position_family", "_from_sort"], kind="mergesort")
    radii = np.linspace(0.68, 0.95, len(table))
    label_offsets = np.linspace(-10.0, 10.0, len(table)) if len(table) > 1 else np.array([8.0])
    handles: list[Line2D] = []
    labels: list[str] = []
    for radius, label_offset, (_, row) in zip(radii, label_offsets, table.iterrows(), strict=True):
        family = str(row["position_family"])
        color = family_colors.get(family, marker_color)
        angle = np.deg2rad(float(row["op_admit_ang_deg"]))
        error = float(row["op_admit_axis_err_deg"])
        transition_label = f"{family} {_format_position(row['pos_from'])}->{_format_position(row['pos_to'])}"
        ax.plot([angle, angle], [0.0, radius], color=color, linewidth=1.8, alpha=0.72)
        ax.scatter([angle], [radius], marker="s", s=80, color=color, edgecolors="white", linewidths=0.8, zorder=4)
        _annotate(
            ax,
            angle,
            min(radius + 0.09, 1.06),
            f"{transition_label}\n{error:+.1f} deg",
            angle_offset_deg=float(label_offset),
            config=config,
        )
        handles.append(Line2D([0], [0], marker="s", color=color, markerfacecolor=color, linewidth=1.8, markersize=7))
        labels.append(transition_label)
    ax.set_title("transition vectors", fontsize=config.compact_title_size, fontweight=config.title_weight, pad=14)
    legend = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.50, -0.16), ncol=1, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.85")


def _annotate(ax, angle: float, radius: float, label: str, *, angle_offset_deg: float, config: PlotConfig) -> None:
    offset = np.deg2rad(angle_offset_deg)
    label_angle = angle + offset
    ha = "left" if np.cos(label_angle) >= 0 else "right"
    text = ax.text(
        label_angle,
        radius,
        label,
        fontsize=config.compact_annotation_size,
        color="0.16",
        ha=ha,
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.90},
    )
    text.set_clip_on(False)


def _operation_axes(points: pd.DataFrame, transitions: pd.DataFrame) -> tuple[float, ...]:
    for table in (points, transitions):
        if "op_admit_axes_deg" in table and table["op_admit_axes_deg"].notna().any():
            text = str(table["op_admit_axes_deg"].dropna().iloc[0])
            return tuple(float(part) for part in text.split(";") if part)
    values = pd.concat(
        [
            points["op_admit_axis_deg"].dropna().astype(float),
            transitions["op_admit_axis_deg"].dropna().astype(float),
        ],
        ignore_index=True,
    )
    return tuple(sorted(values.unique().tolist()))


def _first_float(table: pd.DataFrame, column: str, *, fallback: float = float("nan")) -> float:
    if column not in table or table[column].dropna().empty:
        return fallback
    return float(table[column].dropna().iloc[0])


def _format_position(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return str(value)
    number = float(number)
    if number.is_integer():
        return f"{number:.1f}"
    return f"{number:g}"


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], *, context: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{context} table is missing required columns: {missing}")
