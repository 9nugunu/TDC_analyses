"""Draw family overlays and Kyhl start/end phase rotations."""

from __future__ import annotations

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.polar_drawing import _draw_angle_guides, _wrap180
from deflector_tuning.visualization.polar_grouping import MARKER_ORDER, _format_position


KYHL_PHASE_RADIAL_RADIUS = 1.0


KYHL_PHASE_RADIAL_START_ALPHA = 0.50


KYHL_PHASE_RADIAL_END_ALPHA = 0.50


def _draw_family_overlay(
    ax,
    family_table: pd.DataFrame,
    title: str,
    *,
    config: PlotConfig,
    markers: tuple[str, ...] | None = None,
    guide_angles_deg: tuple[float, ...] = (),
) -> None:
    ax.set_title(title, fontsize=config.title_size, fontweight="bold", pad=12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.grid(color="0.88", linewidth=0.6)
    ax.spines["polar"].set_color("0.20")
    ax.spines["polar"].set_linewidth(0.8)
    if guide_angles_deg:
        _draw_angle_guides(ax, guide_angles_deg, config=config)

    positions = sorted(family_table["tune_position"].dropna().unique(), key=float)
    if not positions:
        return
    marker_names = markers or MARKER_ORDER
    radius_values = np.linspace(0.22, 1.0, len(positions))
    for radius, tune_position in zip(radius_values, positions, strict=True):
        position_rows = family_table[family_table["tune_position"] == tune_position]
        by_marker = {row["marker_name"]: row for _, row in position_rows.iterrows()}
        label_anchor_angle: float | None = None
        for marker in [name for name in marker_names if name in by_marker]:
            angle = np.deg2rad(float(by_marker[marker]["s_phase_deg"]))
            color = MARKER_COLORS.get(marker, "#444444")
            if label_anchor_angle is None or marker == "f_2pi3":
                label_anchor_angle = float(angle)
            ax.plot([angle, angle], [0.14, radius], color=color, alpha=0.28, linewidth=config.line_width)
            ax.scatter(
                [angle],
                [radius],
                marker="o",
                s=max(config.marker_size * 0.42, 14.0),
                facecolors=color,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.90,
                zorder=3,
            )
        if label_anchor_angle is not None:
            _annotate_overlay_position(ax, label_anchor_angle, radius, _format_position(tune_position), config=config)
    ax.set_yticks([])

    if len(marker_names) > 1:
        _draw_family_overlay_marker_legend(ax, config=config, markers=marker_names)


def _draw_kyhl_phase_pair_overlay(
    ax,
    pair_table: pd.DataFrame,
    title: str,
    pair_label: str,
    *,
    config: PlotConfig,
    guide_angles_deg: tuple[float, ...] = (),
) -> None:
    ax.set_title(title, fontsize=config.title_size, fontweight="bold", pad=12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.grid(color="0.88", linewidth=0.6)
    ax.spines["polar"].set_color("0.20")
    ax.spines["polar"].set_linewidth(0.8)
    if guide_angles_deg:
        _draw_angle_guides(ax, guide_angles_deg, config=config)

    ax.text(
        np.deg2rad(315.0),
        0.36,
        pair_label,
        color="0.25",
        fontsize=config.annotation_size,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.88},
    ).set_clip_on(False)
    marker_radii = dict(zip(MARKER_ORDER, np.linspace(0.54, 0.86, len(MARKER_ORDER)), strict=True))
    label_angle_offsets = dict(zip(MARKER_ORDER, np.linspace(11.0, -11.0, len(MARKER_ORDER)), strict=True))
    marker_size = max(config.marker_size * 0.42, 14.0)
    for marker in MARKER_ORDER:
        marker_rows = pair_table[pair_table["marker_name"] == marker]
        if marker_rows["_pair_endpoint"].nunique() < 2:
            continue
        by_endpoint = {row["_pair_endpoint"]: row for _, row in marker_rows.iterrows()}
        if "start" not in by_endpoint or "end" not in by_endpoint:
            continue
        start = by_endpoint["start"]
        end = by_endpoint["end"]
        start_deg = float(start["s_phase_deg"])
        end_deg = float(end["s_phase_deg"])
        delta_deg = _wrap180(end_deg - start_deg)
        radius = marker_radii[marker]
        color = MARKER_COLORS.get(marker, "#444444")
        theta_deg = np.linspace(start_deg, start_deg + delta_deg, 96)
        theta = np.deg2rad(theta_deg)
        _draw_kyhl_phase_radial_line(ax, start_deg, KYHL_PHASE_RADIAL_RADIUS, color, "start", config=config)
        _draw_kyhl_phase_radial_line(ax, start_deg + delta_deg, KYHL_PHASE_RADIAL_RADIUS, color, "end", config=config)
        arc_line = ax.plot(theta, np.full_like(theta, radius), color=color, alpha=0.74, linewidth=config.line_width)[0]
        arc_line.set_gid("kyhl_phase_rotation_arc")
        ax.scatter(
            [np.deg2rad(start_deg)],
            [radius],
            marker="o",
            s=marker_size,
            facecolors="white",
            edgecolors=color,
            linewidths=config.line_width * 0.55,
            zorder=3,
        )
        ax.scatter(
            [np.deg2rad(start_deg + delta_deg)],
            [radius],
            marker="o",
            s=marker_size,
            facecolors=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        _annotate_kyhl_rotation_arc(
            ax,
            marker,
            start_deg + delta_deg / 2.0 + label_angle_offsets.get(marker, 0.0),
            min(radius + 0.075, 1.04),
            delta_deg,
            config=config,
        )
    ax.set_yticks([])


def _draw_kyhl_phase_radial_line(
    ax,
    angle_deg: float,
    radius: float,
    color: str,
    endpoint: str,
    *,
    config: PlotConfig,
) -> None:
    theta = np.deg2rad(angle_deg)
    linestyle = ":" if endpoint == "start" else "-"
    alpha = KYHL_PHASE_RADIAL_START_ALPHA if endpoint == "start" else KYHL_PHASE_RADIAL_END_ALPHA
    line = ax.plot(
        [theta, theta],
        [0.0, radius],
        color=color,
        alpha=alpha,
        linewidth=max(config.line_width * 0.58, 0.8),
        linestyle=linestyle,
        zorder=1,
    )[0]
    line.set_gid(f"kyhl_phase_radial_{endpoint}")


def _draw_family_overlay_marker_legend(ax, *, config: PlotConfig, markers: tuple[str, ...]) -> None:
    handles: list[Line2D] = []
    labels: list[str] = []
    for marker in markers:
        color = MARKER_COLORS.get(marker, "#444444")
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=8,
                linewidth=1.6,
            )
        )
        labels.append(MARKER_LABELS.get(marker, marker))
    if not handles:
        return
    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.08, 0.50),
        frameon=True,
        fontsize=config.label_size,
        handlelength=1.3,
        borderpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.78)
    legend.get_frame().set_edgecolor("0.85")
    for text, marker in zip(legend.get_texts(), markers, strict=True):
        text.set_color(MARKER_COLORS.get(marker, "#444444"))
        text.set_fontweight(config.legend_weight)


def _annotate_kyhl_rotation_arc(
    ax,
    marker: str,
    angle_deg: float,
    radius: float,
    delta_deg: float,
    *,
    config: PlotConfig,
) -> None:
    color = MARKER_COLORS.get(marker, "#444444")
    theta = np.deg2rad(angle_deg)
    text = ax.text(
        theta,
        radius,
        f"{MARKER_LABELS.get(marker, marker)} $\\Delta$={delta_deg:+.1f} deg",
        color=color,
        fontsize=config.compact_annotation_size,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    text.set_clip_on(False)


def _annotate_overlay_position(ax, angle: float, radius: float, label: str, *, config: PlotConfig) -> None:
    angle_deg = float(np.rad2deg(angle))
    angle_offset_deg = 8.0 if np.cos(angle) >= 0 else -8.0
    text_angle = np.deg2rad(angle_deg + angle_offset_deg)
    text_radius = min(radius + 0.035, 1.04)
    ha = "left" if np.cos(angle) >= 0 else "right"
    text = ax.text(
        text_angle,
        text_radius,
        label,
        color="0.25",
        fontsize=config.annotation_size,
        ha=ha,
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.88},
    )
    text.set_clip_on(False)
