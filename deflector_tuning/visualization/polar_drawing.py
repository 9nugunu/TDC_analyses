"""Draw individual polar positions, phase spacing, and shared angle guides."""

from __future__ import annotations

from matplotlib.colors import to_hex, to_rgb
import numpy as np
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    REFERENCE_GUIDE_ALPHA,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LABEL_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
    PlotConfig,
)
from deflector_tuning.visualization.polar_grouping import MARKER_ORDER


MARKER_LABEL_CLUSTER_THRESHOLD_DEG = 22.0


MARKER_LABEL_CLUSTER_SPREAD_DEG = 10.0


MARKER_LABEL_BASE_RADIUS = 1.08


MARKER_LABEL_RADIUS_STEP = 0.04


def _draw_position(
    ax,
    position_table: pd.DataFrame,
    title: str,
    *,
    compact: bool = False,
    config: PlotConfig,
    guide_angles_deg: tuple[float, ...] = (),
) -> None:
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
    if guide_angles_deg:
        _draw_angle_guides(ax, guide_angles_deg, config=config)

    by_marker = {row["marker_name"]: row for _, row in position_table.iterrows()}
    available = [marker for marker in MARKER_ORDER if marker in by_marker]
    angles = {marker: np.deg2rad(float(by_marker[marker]["s_phase_deg"])) for marker in available}
    marker_label_layout = _marker_label_layout(available, angles)

    for marker in available:
        angle = angles[marker]
        text_angle, text_radius = marker_label_layout.get(marker, (angle, MARKER_LABEL_BASE_RADIUS))
        color = MARKER_COLORS.get(marker, "#444444")
        ax.plot([angle, angle], [0, 1.0], color=color, alpha=0.65, linewidth=config.line_width)
        ax.scatter([angle], [1.0], marker="s", s=marker_size, facecolors="white", edgecolors=color, linewidths=config.line_width, zorder=3)
        if not np.isclose(text_angle, angle) or not np.isclose(text_radius, MARKER_LABEL_BASE_RADIUS):
            ax.plot([angle, text_angle], [1.0, text_radius], color=color, alpha=0.32, linewidth=0.8, zorder=2)
        text = ax.text(
            text_angle,
            text_radius,
            MARKER_LABELS.get(marker, marker),
            color="black",
            fontsize=label_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "square,pad=0.12", "edgecolor": color, "facecolor": "white", "linewidth": 0.6},
        )
        text.set_clip_on(False)

    spacing_is_clustered = _has_clustered_marker_angles(available, angles)
    spacing_radii = (0.48, 0.58) if spacing_is_clustered else (0.70, 0.84)
    spacing_angle_offsets = (18.0, -18.0) if spacing_is_clustered else (0.0, 0.0)
    spacing_text_alignments = ("center", "left") if spacing_is_clustered else ("center", "center")
    _draw_spacing_arc(
        ax,
        angles,
        "f_2pi3",
        "f_mean",
        r"$\Delta\phi_{21}$",
        radius=spacing_radii[0],
        label_angle_offset_deg=spacing_angle_offsets[0],
        label_horizontal_alignment=spacing_text_alignments[0],
        compact=compact,
        config=config,
    )
    _draw_spacing_arc(
        ax,
        angles,
        "f_mean",
        "f_pi2",
        r"$\Delta\phi_{32}$",
        radius=spacing_radii[1],
        label_angle_offset_deg=spacing_angle_offsets[1],
        label_horizontal_alignment=spacing_text_alignments[1],
        compact=compact,
        config=config,
    )


def _draw_angle_guides(ax, angles_deg: tuple[float, ...], *, config: PlotConfig) -> None:
    for angle_deg in angles_deg:
        theta = np.deg2rad(angle_deg)
        ax.plot(
            [theta, theta],
            [0.0, 1.02],
            color=REFERENCE_GUIDE_COLOR,
            linestyle=REFERENCE_GUIDE_LINESTYLE,
            linewidth=max(config.line_width - 0.2, 0.8),
            alpha=REFERENCE_GUIDE_ALPHA,
            zorder=1,
        )
        text = ax.text(
            theta,
            1.07,
            f"{angle_deg:.0f}°",
            color=REFERENCE_GUIDE_LABEL_COLOR,
            fontsize=config.annotation_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        text.set_clip_on(False)


def _marker_label_layout(markers: list[str], angles: dict[str, float]) -> dict[str, tuple[float, float]]:
    if len(markers) < 2:
        return {marker: (angles[marker], MARKER_LABEL_BASE_RADIUS) for marker in markers}

    angle_deg_by_marker = _unwrapped_marker_degrees(markers, angles)
    sorted_markers = sorted(markers, key=lambda marker: angle_deg_by_marker[marker])
    layout: dict[str, tuple[float, float]] = {}
    cluster: list[str] = []

    for marker in sorted_markers:
        if not cluster:
            cluster = [marker]
            continue
        previous = cluster[-1]
        if angle_deg_by_marker[marker] - angle_deg_by_marker[previous] <= MARKER_LABEL_CLUSTER_THRESHOLD_DEG:
            cluster.append(marker)
            continue
        _assign_marker_label_cluster(layout, cluster, angle_deg_by_marker)
        cluster = [marker]
    _assign_marker_label_cluster(layout, cluster, angle_deg_by_marker)
    return layout


def _assign_marker_label_cluster(
    layout: dict[str, tuple[float, float]],
    cluster: list[str],
    angle_deg_by_marker: dict[str, float],
) -> None:
    if len(cluster) == 1:
        marker = cluster[0]
        layout[marker] = (np.deg2rad(angle_deg_by_marker[marker]), MARKER_LABEL_BASE_RADIUS)
        return
    cluster_midpoint = (len(cluster) - 1) / 2.0
    for index, marker in enumerate(cluster):
        offset_index = index - cluster_midpoint
        label_angle_deg = angle_deg_by_marker[marker] + MARKER_LABEL_CLUSTER_SPREAD_DEG * offset_index
        label_radius = MARKER_LABEL_BASE_RADIUS + MARKER_LABEL_RADIUS_STEP * abs(offset_index)
        layout[marker] = (np.deg2rad(label_angle_deg), label_radius)


def _has_clustered_marker_angles(markers: list[str], angles: dict[str, float]) -> bool:
    if len(markers) < 2:
        return False
    angle_deg_by_marker = _unwrapped_marker_degrees(markers, angles)
    sorted_angles = sorted(angle_deg_by_marker.values())
    return any(
        right - left <= MARKER_LABEL_CLUSTER_THRESHOLD_DEG
        for left, right in zip(sorted_angles, sorted_angles[1:], strict=False)
    )


def _unwrapped_marker_degrees(markers: list[str], angles: dict[str, float]) -> dict[str, float]:
    raw = {marker: float(np.rad2deg(angles[marker]) % 360.0) for marker in markers}
    if max(raw.values()) - min(raw.values()) > 180.0:
        return {marker: angle - 360.0 if angle > 180.0 else angle for marker, angle in raw.items()}
    return raw


def _draw_spacing_arc(
    ax,
    angles: dict[str, float],
    start_marker: str,
    end_marker: str,
    label: str,
    *,
    radius: float,
    compact: bool,
    config: PlotConfig,
    label_angle_offset_deg: float = 0.0,
    label_horizontal_alignment: str = "center",
) -> None:
    if start_marker not in angles or end_marker not in angles:
        return
    start_deg = np.rad2deg(angles[start_marker])
    end_deg = np.rad2deg(angles[end_marker])
    delta = _wrap180(end_deg - start_deg)
    theta_deg = np.linspace(start_deg, start_deg + delta, 64)
    theta = np.deg2rad(theta_deg)
    pair_color = _blend_marker_colors(start_marker, end_marker)
    ax.plot(theta, np.full_like(theta, radius), color=pair_color, linewidth=config.line_width)
    mid = np.deg2rad(start_deg + delta / 2.0 + label_angle_offset_deg)
    text = ax.text(
        mid,
        radius + 0.06,
        f"{label} = {delta:+.1f}°",
        color=pair_color,
        fontsize=config.compact_annotation_size if compact else config.annotation_size,
        fontweight="bold",
        ha=label_horizontal_alignment,
        va="center",
        rotation=0,
        rotation_mode="anchor",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
    )
    text.set_clip_on(False)


def _blend_marker_colors(start_marker: str, end_marker: str) -> str:
    start_color = np.array(to_rgb(MARKER_COLORS.get(start_marker, "#444444")))
    end_color = np.array(to_rgb(MARKER_COLORS.get(end_marker, "#444444")))
    return to_hex((start_color + end_color) / 2.0)


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0
