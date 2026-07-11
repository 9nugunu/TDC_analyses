"""Polar phase views for marker-sampled S-parameter phases."""

from __future__ import annotations

from collections import OrderedDict
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex, to_rgb
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import (
    IDEAL_PHASE_GUIDE_ANGLES_DEG,
    REFERENCE_GUIDE_ALPHA,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LABEL_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
    PlotConfig,
    apply_plot_style,
    save_figure,
)
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.progress import progress_iter
from deflector_tuning.visualization.simulation_grouping import (
    format_grid_value as _shared_format_grid_value,
    grid_point_depth_group_columns as _shared_grid_point_depth_group_columns,
    varying_sim_sweep_columns as _shared_varying_sim_sweep_columns,
)

MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
REQUIRED_COLUMNS: tuple[str, ...] = ("marker_name", "s_phase_deg")
MARKER_LABEL_CLUSTER_THRESHOLD_DEG = 22.0
MARKER_LABEL_CLUSTER_SPREAD_DEG = 10.0
MARKER_LABEL_BASE_RADIUS = 1.08
MARKER_LABEL_RADIUS_STEP = 0.04
KYHL_PHASE_PAIR_OVERLAY_STEPS: tuple[tuple[str, float, float], ...] = (
    ("cell", 0.5, 1.5),
    ("iris", 1.0, 2.0),
)
KYHL_PHASE_RADIAL_RADIUS = 1.0
KYHL_PHASE_RADIAL_START_ALPHA = 0.50
KYHL_PHASE_RADIAL_END_ALPHA = 0.50


def plot_marker_phase_polar_views(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_prefix: str = "Polar phase",
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write one unit-circle polar phase view per sweep position plus an overview."""

    if marker_points.empty:
        raise ValueError("marker_points is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    require_finite_plot_columns(marker_points, columns=("s_phase_deg",), context="polar marker_points")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: OrderedDict[str, Path] = OrderedDict()
    grouping_mode = _grouping_mode(marker_points)
    groups = list(_iter_position_groups(marker_points))
    if len(groups) > 1:
        for position_label, position_table in progress_iter(
            groups,
            desc="Rendering polar position figures",
            total=len(groups),
        ):
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_position(
                ax,
                position_table,
                _position_plot_title(position_label, position_table, grouping_mode=grouping_mode, title_prefix=title_prefix),
                config=config,
                guide_angles_deg=IDEAL_PHASE_GUIDE_ANGLES_DEG,
            )
            output_stem = _position_output_stem(position_table, position_label, grouping_mode=grouping_mode)
            output_path = folder / f"{output_stem}.png"
            save_figure(fig, output_path, config)
            plt.close(fig)
            paths[position_label] = output_path

    if grouping_mode not in {"grid_point", "sim_sweep"}:
        overview_path = folder / "all_positions.png"
        _save_overview(groups, overview_path, grouping_mode=grouping_mode, title_prefix=title_prefix, config=config)
        paths["overview"] = overview_path
    if grouping_mode == "tune_position":
        family_groups = list(_iter_family_overlay_groups(marker_points))
        for family, family_table in progress_iter(
            family_groups,
            desc="Rendering polar overlays",
            total=len(family_groups),
        ):
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_family_overlay(ax, family_table, f"{family.title()} overlay: {title_prefix}", config=config)
            output_path = folder / f"{family}_overlay.png"
            save_figure(fig, output_path, config)
            plt.close(fig)
            paths[f"{family}_overlay"] = output_path
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_family_overlay(
                ax,
                family_table,
                f"{family.title()} {MARKER_LABELS['f_2pi3']} overlay: {title_prefix}",
                config=config,
                markers=("f_2pi3",),
                guide_angles_deg=IDEAL_PHASE_GUIDE_ANGLES_DEG,
            )
            output_path = folder / f"{family}_f_2pi3_overlay.png"
            save_figure(fig, output_path, config)
            plt.close(fig)
            paths[f"{family}_f_2pi3_overlay"] = output_path
        kyhl_pair_table = _kyhl_phase_pair_overlay_table(marker_points)
        if not kyhl_pair_table.empty:
            for _, pair_table in kyhl_pair_table.groupby("_pair_order", sort=True):
                pair_label = str(pair_table["_pair_label"].iloc[0])
                family = str(pair_table["_pair_family"].iloc[0])
                start = float(pair_table["_pair_start"].iloc[0])
                end = float(pair_table["_pair_end"].iloc[0])
                fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
                _draw_kyhl_phase_pair_overlay(
                    ax,
                    pair_table,
                    _kyhl_phase_pair_overlay_title(family, start, end),
                    pair_label,
                    config=config,
                    guide_angles_deg=IDEAL_PHASE_GUIDE_ANGLES_DEG,
                )
                output_path = folder / f"kyhl_phase_{family}_overlay.png"
                save_figure(fig, output_path, config)
                plt.close(fig)
                paths[f"kyhl_phase_{family}_overlay"] = output_path
    return paths


def _iter_position_groups(marker_points: pd.DataFrame):
    table = marker_points.copy()
    grouping_mode = _grouping_mode(table)
    if grouping_mode == "tune_position":
        table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
        group_columns = _position_group_columns(table, base_columns=("tune_position",))
        table = table.sort_values(["_position_sort", *group_columns, "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            yield _format_group_label(group_key, group_columns), group
        return
    if grouping_mode == "grid_point":
        table["_r_sort"] = pd.to_numeric(table["sim_r_c"], errors="coerce")
        table["_w_sort"] = pd.to_numeric(table["sim_w_c"], errors="coerce")
        if "tune_position" in table and table["tune_position"].dropna().nunique() > 0:
            table["position_family"] = table["tune_position"].map(lambda value: "unknown" if pd.isna(value) else _position_family(value))
            table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
            group_columns = ["position_family", *_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
            table = table.sort_values(
                [*group_columns, "_r_sort", "_w_sort", "_position_sort", "source_file", "marker_name"],
                kind="mergesort",
            )
            for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
                values = group_key if isinstance(group_key, tuple) else (group_key,)
                depth_label = _num_depth_label(group) if "sim_NumDepth" in group_columns else None
                yield _format_grid_family_label(str(values[0]), values[-2], values[-1], depth_label=depth_label), group
            return
        group_columns = [*_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
        table = table.sort_values([*group_columns, "_r_sort", "_w_sort", "source_file", "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            depth_label = _num_depth_label(group) if "sim_NumDepth" in group_columns else None
            yield _format_grid_label(values[-2], values[-1], depth_label=depth_label), group
        return
    if grouping_mode == "sim_sweep":
        sweep_columns = _varying_sim_sweep_columns(table)
        group_columns = _sim_sweep_group_columns(table, sweep_columns)
        sort_columns = [f"_{column}_sort" for column in group_columns]
        for column, sort_column in zip(group_columns, sort_columns, strict=True):
            table[sort_column] = pd.to_numeric(table[column], errors="coerce")
        table = table.sort_values([*sort_columns, *group_columns, "source_file", "marker_name"], kind="mergesort")
        for values, group in table.groupby(group_columns, sort=False, dropna=False):
            if len(group_columns) == 1 and not isinstance(values, tuple):
                values = (values,)
            yield _format_sim_sweep_label(group_columns, values), group
        return
    if grouping_mode == "source_file":
        group_columns = _position_group_columns(table, base_columns=("source_file",))
        table = table.sort_values([*group_columns, "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            yield _format_group_label(group_key, group_columns), group
        return
    yield "all", table.sort_values(["source_file", "marker_name"], kind="mergesort")


def _grouping_mode(marker_points: pd.DataFrame) -> str:
    if _has_multiple_grid_points(marker_points):
        return "grid_point"
    if _has_multiple_sim_sweep_points(marker_points):
        return "sim_sweep"
    if _has_multiple_tune_positions(marker_points):
        return "tune_position"
    if _has_multiple_source_files(marker_points):
        return "source_file"
    return "all"


def _is_grid_scan(marker_points: pd.DataFrame) -> bool:
    if "scan_type" not in marker_points:
        return False
    scan_types = set(marker_points["scan_type"].dropna().astype(str).str.lower())
    return "grid_2d" in scan_types


def _has_multiple_tune_positions(marker_points: pd.DataFrame) -> bool:
    if "tune_position" not in marker_points:
        return False
    return marker_points["tune_position"].dropna().nunique() > 1


def _has_multiple_grid_points(marker_points: pd.DataFrame) -> bool:
    if "sim_r_c" not in marker_points or "sim_w_c" not in marker_points:
        return False
    grid_points = marker_points[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
    return len(grid_points) > 1


def _has_multiple_sim_sweep_points(marker_points: pd.DataFrame) -> bool:
    sweep_columns = _varying_sim_sweep_columns(marker_points)
    if not sweep_columns:
        return False
    sweep_points = marker_points[sweep_columns].dropna(how="all").drop_duplicates()
    return len(sweep_points) > 1


def _has_multiple_source_files(marker_points: pd.DataFrame) -> bool:
    if "source_file" not in marker_points:
        return False
    return marker_points["source_file"].dropna().nunique() > 1


def _position_group_columns(table: pd.DataFrame, *, base_columns: tuple[str, ...]) -> list[str]:
    group_columns = list(base_columns)
    if not _has_duplicate_markers(table, group_columns):
        return group_columns
    for column in ("source_file", "s_name", "port_side"):
        if column in table and table[column].dropna().nunique() > 1 and column not in group_columns:
            group_columns.append(column)
        if not _has_duplicate_markers(table, group_columns):
            break
    return group_columns


def _has_duplicate_markers(table: pd.DataFrame, group_columns: list[str]) -> bool:
    required_columns = [*group_columns, "marker_name"]
    if any(column not in table for column in required_columns):
        return False
    return bool(table.duplicated(required_columns, keep=False).any())


def _save_overview(
    groups: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    *,
    grouping_mode: str,
    title_prefix: str,
    config: PlotConfig,
) -> None:
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
        _draw_position(
            ax,
            position_table,
            _position_plot_title(position_label, position_table, grouping_mode=grouping_mode, title_prefix=title_prefix),
            compact=False,
            config=config,
            guide_angles_deg=IDEAL_PHASE_GUIDE_ANGLES_DEG,
        )
    for ax in flat_axes[len(groups) :]:
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.32, hspace=0.44)
    save_figure(fig, output_path, config)
    plt.close(fig)


def _iter_family_overlay_groups(marker_points: pd.DataFrame):
    table = marker_points.copy()
    table = table.dropna(subset=["tune_position"]).copy()
    if table.empty:
        return
    table["position_family"] = table["tune_position"].map(_position_family)
    table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table.sort_values(["position_family", "_position_sort", "marker_name"], kind="mergesort")
    for family, group in table.groupby("position_family", sort=False, dropna=False):
        if family not in {"cell", "iris"} or group["tune_position"].dropna().nunique() < 1:
            continue
        yield str(family), group.copy()


def _kyhl_phase_pair_overlay_table(marker_points: pd.DataFrame) -> pd.DataFrame:
    table = marker_points.copy()
    table = table.dropna(subset=["tune_position"]).copy()
    if table.empty:
        return pd.DataFrame()
    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table[table["_tune_sort"].notna() & table["marker_name"].isin(MARKER_ORDER)].copy()
    if table.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    marker_order = {marker: index for index, marker in enumerate(MARKER_ORDER)}
    for pair_order, (family, start, end) in enumerate(KYHL_PHASE_PAIR_OVERLAY_STEPS):
        is_start = np.isclose(table["_tune_sort"], start, atol=1e-9, equal_nan=False)
        is_end = np.isclose(table["_tune_sort"], end, atol=1e-9, equal_nan=False)
        pair_rows = table[is_start | is_end].copy()
        if pair_rows.empty or not bool(is_start.any() and is_end.any()):
            continue
        complete_markers = [
            marker
            for marker, group in pair_rows.groupby("marker_name", sort=False)
            if {"start", "end"}.issubset(
                set(np.where(np.isclose(group["_tune_sort"], start, atol=1e-9), "start", "end"))
            )
        ]
        if not complete_markers:
            continue
        pair_rows = pair_rows[pair_rows["marker_name"].isin(complete_markers)].copy()
        pair_rows["_pair_order"] = pair_order
        pair_rows["_pair_family"] = family
        pair_rows["_pair_start"] = start
        pair_rows["_pair_end"] = end
        pair_rows["_pair_label"] = f"{family} {_format_position(start)}->{_format_position(end)}"
        pair_rows["_pair_endpoint"] = np.where(np.isclose(pair_rows["_tune_sort"], start, atol=1e-9), "start", "end")
        pair_rows["_marker_order"] = pair_rows["marker_name"].map(marker_order)
        frames.append(pair_rows)

    if not frames:
        return pd.DataFrame()
    overlay = pd.concat(frames, ignore_index=True)
    return overlay.sort_values(["_pair_order", "_marker_order", "_pair_endpoint"], kind="mergesort")


def _kyhl_phase_pair_overlay_title(family: str, start: float, end: float) -> str:
    return f"KYHL {family} phase overlay: {_format_position(start)}->{_format_position(end)}"


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


def _format_position(position: object) -> str:
    try:
        value = _snap_tune_position(float(position))
    except (TypeError, ValueError):
        return str(position)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _format_group_label(group_key: object, group_columns: list[str]) -> str:
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    if tuple(group_columns) == ("sim_r_c", "sim_w_c"):
        return _format_grid_label(values[0], values[1])
    parts: list[str] = []
    for column, value in zip(group_columns, values, strict=True):
        if column == "tune_position":
            parts.append(_format_position(value))
        elif column == "sim_r_c":
            parts.append(f"r_c={_format_grid_value(value)}")
        elif column == "sim_w_c":
            parts.append(f"w_c={_format_grid_value(value)}")
        elif column == "source_file":
            parts.append(Path(str(value)).stem)
        elif column == "s_name":
            parts.append(str(value))
        elif column == "port_side" and not pd.isna(value):
            parts.append(f"port={value}")
        elif not pd.isna(value):
            parts.append(str(value))
    return " | ".join(parts)


def _position_output_stem(position_table: pd.DataFrame, position_label: str, *, grouping_mode: str) -> str:
    if grouping_mode == "tune_position" and " | " not in position_label:
        return _tune_position_filename_label(position_table)
    if grouping_mode == "grid_point":
        return _grid_point_filename_label(position_table, position_label)
    if grouping_mode == "sim_sweep":
        return f"{_simulation_family_label(position_table)}_{_safe_label(position_label)}"
    return f"position_{_safe_label(position_label)}"


def _position_plot_title(position_label: str, position_table: pd.DataFrame, *, grouping_mode: str, title_prefix: str) -> str:
    if grouping_mode == "grid_point":
        family = _grid_point_family(position_table)
        if family is not None:
            grid_label = _format_grid_label(
                position_table["sim_r_c"].iloc[0],
                position_table["sim_w_c"].iloc[0],
                depth_label=_num_depth_label(position_table),
            )
            return f"{family.title()} polar: {grid_label}"
        return f"Grid polar: {position_label}"
    if grouping_mode == "sim_sweep":
        return f"{_simulation_family_label(position_table).title()} polar: {position_label}"
    return f"{position_label}: {title_prefix}"


def _tune_position_filename_label(position_table: pd.DataFrame) -> str:
    tune_positions = position_table["tune_position"].dropna().unique()
    if len(tune_positions) != 1:
        return f"position_{_safe_label(_format_position(tune_positions[0] if len(tune_positions) else 'unknown'))}"
    tune_position = tune_positions[0]
    return f"{_position_family(tune_position)}_{_safe_label(_format_position(tune_position))}"


def _position_family(tune_position: object) -> str:
    value = _snap_tune_position(float(tune_position))
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{_safe_label(f'{fractional:g}')}"


def _snap_tune_position(value: float) -> float:
    doubled = round(value * 2.0)
    snapped = doubled / 2.0
    if abs(value - snapped) < 1e-6:
        return snapped
    return value


def _format_grid_label(sim_r_c: object, sim_w_c: object, *, depth_label: str | None = None) -> str:
    grid_label = f"r_c={_format_grid_value(sim_r_c)}, w_c={_format_grid_value(sim_w_c)}"
    if depth_label is None:
        return grid_label
    return f"depth={depth_label.removeprefix('depth_')} {grid_label}"


def _format_grid_family_label(family: str, sim_r_c: object, sim_w_c: object, *, depth_label: str | None = None) -> str:
    return f"{family} {_format_grid_label(sim_r_c, sim_w_c, depth_label=depth_label)}"


def _grid_point_filename_label(position_table: pd.DataFrame, position_label: str) -> str:
    family = _grid_point_family(position_table)
    depth_suffix = _grid_point_depth_filename_suffix(position_table)
    if family is not None:
        return (
            f"{family}{depth_suffix}_"
            f"{_safe_label(_format_grid_label(position_table['sim_r_c'].iloc[0], position_table['sim_w_c'].iloc[0]))}"
        )
    return f"grid_{_safe_label(position_label)}"


def _grid_point_family(position_table: pd.DataFrame) -> str | None:
    if "position_family" not in position_table:
        return None
    families = [str(family) for family in position_table["position_family"].dropna().unique()]
    if len(families) == 1 and families[0] != "unknown":
        return families[0]
    return None


def _grid_point_depth_group_columns(table: pd.DataFrame) -> list[str]:
    return _shared_grid_point_depth_group_columns(table)


def _grid_point_depth_filename_suffix(position_table: pd.DataFrame) -> str:
    depth_label = _num_depth_label(position_table)
    if depth_label is None:
        return ""
    return f"_{depth_label}"


def _num_depth_label(table: pd.DataFrame) -> str | None:
    if "sim_NumDepth" not in table:
        return None
    values = pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().unique()
    if len(values) != 1:
        return None
    return f"depth_{_format_num_depth(values[0])}"


def _simulation_family_label(position_table: pd.DataFrame) -> str:
    text_parts: list[str] = []
    for column in ("dataset_id", "source_file"):
        if column in position_table:
            text_parts.extend(str(value).lower() for value in position_table[column].dropna().unique())
    joined = " ".join(text_parts)
    if "iris" in joined:
        return "iris"
    if "cell" in joined:
        return "cell"
    if "tune_position" in position_table:
        tune_positions = position_table["tune_position"].dropna().unique()
        if len(tune_positions) == 1:
            try:
                family = _position_family(tune_positions[0])
            except (TypeError, ValueError):
                family = ""
            if family in {"iris", "cell"}:
                return family
    return "sim"


def _format_sim_sweep_label(sweep_columns: list[str], values: tuple[object, ...]) -> str:
    parts = []
    for column, value in zip(sweep_columns, values, strict=True):
        if column == "sim_NumDepth":
            parts.append(f"depth={_format_num_depth(value)}")
        else:
            parts.append(f"{_sim_sweep_column_label(column)}={_format_grid_value(value)}")
    return "_".join(parts)


def _sim_sweep_group_columns(table: pd.DataFrame, sweep_columns: list[str]) -> list[str]:
    group_columns = []
    if "sim_NumDepth" in table and table["sim_NumDepth"].dropna().nunique() > 1:
        group_columns.append("sim_NumDepth")
    group_columns.extend(column for column in sweep_columns if column not in group_columns)
    if "sim_NumDepth" not in group_columns and "tune_position" in table and table["tune_position"].dropna().nunique() > 1:
        group_columns.append("tune_position")
    return group_columns


def _varying_sim_sweep_columns(marker_points: pd.DataFrame) -> list[str]:
    return _shared_varying_sim_sweep_columns(marker_points)


def _sim_sweep_column_label(column: str) -> str:
    if column == "tune_position":
        return "tune_position"
    label = column.removeprefix("sim_")
    if label.startswith("Depth") and len(label) > len("Depth"):
        label = label[len("Depth") :]
    return _camel_to_snake(label)


def _format_num_depth(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}p0"
    return _format_grid_value(value)


def _camel_to_snake(label: str) -> str:
    converted = []
    previous_is_lower_or_digit = False
    for character in label:
        if character in {" ", "-", "."}:
            converted.append("_")
            previous_is_lower_or_digit = False
            continue
        if character.isupper() and previous_is_lower_or_digit:
            converted.append("_")
        converted.append(character.lower())
        previous_is_lower_or_digit = character.islower() or character.isdigit()
    return "".join(converted)


def _format_grid_value(value: object) -> str:
    return _shared_format_grid_value(value)


def _safe_label(label: str) -> str:
    sanitized = (
        label.replace("=", "_")
        .replace(",", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace(" ", "_")
        .replace("|", "_")
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized
