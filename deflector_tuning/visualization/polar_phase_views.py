"""Render the ordered polar phase outputs planned from marker tables.

Grouping and output naming live in ``polar_grouping``/``polar_plot_plans``;
axes drawing lives in ``polar_drawing``/``polar_overlays``. This entry point
owns plot validation, style setup, progress reporting, and figure saving.
"""

from __future__ import annotations

from collections import OrderedDict
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.progress import progress_iter
from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style, save_figure
from deflector_tuning.visualization.polar_drawing import (
    MARKER_LABEL_BASE_RADIUS,
    MARKER_LABEL_CLUSTER_SPREAD_DEG,
    MARKER_LABEL_CLUSTER_THRESHOLD_DEG,
    MARKER_LABEL_RADIUS_STEP,
    _draw_position,
)
from deflector_tuning.visualization.polar_grouping import (
    KYHL_PHASE_PAIR_OVERLAY_STEPS,
    MARKER_ORDER,
    NO_PORT_EXTENSION_SOURCE_PATTERN,
    _position_plot_title,
)
from deflector_tuning.visualization.polar_overlays import (
    KYHL_PHASE_RADIAL_END_ALPHA,
    KYHL_PHASE_RADIAL_RADIUS,
    KYHL_PHASE_RADIAL_START_ALPHA,
    _draw_family_overlay,
    _draw_kyhl_phase_pair_overlay,
)
from deflector_tuning.visualization.polar_plot_plans import (
    NO_PORT_EXTENSION_FOLDER,
    PolarFamilyOverlayPlan,
    PolarKyhlPairPlan,
    PolarOverviewPlan,
    PolarPlotPlan,
    PolarPositionPlan,
    build_polar_plot_plans,
)


REQUIRED_COLUMNS: tuple[str, ...] = ("marker_name", "s_phase_deg")


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
    plans = build_polar_plot_plans(
        marker_points,
        folder,
        title_prefix=title_prefix,
    )
    paths: OrderedDict[str, Path] = OrderedDict()
    for plan in progress_iter(
        plans,
        desc="Rendering polar figures",
        total=len(plans),
    ):
        if isinstance(plan, PolarPositionPlan):
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_position(
                ax,
                plan.position_table,
                plan.title,
                config=config,
                guide_angles_deg=config.ideal_phase_guide_angles_deg,
            )
            save_figure(fig, plan.output_path, config)
            plt.close(fig)
            paths[plan.key] = plan.output_path
        elif isinstance(plan, PolarOverviewPlan):
            _save_overview(
                plan.groups,
                plan.output_path,
                grouping_mode=plan.grouping_mode,
                title_prefix=plan.title_prefix,
                config=config,
            )
            paths[plan.key] = plan.output_path
        elif isinstance(plan, PolarFamilyOverlayPlan):
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_family_overlay(
                ax,
                plan.family_table,
                plan.title,
                config=config,
                markers=plan.markers,
                guide_angles_deg=(
                    config.ideal_phase_guide_angles_deg if plan.markers else ()
                ),
            )
            save_figure(fig, plan.output_path, config)
            plt.close(fig)
            paths[plan.key] = plan.output_path
        else:
            fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
            _draw_kyhl_phase_pair_overlay(
                ax,
                plan.pair_table,
                plan.title,
                plan.pair_label,
                config=config,
                guide_angles_deg=config.ideal_phase_guide_angles_deg,
            )
            save_figure(fig, plan.output_path, config)
            plt.close(fig)
            paths[plan.key] = plan.output_path
    return paths


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
            guide_angles_deg=config.ideal_phase_guide_angles_deg,
        )
    for ax in flat_axes[len(groups) :]:
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.32, hspace=0.44)
    save_figure(fig, output_path, config)
    plt.close(fig)
