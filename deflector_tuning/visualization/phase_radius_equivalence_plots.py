"""Plot raw before/after phase changes on a simulated radius-equivalence axis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)


def plot_phase_radius_equivalence(
    projections: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_path: str | Path,
    *,
    baseline_r_c_mm: float,
    fixed_w_c_mm: float | None = None,
    config: PlotConfig | None = None,
) -> Path:
    """Plot simulated phase-change curves and raw phase-change projections."""

    if not projections:
        raise ValueError("projections is empty")
    config = config or PlotConfig()
    apply_plot_style(config)
    figure, axes = plt.subplots(1, len(projections), figsize=(6.8 * len(projections), 5.2), sharey=True)
    if len(projections) == 1:
        axes = [axes]
    for axis, (label, (estimate, curves)) in zip(axes, projections.items(), strict=True):
        for marker_name, marker_curve in curves.groupby("marker_name", sort=False):
            marker_key = str(marker_name)
            color = MARKER_COLORS.get(marker_key, "#444444")
            marker_label = MARKER_LABELS.get(marker_key, marker_key)
            axis.plot(
                marker_curve["delta_r_c_um"],
                marker_curve["sim_phase_delta_deg"],
                color=color,
                linewidth=config.line_width,
                label=f"simulation {marker_label}",
            )
            observation = estimate[estimate["marker_name"] == marker_name].iloc[0]
            axis.scatter(
                observation["marker_equivalent_delta_r_c_um"],
                observation["phase_delta_deg"],
                s=72,
                marker="o",
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
                label=f"raw {marker_label}",
            )
        shared_delta_um = float(estimate["equivalent_delta_r_c_um"].iloc[0])
        axis.axvline(
            shared_delta_um,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            label=f"joint fit = {shared_delta_um:.1f} μm",
        )
        apply_axis_text_style(
            axis,
            xlabel=f"Design-anchored equivalent Δr_c from {baseline_r_c_mm:g} mm [μm]",
            ylabel="Δ S-parameter phase (after - before) [deg]",
            title=label,
            config=config,
            compact=True,
        )
        axis.grid(True, color="0.86", linewidth=0.8)
        legend = axis.legend(frameon=False, fontsize=9.5, loc="best")
        apply_legend_text_style(legend, config)
        match_legend_text_colors_to_handles(legend)
    simulation_label = (
        rf"simulation $r_c$ line ($w_c={fixed_w_c_mm:g}$ mm)"
        if fixed_w_c_mm is not None
        else r"one-dimensional simulation $r_c$ sweep"
    )
    figure.suptitle(
        f"Measured phase change projected onto {simulation_label}",
        fontsize=config.compact_title_size,
        fontweight=config.title_weight,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def plot_raw_phase_position_on_simulation_line(
    second_iris_position: pd.DataFrame,
    second_iris_curves: pd.DataFrame,
    first_iris_observation: pd.DataFrame,
    output_path: str | Path,
    *,
    baseline_r_c_mm: float,
    target_r_c_mm: float,
    fixed_w_c_mm: float,
    config: PlotConfig | None = None,
) -> Path:
    """Show the raw tuning state on a port-2 simulated ``r_c`` phase line.

    The first-iris panel retains its measured phase change as context.  It is
    intentionally not projected onto the port-2 simulation trace.
    """

    _require_columns(
        second_iris_position,
        ("marker_name", "before_phase_deg", "after_phase_deg", "marker_equivalent_r_c_mm", "shared_equivalent_r_c_mm", "remaining_to_target_um"),
        "second_iris_position",
    )
    _require_columns(second_iris_curves, ("marker_name", "sim_r_c", "raw_anchored_phase_deg"), "second_iris_curves")
    _require_columns(first_iris_observation, ("marker_name", "before_phase_deg", "after_phase_deg", "phase_delta_deg"), "first_iris_observation")
    config = config or PlotConfig()
    apply_plot_style(config)
    figure, (first_axis, second_axis) = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [0.8, 1.45]})

    marker_order = second_iris_position["marker_name"].tolist()
    x_before, x_after = 0.0, 1.0
    for index, marker_name in enumerate(marker_order):
        first_row = first_iris_observation[first_iris_observation["marker_name"] == marker_name].iloc[0]
        color = MARKER_COLORS.get(str(marker_name), "#444444")
        label = MARKER_LABELS.get(str(marker_name), str(marker_name))
        first_axis.plot(
            [x_before, x_after],
            [first_row["before_phase_deg"], first_row["after_phase_deg"]],
            color=color,
            linewidth=config.line_width,
            marker="o",
            markersize=6.5,
            label=f"{label}: Δφ = {first_row['phase_delta_deg']:+.2f}°",
        )
    first_axis.set_xlim(-0.22, 1.22)
    first_axis.set_xticks([x_before, x_after], ["before", "Torque 13.5"])
    apply_axis_text_style(
        first_axis,
        xlabel="First iris / port 1 raw measurement",
        ylabel="Raw S11 phase [deg]",
        title="First iris: measured Δφ only",
        config=config,
        compact=True,
    )
    first_axis.grid(True, color="0.86", linewidth=0.8)
    legend = first_axis.legend(frameon=False, fontsize=9.0, loc="best")
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)

    for marker_name, marker_curve in second_iris_curves.groupby("marker_name", sort=False):
        color = MARKER_COLORS.get(str(marker_name), "#444444")
        label = MARKER_LABELS.get(str(marker_name), str(marker_name))
        second_axis.plot(
            marker_curve["sim_r_c"],
            marker_curve["raw_anchored_phase_deg"],
            color=color,
            linewidth=config.line_width,
            label=label,
        )
        row = second_iris_position[second_iris_position["marker_name"] == marker_name].iloc[0]
        second_axis.scatter(
            baseline_r_c_mm,
            row["before_phase_deg"],
            s=60,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=4,
            label="raw before" if marker_name == marker_order[0] else None,
        )
        second_axis.scatter(
            row["marker_equivalent_r_c_mm"],
            row["after_phase_deg"],
            s=62,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="raw Torque 13.5" if marker_name == marker_order[0] else None,
        )

    current_r_c_mm = float(second_iris_position["shared_equivalent_r_c_mm"].iloc[0])
    offset_from_target_um = (current_r_c_mm - target_r_c_mm) * 1e3
    second_axis.axvline(baseline_r_c_mm, color="0.45", linestyle="--", linewidth=1.0, label=f"design / raw anchor = {baseline_r_c_mm:.2f} mm")
    second_axis.axvline(current_r_c_mm, color="0.15", linestyle="--", linewidth=1.35, label=f"current joint fit = {current_r_c_mm:.3f} mm")
    if abs(target_r_c_mm - baseline_r_c_mm) > 1e-9:
        second_axis.axvline(target_r_c_mm, color="#c51b7d", linestyle="--", linewidth=1.35, label=f"reference target = {target_r_c_mm:.2f} mm")
    second_axis.annotate(
        f"current: {current_r_c_mm:.3f} mm\nfrom design: {offset_from_target_um:+.0f} μm",
        xy=(current_r_c_mm, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(7, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9.6,
        color="0.1",
    )
    individual_positions = pd.to_numeric(second_iris_position["marker_equivalent_r_c_mm"], errors="coerce")
    lower_bound = min(baseline_r_c_mm, target_r_c_mm, current_r_c_mm, float(individual_positions.min()))
    upper_bound = max(baseline_r_c_mm, target_r_c_mm, current_r_c_mm, float(individual_positions.max()))
    margin = max(0.035, 0.30 * (upper_bound - lower_bound))
    second_axis.set_xlim(lower_bound - margin, upper_bound + margin)
    apply_axis_text_style(
        second_axis,
        xlabel=r"Simulation radius $r_c$ [mm]",
        ylabel="S11 phase [deg], raw reference anchored",
        title="Second iris / port 2: raw state on simulation $r_c$ line",
        config=config,
        compact=True,
    )
    second_axis.grid(True, color="0.86", linewidth=0.8)
    legend = second_axis.legend(frameon=False, fontsize=8.6, loc="lower left", ncol=2)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    figure.suptitle(
        rf"Torque 13.5 state located on the simulated $r_c$ phase scan ($w_c={fixed_w_c_mm:g}$ mm)",
        fontsize=config.compact_title_size,
        fontweight=config.title_weight,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def plot_phase_cmp_bars(
    comparison: pd.DataFrame,
    output_path: str | Path,
    *,
    current_label: str = "Current",
    config: PlotConfig | None = None,
) -> Path:
    """Plot measured and simulated phases before/after plus their phase change."""

    required = (
        "marker_name",
        "measured_before_phase_deg",
        "simulated_before_phase_deg",
        "measured_current_phase_deg",
        "simulated_current_phase_deg",
        "measured_phase_delta_deg",
        "simulated_phase_delta_deg",
        "before_r_c_mm",
        "current_r_c_mm",
    )
    _require_columns(comparison, required, "comparison")
    if comparison.empty:
        raise ValueError("comparison is empty")
    config = config or PlotConfig()
    apply_plot_style(config)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 5.1), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]})
    x = np.arange(len(comparison), dtype=float)
    width = 0.36
    labels = [MARKER_LABELS.get(str(name), str(name)) for name in comparison["marker_name"]]
    experiment_color = "#2c7fb8"
    simulation_color = "#f28e2b"
    panels = (
        (
            "measured_before_phase_deg",
            "simulated_before_phase_deg",
            f"Before tuning\nequivalent $r_c={float(comparison['before_r_c_mm'].iloc[0]):.3f}$ mm",
            "S11 phase [deg]",
        ),
        (
            "measured_current_phase_deg",
            "simulated_current_phase_deg",
            f"{current_label}\nequivalent $r_c={float(comparison['current_r_c_mm'].iloc[0]):.3f}$ mm",
            "S11 phase [deg]",
        ),
        (
            "measured_phase_delta_deg",
            "simulated_phase_delta_deg",
            r"Tuning response: after $-$ before",
            "Δ S11 phase [deg]",
        ),
    )
    for panel_index, (axis, (measured_column, simulated_column, title, ylabel)) in enumerate(zip(axes, panels, strict=True)):
        measured_bars = axis.bar(
            x - width / 2,
            comparison[measured_column],
            width,
            color=experiment_color,
            label="experiment",
        )
        simulated_bars = axis.bar(
            x + width / 2,
            comparison[simulated_column],
            width,
            color=simulation_color,
            hatch="//",
            edgecolor="#8a4f08",
            linewidth=0.7,
            label="simulation",
        )
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_xticks(x, labels)
        apply_axis_text_style(
            axis,
            xlabel="frequency marker",
            ylabel=ylabel,
            title=title,
            config=config,
            compact=True,
        )
        axis.grid(True, axis="y", color="0.86", linewidth=0.8)
        axis.bar_label(measured_bars, fmt="%.1f", padding=3, fontsize=8.3, color=experiment_color)
        axis.bar_label(simulated_bars, fmt="%.1f", padding=3, fontsize=8.3, color="#8a4f08")
        if panel_index == 0:
            legend = axis.legend(frameon=False, fontsize=9.0, loc="best")
            apply_legend_text_style(legend, config)
    figure.suptitle(
        "Experiment vs simulation at fitted equivalent radius states",
        fontsize=config.compact_title_size,
        fontweight=config.title_weight,
    )
    figure.text(
        0.5,
        0.015,
        "Absolute phase includes a reference-plane offset; the right panel is the fitted observable.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="0.25",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], table_name: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
