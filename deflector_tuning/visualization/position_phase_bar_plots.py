"""Grouped bar views of marker phase at selected tuning positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.marker_styles import MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_plot_style,
    save_figure,
)

MARKER_ORDER = ("f_2pi3", "f_mean", "f_pi2")
IDEAL_PHASE_TARGETS_DEG = {
    1.0: {
        "f_2pi3": 180.0,
        "f_mean": 180.0,
        "f_pi2": 180.0,
    },
    2.0: {
        "f_2pi3": 60.0,
        "f_mean": 0.0,
        "f_pi2": 300.0,
    },
}
POSITION_IDEAL_COLOR = "#d9dde3"
ADVANCE_MEASURED_COLOR = "#3f6f7d"
ADVANCE_MEASURED_EDGE_COLOR = "#27464f"
ADVANCE_IDEAL_COLOR = "#d9dde3"
ADVANCE_IDEAL_EDGE_COLOR = "#6b7075"
ADVANCE_EXCESS_COLOR = "#c44e52"
ADVANCE_DEFICIT_COLOR = "#4c78a8"
ADVANCE_IDEAL_BAR_WIDTH = 0.48
ADVANCE_MEASURED_BAR_WIDTH = 0.48
BAR_TABLE_COLUMNS = (
    "dataset_id",
    "source_file",
    "tune_position",
    "marker_name",
    "freq_target_ghz",
    "freq_ghz",
    "s_phase_deg",
)
TARGET_TABLE_COLUMNS = BAR_TABLE_COLUMNS + (
    "phase_0to360_deg",
    "ideal_phase_deg",
    "phase_error_deg",
    "phase_advance_0to360_deg",
    "ideal_phase_advance_deg",
    "phase_advance_error_deg",
)


def build_position_phase_bar_table(
    marker_points: pd.DataFrame,
    *,
    positions: Sequence[float],
    marker_order: Sequence[str] = MARKER_ORDER,
) -> pd.DataFrame:
    """Select one marker-phase row for each requested mode and position."""

    missing = [column for column in BAR_TABLE_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    requested_positions = tuple(float(position) for position in positions)
    requested_markers = tuple(str(marker) for marker in marker_order)
    if not requested_positions:
        raise ValueError("positions is empty")
    if len(set(requested_positions)) != len(requested_positions):
        raise ValueError(f"positions contains duplicates: {requested_positions}")

    numeric_positions = pd.to_numeric(marker_points["tune_position"], errors="coerce")
    selected_rows: list[pd.Series] = []
    for marker_name in requested_markers:
        marker_mask = marker_points["marker_name"].astype(str).eq(marker_name)
        for position in requested_positions:
            pair = marker_points[
                marker_mask
                & np.isclose(
                    numeric_positions.to_numpy(dtype=float),
                    position,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=False,
                )
            ]
            if len(pair) != 1:
                raise ValueError(
                    "expected exactly one row for marker "
                    f"'{marker_name}' at position {position}; found {len(pair)}"
                )
            selected_rows.append(pair.iloc[0])

    return pd.DataFrame(selected_rows).loc[:, BAR_TABLE_COLUMNS].reset_index(drop=True)


def build_position_phase_target_table(
    marker_points: pd.DataFrame,
    *,
    positions: Sequence[float],
    marker_order: Sequence[str] = MARKER_ORDER,
    ideal_phase_targets_deg: Mapping[float, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Add 0-to-360 phase targets and phase advance to the selected rows."""

    requested_positions = tuple(float(position) for position in positions)
    if len(requested_positions) != 2:
        raise ValueError(
            "phase target comparison requires exactly two positions; "
            f"received {requested_positions}"
        )
    requested_markers = tuple(str(marker) for marker in marker_order)
    targets = ideal_phase_targets_deg or IDEAL_PHASE_TARGETS_DEG
    comparison = build_position_phase_bar_table(
        marker_points,
        positions=requested_positions,
        marker_order=requested_markers,
    )

    comparison["phase_0to360_deg"] = (
        pd.to_numeric(comparison["s_phase_deg"], errors="coerce") % 360.0
    )
    comparison["ideal_phase_deg"] = [
        _ideal_phase_for_position(
            targets,
            position=float(row.tune_position),
            marker_name=str(row.marker_name),
        )
        for row in comparison.itertuples(index=False)
    ]
    comparison["phase_error_deg"] = _wrap180(
        comparison["ideal_phase_deg"] - comparison["phase_0to360_deg"]
    )
    comparison["phase_advance_0to360_deg"] = np.nan
    comparison["ideal_phase_advance_deg"] = np.nan
    comparison["phase_advance_error_deg"] = np.nan

    from_position, to_position = requested_positions
    for marker_name in requested_markers:
        marker_rows = comparison[comparison["marker_name"].eq(marker_name)].set_index(
            "tune_position"
        )
        measured_advance = (
            float(marker_rows.at[to_position, "phase_0to360_deg"])
            - float(marker_rows.at[from_position, "phase_0to360_deg"])
        ) % 360.0
        ideal_advance = (
            float(marker_rows.at[to_position, "ideal_phase_deg"])
            - float(marker_rows.at[from_position, "ideal_phase_deg"])
        ) % 360.0
        to_mask = comparison["marker_name"].eq(marker_name) & np.isclose(
            comparison["tune_position"].to_numpy(dtype=float),
            to_position,
            rtol=0.0,
            atol=1e-12,
        )
        comparison.loc[to_mask, "phase_advance_0to360_deg"] = measured_advance
        comparison.loc[to_mask, "ideal_phase_advance_deg"] = ideal_advance
        comparison.loc[to_mask, "phase_advance_error_deg"] = float(
            _wrap180(ideal_advance - measured_advance)
        )

    return comparison.loc[:, TARGET_TABLE_COLUMNS]


def classify_phase_advance_difference(
    measured_phase_deg: float,
    ideal_phase_deg: float,
) -> tuple[str, float]:
    """Return whether measured phase advance exceeds or misses the ideal."""

    difference = float(measured_phase_deg) - float(ideal_phase_deg)
    if np.isclose(difference, 0.0, rtol=0.0, atol=1e-12):
        return "matched", 0.0
    return ("excess" if difference > 0.0 else "deficit"), abs(difference)


def plot_position_phase_bars(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    positions: Sequence[float],
    marker_order: Sequence[str] = MARKER_ORDER,
    ideal_phase_targets_deg: Mapping[float, Mapping[str, float]] | None = None,
    config: PlotConfig | None = None,
) -> Path:
    """Plot measured and ideal phase at two positions on a 0-to-360 axis."""

    config = config or PlotConfig()
    apply_plot_style(config)
    comparison = build_position_phase_target_table(
        marker_points,
        positions=positions,
        marker_order=marker_order,
        ideal_phase_targets_deg=ideal_phase_targets_deg,
    )
    require_finite_plot_columns(
        comparison,
        columns=("phase_0to360_deg", "ideal_phase_deg"),
        context="position phase bars",
    )

    requested_positions = tuple(float(position) for position in positions)
    requested_markers = tuple(str(marker) for marker in marker_order)
    group_centers = np.arange(len(requested_markers), dtype=float)
    bar_width = min(0.30, 0.68 / len(requested_positions))
    offsets = (
        np.arange(len(requested_positions), dtype=float)
        - (len(requested_positions) - 1) / 2.0
    ) * bar_width
    colors = ("#2c7fb8", "#f28e2b")
    edge_colors = ("#174f73", "#8a4f08")
    hatches = ("", "//")

    figure, phase_axis = plt.subplots(figsize=(9.2, 6.4))
    individual_ticks: list[float] = []
    individual_labels: list[str] = []
    for position_index, position in enumerate(requested_positions):
        position_rows = comparison[
            np.isclose(
                comparison["tune_position"].to_numpy(dtype=float),
                position,
                rtol=0.0,
                atol=1e-12,
            )
        ].set_index("marker_name")
        values = position_rows.loc[
            list(requested_markers), "phase_0to360_deg"
        ].to_numpy(dtype=float)
        ideal_values = position_rows.loc[
            list(requested_markers), "ideal_phase_deg"
        ].to_numpy(dtype=float)
        phase_errors = position_rows.loc[
            list(requested_markers), "phase_error_deg"
        ].to_numpy(dtype=float)
        x_values = group_centers + offsets[position_index]
        render_width = bar_width * 0.90
        phase_axis.bar(
            x_values,
            ideal_values,
            width=render_width,
            color=POSITION_IDEAL_COLOR,
            edgecolor=ADVANCE_IDEAL_EDGE_COLOR,
            linewidth=0.9,
            zorder=2,
        )
        bars = phase_axis.bar(
            x_values,
            values,
            width=render_width,
            color=colors[position_index % len(colors)],
            edgecolor=edge_colors[position_index % len(edge_colors)],
            linewidth=0.9,
            hatch=hatches[position_index % len(hatches)],
            zorder=3,
        )
        for marker_name, bar, value, ideal, phase_error in zip(
            requested_markers,
            bars,
            values,
            ideal_values,
            phase_errors,
            strict=True,
        ):
            difference_kind, difference_magnitude = classify_phase_advance_difference(
                value,
                ideal,
            )
            if difference_kind != "matched":
                difference_color = (
                    ADVANCE_EXCESS_COLOR
                    if difference_kind == "excess"
                    else ADVANCE_DEFICIT_COLOR
                )
                difference_hatch = (
                    "///" if difference_kind == "excess" else "\\\\\\"
                )
                phase_axis.bar(
                    bar.get_x() + bar.get_width() / 2.0,
                    difference_magnitude,
                    bottom=min(value, ideal),
                    width=render_width,
                    color=difference_color,
                    edgecolor=difference_color,
                    linewidth=0.7,
                    hatch=difference_hatch,
                    alpha=0.55,
                    zorder=4,
                )
                error_label = f"{phase_error:+.1f}°"
                if difference_magnitude >= 18.0:
                    phase_axis.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(value, ideal) + difference_magnitude / 2.0,
                        error_label,
                        ha="center",
                        va="center",
                        fontsize=max(config.annotation_size, config.compact_label_size),
                        fontweight=config.label_weight,
                        color="white",
                        zorder=6,
                    )
                else:
                    place_left = marker_name == "f_pi2" and np.isclose(
                        position,
                        1.0,
                    )
                    phase_axis.annotate(
                        error_label,
                        xy=(
                            bar.get_x() if place_left else bar.get_x() + bar.get_width(),
                            min(value, ideal) + difference_magnitude / 2.0,
                        ),
                        xytext=(-7, 0) if place_left else (7, 0),
                        textcoords="offset points",
                        ha="right" if place_left else "left",
                        va="center",
                        fontsize=max(config.annotation_size, config.compact_label_size),
                        fontweight=config.label_weight,
                        color=difference_color,
                        clip_on=False,
                        zorder=6,
                    )
            label_y = (
                max(value, ideal) + 7.0
                if abs(value - ideal) < 18.0
                else value + 7.0
            )
            phase_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{value:.1f}°",
                ha="center",
                va="bottom",
                fontsize=config.annotation_size,
                fontweight=config.label_weight,
                color=edge_colors[position_index % len(edge_colors)],
                clip_on=False,
                zorder=6,
            )
        phase_axis.bar(
            x_values,
            ideal_values,
            width=render_width,
            facecolor="none",
            edgecolor=ADVANCE_IDEAL_EDGE_COLOR,
            linewidth=1.1,
            zorder=5,
        )
        individual_ticks.extend(x_values.tolist())
        individual_labels.extend(
            [_format_position(position)] * len(requested_markers)
        )

    tick_order = np.argsort(individual_ticks)
    phase_axis.set_xticks(
        np.asarray(individual_ticks)[tick_order],
        np.asarray(individual_labels, dtype=object)[tick_order],
    )
    for center, marker_name in zip(group_centers, requested_markers, strict=True):
        phase_axis.text(
            center,
            -0.095,
            MARKER_LABELS.get(marker_name, marker_name),
            transform=phase_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=config.label_size,
            fontweight=config.label_weight,
        )

    position_text = " and ".join(
        _format_position(position) for position in requested_positions
    )
    phase_axis.set_ylim(-10.0, 380.0)
    phase_axis.set_yticks([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    phase_axis.grid(True, axis="y", color="0.86", linewidth=0.8, zorder=0)
    phase_axis.spines["top"].set_visible(False)
    phase_axis.spines["right"].set_visible(False)
    apply_axis_text_style(
        phase_axis,
        ylabel=r"$S_{11}$ phase [deg]",
        title=f"$S_{{11}}$ phase at positions {position_text}",
        config=config,
        compact=True,
    )

    figure.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.16)
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def plot_position_phase_advance_bars(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    positions: Sequence[float],
    marker_order: Sequence[str] = MARKER_ORDER,
    ideal_phase_targets_deg: Mapping[float, Mapping[str, float]] | None = None,
    config: PlotConfig | None = None,
) -> Path:
    """Plot measured and ideal phase advance as a separate bar figure."""

    config = config or PlotConfig()
    apply_plot_style(config)
    comparison = build_position_phase_target_table(
        marker_points,
        positions=positions,
        marker_order=marker_order,
        ideal_phase_targets_deg=ideal_phase_targets_deg,
    )
    requested_positions = tuple(float(position) for position in positions)
    requested_markers = tuple(str(marker) for marker in marker_order)
    group_centers = np.arange(len(requested_markers), dtype=float)
    to_position = requested_positions[1]
    advance_rows = comparison[
        np.isclose(
            comparison["tune_position"].to_numpy(dtype=float),
            to_position,
            rtol=0.0,
            atol=1e-12,
        )
    ].set_index("marker_name")
    require_finite_plot_columns(
        advance_rows.reset_index(),
        columns=(
            "phase_advance_0to360_deg",
            "ideal_phase_advance_deg",
        ),
        context="position phase advance bars",
    )
    measured_advances = advance_rows.loc[
        list(requested_markers), "phase_advance_0to360_deg"
    ].to_numpy(dtype=float)
    ideal_advances = advance_rows.loc[
        list(requested_markers), "ideal_phase_advance_deg"
    ].to_numpy(dtype=float)
    phase_advance_errors = advance_rows.loc[
        list(requested_markers), "phase_advance_error_deg"
    ].to_numpy(dtype=float)

    figure, advance_axis = plt.subplots(figsize=(8.6, 5.8))
    ideal_bars = advance_axis.bar(
        group_centers,
        ideal_advances,
        width=ADVANCE_IDEAL_BAR_WIDTH,
        color=ADVANCE_IDEAL_COLOR,
        edgecolor=ADVANCE_IDEAL_EDGE_COLOR,
        linewidth=0.9,
        zorder=2,
        label="Ideal",
    )
    measured_bars = advance_axis.bar(
        group_centers,
        measured_advances,
        width=ADVANCE_MEASURED_BAR_WIDTH,
        color=ADVANCE_MEASURED_COLOR,
        edgecolor=ADVANCE_MEASURED_EDGE_COLOR,
        linewidth=0.9,
        zorder=3,
        label="Measured",
    )
    for bar, measured, ideal, phase_advance_error in zip(
        measured_bars,
        measured_advances,
        ideal_advances,
        phase_advance_errors,
        strict=True,
    ):
        center = bar.get_x() + bar.get_width() / 2.0
        difference_kind, difference_magnitude = classify_phase_advance_difference(
            measured,
            ideal,
        )
        if difference_kind != "matched":
            difference_color = (
                ADVANCE_EXCESS_COLOR
                if difference_kind == "excess"
                else ADVANCE_DEFICIT_COLOR
            )
            difference_hatch = "///" if difference_kind == "excess" else "\\\\\\"
            difference_bottom = min(measured, ideal)
            advance_axis.bar(
                center,
                difference_magnitude,
                bottom=difference_bottom,
                width=ADVANCE_MEASURED_BAR_WIDTH,
                color=difference_color,
                edgecolor=difference_color,
                linewidth=0.7,
                hatch=difference_hatch,
                alpha=0.72,
                zorder=4,
            )
        advance_axis.text(
            center,
            measured + 7.0,
            f"{measured:.1f}°",
            ha="center",
            va="bottom",
            fontsize=config.annotation_size,
            fontweight=config.label_weight,
            color=ADVANCE_MEASURED_EDGE_COLOR,
        )
        advance_axis.annotate(
            f"Ideal {ideal:.0f}°",
            xy=(center + ADVANCE_IDEAL_BAR_WIDTH / 2.0, ideal),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=config.compact_annotation_size,
            fontweight=config.label_weight,
            color=ADVANCE_IDEAL_EDGE_COLOR,
        )
        if difference_magnitude >= 5.0:
            advance_axis.text(
                center,
                min(measured, ideal) + difference_magnitude / 2.0,
                f"{phase_advance_error:+.1f}°",
                ha="center",
                va="center",
                fontsize=config.compact_annotation_size,
                fontweight=config.label_weight,
                color="white",
                zorder=5,
            )
        elif difference_magnitude > 0.0:
            advance_axis.text(
                center,
                ideal - 12.0,
                f"{phase_advance_error:+.1f}°",
                ha="center",
                va="center",
                fontsize=config.compact_annotation_size,
                fontweight=config.label_weight,
                color="white",
                zorder=5,
            )

    advance_axis.bar(
        group_centers,
        ideal_advances,
        width=ADVANCE_IDEAL_BAR_WIDTH,
        facecolor="none",
        edgecolor=ADVANCE_IDEAL_EDGE_COLOR,
        linewidth=1.1,
        zorder=5,
    )

    advance_axis.set_xticks(
        group_centers,
        [MARKER_LABELS.get(marker_name, marker_name) for marker_name in requested_markers],
    )
    advance_axis.set_ylim(-10.0, 380.0)
    advance_axis.set_yticks([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    advance_axis.grid(True, axis="y", color="0.86", linewidth=0.8, zorder=0)
    advance_axis.spines["top"].set_visible(False)
    advance_axis.spines["right"].set_visible(False)
    apply_axis_text_style(
        advance_axis,
        ylabel="Phase advance [deg]",
        title="Phase advance: position 1.0 to 2.0",
        config=config,
        compact=True,
    )
    figure.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.15)
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _format_position(position: float) -> str:
    return f"{position:.1f}"


def _ideal_phase_for_position(
    targets: Mapping[float, Mapping[str, float]],
    *,
    position: float,
    marker_name: str,
) -> float:
    matching = [
        marker_targets
        for target_position, marker_targets in targets.items()
        if np.isclose(
            float(target_position),
            position,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(matching) != 1 or marker_name not in matching[0]:
        raise ValueError(
            "missing unique ideal phase target for "
            f"marker '{marker_name}' at position {position}"
        )
    return float(matching[0][marker_name]) % 360.0


def _wrap180(value: pd.Series | float) -> pd.Series | float:
    return (value + 180.0) % 360.0 - 180.0
