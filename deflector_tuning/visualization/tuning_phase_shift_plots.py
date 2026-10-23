"""Rendering of campaign phase tables; no dataset loading or calculation."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
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
    save_figure,
)


from deflector_tuning.analysis.tuning_phase_shifts import MARKER_ORDER


def _position_color(label: str) -> str:
    return {"iris": "#f58518", "cell": "#4c78a8"}.get(label.split()[0].lower(), "#54a24b")

def _plot_combined(
    shifts: pd.DataFrame,
    output_path: Path,
    display_states: Sequence[str],
    config: PlotConfig,
) -> Path:
    apply_plot_style(config)
    figure, axis = plt.subplots(figsize=(10.4, 5.8))
    marker_x, centers, state_width = _state_marker_coordinates(display_states)
    labels = list(shifts["position_label"].drop_duplicates())
    width = 0.76 / len(labels)
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width
    for offset, label in zip(offsets, labels, strict=True):
        values = _phase_values(shifts, label, display_states)
        bars = axis.bar(
            marker_x + offset,
            values,
            width=width * 0.9,
            color=_position_color(label),
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        axis.bar_label(bars, fmt="%+.1f", padding=3, fontsize=config.compact_annotation_size, color=_position_color(label))
    _style_phase_axis(axis, marker_x, centers, state_width, display_states, config)
    legend = axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    apply_axis_text_style(
        axis,
        xlabel="Frequency mode",
        ylabel=r"$\Delta$ S11 phase [deg]",
        title="S11 phase shift during tuning at fixed positions",
        config=config,
        compact=True,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _plot_one_position(
    shifts: pd.DataFrame,
    output_path: Path,
    position_label: str,
    display_states: Sequence[str],
    config: PlotConfig,
) -> Path:
    apply_plot_style(config)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    marker_x, centers, state_width = _state_marker_coordinates(display_states)
    bars = axis.bar(
        marker_x,
        _phase_values(shifts, position_label, display_states),
        width=0.58,
        color=_position_color(position_label),
        edgecolor="white",
        linewidth=0.8,
    )
    axis.bar_label(bars, fmt="%+.1f", padding=3, fontsize=config.compact_annotation_size, color=_position_color(position_label))
    _style_phase_axis(axis, marker_x, centers, state_width, display_states, config)
    apply_axis_text_style(
        axis,
        xlabel="Frequency mode",
        ylabel=r"$\Delta$ S11 phase [deg]",
        title=f"{position_label}: S11 phase shift",
        config=config,
        compact=True,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _plot_iris_absolute_phase(
    shifts: pd.DataFrame,
    output_path: Path,
    state_order: Sequence[str],
    config: PlotConfig,
    position_label: str = "Iris 2.0",
) -> Path:
    """Plot raw S11 phase at the requested position across tuning states."""
    apply_plot_style(config)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    marker_x, centers, state_width = _state_marker_coordinates(state_order)
    colors = _absolute_phase_colors(state_order)
    bars = axis.bar(
        marker_x,
        _absolute_phase_values(shifts, position_label, state_order),
        width=0.58,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    labels = axis.bar_label(
        bars,
        fmt="%+.1f",
        padding=3,
        fontsize=config.compact_annotation_size,
    )
    for label, color in zip(labels, colors, strict=True):
        label.set_color(color)
    _style_phase_axis(axis, marker_x, centers, state_width, state_order, config)
    apply_axis_text_style(
        axis,
        xlabel="Frequency mode",
        ylabel="S11 phase [deg]",
        title=f"{position_label}: S11 phase",
        config=config,
        compact=True,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _state_marker_coordinates(display_states: Sequence[str]) -> tuple[np.ndarray, np.ndarray, float]:
    marker_count = len(MARKER_ORDER)
    state_width = marker_count + 0.85
    marker_x = np.concatenate(
        [np.arange(marker_count, dtype=float) + index * state_width for index in range(len(display_states))]
    )
    centers = np.arange(len(display_states), dtype=float) * state_width + (marker_count - 1) / 2.0
    return marker_x, centers, state_width


def _absolute_phase_values(
    shifts: pd.DataFrame,
    position_label: str,
    state_order: Sequence[str],
) -> list[float]:
    """Return raw S11 phase values in state-major, marker-minor order."""

    return _phase_values(
        shifts,
        position_label,
        state_order,
        value_column="s_phase_deg",
    )


def _absolute_phase_colors(state_order: Sequence[str]) -> list[str]:
    """Return polar-marker colors in state-major, marker-minor order."""

    return [
        MARKER_COLORS[marker_name]
        for _ in state_order
        for marker_name in MARKER_ORDER
    ]


def _phase_values(
    shifts: pd.DataFrame,
    position_label: str,
    display_states: Sequence[str],
    *,
    value_column: str = "delta_phase_deg",
) -> list[float]:
    values: list[float] = []
    for state_id in display_states:
        for marker_name in MARKER_ORDER:
            rows = shifts.loc[
                shifts["state_id"].eq(state_id)
                & shifts["position_label"].eq(position_label)
                & shifts["marker_name"].eq(marker_name)
            ]
            values.append(float(rows.iloc[0][value_column]) if not rows.empty else np.nan)
    return values


def _style_phase_axis(
    axis,
    marker_x: np.ndarray,
    centers: np.ndarray,
    state_width: float,
    display_states: Sequence[str],
    config: PlotConfig,
) -> None:
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    for boundary in (np.arange(1, len(display_states), dtype=float) * state_width - 0.425):
        axis.axvline(boundary, color="0.87", linewidth=0.8, zorder=0)
    axis.set_xticks(marker_x, [MARKER_LABELS[marker] for _ in display_states for marker in MARKER_ORDER])
    axis.tick_params(axis="x", pad=7)
    top_axis = axis.secondary_xaxis("top")
    top_axis.set_xticks(centers, display_states)
    top_axis.tick_params(axis="x", labelsize=config.compact_tick_size)
    for tick in top_axis.get_xticklabels():
        tick.set_fontweight(config.tick_weight)
    axis.grid(True, axis="y", color="0.86", linewidth=0.8)
    axis.set_axisbelow(True)
