"""Compact plots for 3D field-energy and local Slater summaries."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)

E_COLORS = ("#3b82c4", "#70a9d7", "#b7d5ea")
H_COLORS = ("#c65a4a", "#df8a76", "#efc0b4")


def plot_field3d_components(
    energy_table: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Plot E/H component shares for every exported case."""

    config = config or PlotConfig()
    apply_plot_style(config)
    labels = [_short_case_label(value) for value in energy_table["case_id"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    for ax, prefix, colors in (
        (axes[0], "e", E_COLORS),
        (axes[1], "h", H_COLORS),
    ):
        bottom = np.zeros(len(energy_table))
        for component, color in zip(("x", "y", "z"), colors, strict=True):
            values = energy_table[f"{prefix}{component}_pct"].to_numpy(float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=rf"${prefix.upper()}_{component}$",
            )
            bottom += values
        ax.set_ylim(0.0, 100.0)
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        apply_axis_text_style(
            ax,
            ylabel="Energy share [%]",
            title=f"{prefix.upper()} components",
            config=config,
            compact=True,
        )
        legend = ax.legend(ncol=3, frameon=False, loc="upper right")
        apply_legend_text_style(legend, config)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    apply_axis_text_style(
        axes[-1],
        xlabel="Field case",
        config=config,
        compact=True,
    )
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def plot_slater_positions(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Plot local electric, magnetic, and signed E-minus-H Slater terms."""

    config = config or PlotConfig()
    apply_plot_style(config)
    ordered, x, groups = _slater_grouped_layout(table)
    labels = [
        _short_case_label(value, fallback=f"P{index}")
        for index, value in enumerate(ordered["case_ids"], start=1)
    ]
    scale = 1.0e6
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    baseline_k = ordered["k_over_u"].to_numpy(float) * scale
    inserted_k = ordered["inserted_k_over_u"].to_numpy(float) * scale
    for group_index, (group_name, indices) in enumerate(groups):
        group_x = x[indices]
        ax.axvspan(
            group_x[0] - 0.55,
            group_x[-1] + 0.55,
            color=("#eaf3fa" if group_name == "Cell" else "#fdf0ed"),
            alpha=0.55,
            zorder=0,
        )
        ax.plot(
            group_x,
            baseline_k[indices],
            color="#222222",
            marker="D",
            markersize=6,
            linewidth=1.8,
            label=(
                r"$K/U$ (NoPlunger field)"
                if group_index == 0
                else None
            ),
        )
        ax.plot(
            group_x,
            inserted_k[indices],
            color="#d97706",
            linestyle="--",
            marker="o",
            markersize=6,
            linewidth=1.8,
            label=(
                r"$K/U$ (inserted-state field)"
                if group_index == 0
                else None
            ),
        )
        ax.text(
            float(np.mean(group_x)),
            -0.10,
            group_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=config.compact_label_size,
            fontweight=config.label_weight,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    apply_axis_text_style(
        ax,
        xlabel="Plunger position",
        ylabel=r"Normalized local term $K/U$ [ppm]",
        title="Normalized local Slater terms: baseline vs inserted state",
        config=config,
        compact=True,
    )
    ax.xaxis.labelpad = 42
    legend = ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        fontsize=max(config.compact_legend_size - 2, 8),
    )
    apply_legend_text_style(legend, config)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def plot_slater_volumes(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Plot cumulative NoPlunger energy over each inserted cylinder."""

    config = config or PlotConfig()
    apply_plot_style(config)
    ordered, x, groups = _slater_grouped_layout(table)
    labels = [
        _short_case_label(value, fallback=f"P{index}")
        for index, value in enumerate(ordered["case_ids"], start=1)
    ]
    scale = 1.0e12
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.bar(
        x - width / 2,
        ordered["e_j"].to_numpy(float) * scale,
        width,
        color=E_COLORS[0],
        label=r"$U_E^{\Delta V}$",
    )
    ax.bar(
        x + width / 2,
        ordered["h_j"].to_numpy(float) * scale,
        width,
        color=H_COLORS[0],
        label=r"$U_H^{\Delta V}$",
    )
    k_values = ordered["k_e_minus_h_j"].to_numpy(float) * scale
    for group_index, (group_name, indices) in enumerate(groups):
        group_x = x[indices]
        ax.axvspan(
            group_x[0] - 0.55,
            group_x[-1] + 0.55,
            color=("#eaf3fa" if group_name == "Cell" else "#fdf0ed"),
            alpha=0.55,
            zorder=0,
        )
        ax.plot(
            group_x,
            k_values[indices],
            color="#222222",
            marker="D",
            markersize=6,
            linewidth=1.8,
            label=r"$K_{E-H}$" if group_index == 0 else None,
        )
        ax.text(
            float(np.mean(group_x)),
            -0.10,
            group_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=config.compact_label_size,
            fontweight=config.label_weight,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    apply_axis_text_style(
        ax,
        xlabel="Plunger position",
        ylabel="Integrated energy [pJ]",
        title="Cumulative Slater overlap",
        config=config,
        compact=True,
    )
    ax.xaxis.labelpad = 42
    legend = ax.legend(frameon=False, ncol=3, loc="best")
    apply_legend_text_style(legend, config)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _slater_grouped_layout(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[str, np.ndarray]]]:
    """Order half-cell and iris positions into visibly separated groups."""

    depths = table["case_ids"].map(_case_depth)
    families = depths.map(
        lambda value: "Cell" if np.isclose(value % 1.0, 0.5) else "Iris"
    )
    ordered = (
        table.assign(_depth=depths, _family=families)
        .sort_values(
            ["_family", "_depth"],
            key=lambda values: values.map({"Cell": 0, "Iris": 1})
            if values.name == "_family"
            else values,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    group_names = list(dict.fromkeys(ordered["_family"]))
    x = np.empty(len(ordered), dtype=float)
    groups: list[tuple[str, np.ndarray]] = []
    cursor = 0.0
    for group_name in group_names:
        indices = np.flatnonzero(ordered["_family"].to_numpy() == group_name)
        x[indices] = cursor + np.arange(len(indices), dtype=float)
        groups.append((group_name, indices))
        cursor = x[indices[-1]] + 2.0
    return ordered.drop(columns=["_depth", "_family"]), x, groups


def _case_depth(value: object) -> float:
    match = re.search(
        r"NumDepth\s*([0-9]+(?:\.[0-9]+)?)",
        str(value),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot infer plunger position from case_ids={value!r}")
    return float(match.group(1))


def _short_case_label(value: object, *, fallback: str | None = None) -> str:
    text = str(value)
    match = re.search(r"NumDepth\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match is not None:
        return match.group(1)
    if text.lower() == "noplunger":
        return "No plunger"
    if text.lower() == "default":
        return "Default"
    return fallback or text
