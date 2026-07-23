"""Plots for matched cell-center and iris-center response comparisons."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "pair_index",
    "cell_pos_from",
    "cell_pos_to",
    "iris_pos_from",
    "iris_pos_to",
    "admit_ratio_iris_cell",
    "phase_ratio_iris_cell",
    "cell_phase_err_deg",
    "iris_phase_err_deg",
    "cell_admit_axis_err_abs_deg",
    "iris_admit_axis_err_abs_deg",
)
FINITE_COLUMNS: tuple[str, ...] = (
    "admit_ratio_iris_cell",
    "phase_ratio_iris_cell",
    "cell_phase_err_deg",
    "iris_phase_err_deg",
    "cell_admit_axis_err_abs_deg",
    "iris_admit_axis_err_abs_deg",
)
MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
MARKER_TICK_LABELS: dict[str, str] = dict(MARKER_LABELS)
FAMILY_COLORS: dict[str, str] = {"cell": "#4c78a8", "iris": "#f58518"}


def plot_cell_iris_response_comparison(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write plots that compare matched cell-center and iris-center transitions."""

    if comparison.empty:
        raise ValueError("cell_iris_response_comparison is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in comparison]
    if missing:
        raise ValueError(f"cell_iris_response_comparison is missing required columns: {missing}")
    require_finite_plot_columns(
        comparison,
        columns=FINITE_COLUMNS,
        context="cell_iris_response_comparison",
        id_columns=("marker_name", "pair_index"),
    )

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = _prepare_table(comparison)

    paths: OrderedDict[str, Path] = OrderedDict()
    paths["iris_over_cell_admittance_response_ratio"] = _plot_response_ratio(
        table,
        folder / "iris_over_cell_admittance_response_ratio.png",
        value_column="admit_ratio_iris_cell",
        ylabel="Iris / Cell",
        title="Iris/Cell |Y11| ratio",
        config=config,
    )
    paths["iris_over_cell_phase_step_ratio"] = _plot_response_ratio(
        table,
        folder / "iris_over_cell_phase_step_ratio.png",
        value_column="phase_ratio_iris_cell",
        ylabel=r"$|\Delta\phi_I| / |\Delta\phi_C|$",
        title=r"Iris/Cell $\Delta\phi$ ratio",
        config=config,
    )
    paths["cell_vs_iris_phase_residual"] = _plot_cell_iris_bars(
        table,
        folder / "cell_vs_iris_phase_residual.png",
        cell_column="cell_phase_err_deg",
        iris_column="iris_phase_err_deg",
        ylabel=r"$|\phi - \phi_0|$ [deg]",
        title="Phase residual",
        config=config,
    )
    paths["cell_vs_iris_operation_axis_error"] = _plot_cell_iris_bars(
        table,
        folder / "cell_vs_iris_operation_axis_error.png",
        cell_column="cell_admit_axis_err_abs_deg",
        iris_column="iris_admit_axis_err_abs_deg",
        ylabel=r"$|\phi - \phi_{axis}|$ [deg]",
        title="Axis error",
        config=config,
    )
    return paths


def _prepare_table(comparison: pd.DataFrame) -> pd.DataFrame:
    table = comparison.copy()
    table["_marker_order"] = table["marker_name"].map(_marker_sort_key)
    table["_pair_sort"] = pd.to_numeric(table["pair_index"], errors="coerce")
    table["_cell_from_sort"] = pd.to_numeric(table["cell_pos_from"], errors="coerce")
    table["_iris_from_sort"] = pd.to_numeric(table["iris_pos_from"], errors="coerce")
    table["comparison_label"] = table.apply(_comparison_label, axis=1)
    return table.sort_values(
        ["_marker_order", "marker_name", "_pair_sort", "_cell_from_sort", "_iris_from_sort"],
        kind="mergesort",
    )


def _plot_response_ratio(
    table: pd.DataFrame,
    output_path: Path,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    config: PlotConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(table))
    x_values = np.arange(len(table))
    colors = [MARKER_COLORS.get(str(marker_name), "#666666") for marker_name in table["marker_name"]]
    ax.bar(
        x_values,
        table[value_column].astype(float),
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, label="equal")
    _format_x_axis(ax, table)
    apply_axis_text_style(ax, xlabel="Transition", ylabel=ylabel, title=title, config=config)
    ax.grid(True, axis="y", color="0.86", linewidth=0.8)
    marker_handles = _marker_handles(table)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend([*handles, *marker_handles], [*labels, *[handle.get_label() for handle in marker_handles]], frameon=True)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_cell_iris_bars(
    table: pd.DataFrame,
    output_path: Path,
    *,
    cell_column: str,
    iris_column: str,
    ylabel: str,
    title: str,
    config: PlotConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(table))
    x_values = np.arange(len(table))
    width = 0.36
    ax.bar(
        x_values - width / 2.0,
        table[cell_column].astype(float),
        width=width,
        color=FAMILY_COLORS["cell"],
        edgecolor="white",
        linewidth=0.8,
        label="Cell",
    )
    ax.bar(
        x_values + width / 2.0,
        table[iris_column].astype(float),
        width=width,
        color=FAMILY_COLORS["iris"],
        edgecolor="white",
        linewidth=0.8,
        label="Iris",
    )
    _format_x_axis(ax, table)
    apply_axis_text_style(ax, xlabel="Transition", ylabel=ylabel, title=title, config=config)
    ax.grid(True, axis="y", color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _format_x_axis(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table["comparison_label"], rotation=35, ha="right")
    ax.tick_params(axis="x", labelrotation=35)


def _comparison_label(row: pd.Series) -> str:
    marker = str(row["marker_name"])
    marker_label = MARKER_TICK_LABELS.get(marker, marker)
    pair = _format_position(row["pair_index"])
    cell = f"C{_format_position(row['cell_pos_from'])}-{_format_position(row['cell_pos_to'])}"
    iris = f"I{_format_position(row['iris_pos_from'])}-{_format_position(row['iris_pos_to'])}"
    return f"{marker_label} P{pair}\n{cell} | {iris}"


def _figure_size(table: pd.DataFrame) -> tuple[float, float]:
    width = min(max(8.8, 1.15 * len(table) + 5.8), 17.0)
    return width, 5.4


def _marker_handles(table: pd.DataFrame) -> list[Patch]:
    handles: list[Patch] = []
    seen: set[str] = set()
    for marker_name in table["marker_name"]:
        marker = str(marker_name)
        if marker in seen:
            continue
        seen.add(marker)
        handles.append(
            Patch(
                facecolor=MARKER_COLORS.get(marker, "#666666"),
                edgecolor="white",
                label=MARKER_LABELS.get(marker, marker),
            )
        )
    return handles


def _marker_sort_key(marker_name: object) -> int:
    marker = str(marker_name)
    try:
        return MARKER_ORDER.index(marker)
    except ValueError:
        return len(MARKER_ORDER)


def _format_position(position: object) -> str:
    try:
        value = float(position)
    except (TypeError, ValueError):
        return str(position)
    return f"{value:g}"
