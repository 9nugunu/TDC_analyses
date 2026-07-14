"""Plots for coupler-cavity frequency, external-Q, and beta estimates."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_file",
    "coupler_freq_ghz",
    "match_freq_ghz",
    "freq_delta_mhz",
    "q_ext",
    "q_ext_target",
    "beta",
    "beta_status",
    "is_valid",
)
FREQUENCY_COLUMNS: tuple[str, ...] = ("coupler_freq_ghz", "match_freq_ghz", "freq_delta_mhz")
EXTERNAL_Q_COLUMNS: tuple[str, ...] = ("q_ext",)
BETA_COLUMNS: tuple[str, ...] = ("beta",)


def plot_coupler_cavity_parameters(
    estimates: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write coupler-cavity parameter summary plots."""

    if estimates.empty:
        raise ValueError("coupler_cavity_parameter_estimates is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in estimates]
    if missing:
        raise ValueError(f"coupler_cavity_parameter_estimates is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = _prepare_table(estimates)
    valid_table = _valid_rows(table)
    if valid_table.empty:
        raise ValueError("coupler_cavity_parameter_estimates has no valid rows to plot")

    paths: OrderedDict[str, Path] = OrderedDict()
    paths["coupler_frequency_shift"] = _plot_frequency_shift(
        valid_table,
        folder / "coupler_frequency_shift.png",
        config=config,
    )
    paths["external_quality_factor"] = _plot_external_quality_factor(
        valid_table,
        folder / "external_quality_factor.png",
        config=config,
    )
    beta_table = _finite_rows(valid_table, BETA_COLUMNS)
    if not beta_table.empty:
        paths["coupling_beta"] = _plot_coupling_beta(
            beta_table,
            folder / "coupling_beta.png",
            config=config,
        )
        geometry_paths = _plot_coupler_beta_geometry_sweeps(beta_table, folder, config=config)
        paths.update(geometry_paths)
    return paths


def _prepare_table(estimates: pd.DataFrame) -> pd.DataFrame:
    table = estimates.copy()
    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce") if "tune_position" in table else np.nan
    table["_run_sort"] = pd.to_numeric(table["run_id"], errors="coerce") if "run_id" in table else np.nan
    table["_source_sort"] = table["source_file"].astype(str)
    table["plot_label"] = table.apply(_row_label, axis=1)
    return table.sort_values(["_tune_sort", "_run_sort", "_source_sort"], kind="mergesort").reset_index(drop=True)


def _valid_rows(table: pd.DataFrame) -> pd.DataFrame:
    mask = table["is_valid"].astype(bool)
    valid = table[mask].copy()
    require_finite_plot_columns(
        valid,
        columns=FREQUENCY_COLUMNS,
        context="coupler_cavity_parameter_estimates",
        id_columns=("source_file", "tune_position", "run_id"),
    )
    require_finite_plot_columns(
        valid,
        columns=EXTERNAL_Q_COLUMNS,
        context="coupler_cavity_parameter_estimates",
        id_columns=("source_file", "tune_position", "run_id"),
    )
    return valid


def _finite_rows(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    numeric = table[list(columns)].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    return table[mask].copy()


def _plot_frequency_shift(table: pd.DataFrame, output_path: Path, *, config: PlotConfig) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(table))
    x_values = np.arange(len(table))
    shift_mhz = table["freq_delta_mhz"].astype(float).to_numpy()
    colors = np.where(shift_mhz >= 0.0, "#4c78a8", "#f58518")
    ax.bar(
        x_values,
        shift_mhz,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0, label="matching frequency")
    _format_x_axis(ax, table)
    apply_axis_text_style(
        ax,
        xlabel="Coupler measurement position",
        ylabel="Freq. offset [MHz]",
        title="Coupler frequency offset from matching target",
        config=config,
    )
    ax.grid(True, axis="y", color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_external_quality_factor(table: pd.DataFrame, output_path: Path, *, config: PlotConfig) -> Path:
    fig, ax = plt.subplots(figsize=_figure_size(table))
    x_values = np.arange(len(table))
    external_q = table["q_ext"].astype(float)
    ax.plot(
        x_values,
        external_q,
        marker="o",
        markersize=6,
        linewidth=2.0,
        color="#4c78a8",
        label=r"estimated $Q_{ec}$",
    )
    target_q = pd.to_numeric(table["q_ext_target"], errors="coerce")
    if np.isfinite(target_q.to_numpy(dtype=float)).any():
        ax.plot(
            x_values,
            target_q,
            marker="s",
            markersize=5,
            linewidth=1.6,
            linestyle="--",
            color="#54a24b",
            label=r"target $Q_{ec}$",
        )
    _apply_positive_log_scale_when_useful(ax, external_q)
    _format_x_axis(ax, table)
    apply_axis_text_style(
        ax,
        xlabel="Coupler measurement position",
        ylabel=r"External $Q_{ec}$",
        title="External quality factor estimate",
        config=config,
    )
    ax.grid(True, axis="y", color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_coupling_beta(table: pd.DataFrame, output_path: Path, *, config: PlotConfig) -> Path:
    require_finite_plot_columns(
        table,
        columns=BETA_COLUMNS,
        context="coupler_cavity_parameter_estimates",
        id_columns=("source_file", "tune_position", "run_id"),
    )
    fig, ax = plt.subplots(figsize=_figure_size(table))
    x_values = np.arange(len(table))
    beta = table["beta"].astype(float)
    colors = np.where(beta.to_numpy() >= 1.0, "#4c78a8", "#f58518")
    ax.bar(
        x_values,
        beta,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, label=r"matched $\beta=1$")
    _format_x_axis(ax, table)
    apply_axis_text_style(
        ax,
        xlabel="Coupler measurement position",
        ylabel=r"Coupling coefficient $\beta$",
        title="Coupler beta estimate",
        config=config,
    )
    ax.grid(True, axis="y", color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_coupler_beta_geometry_sweeps(table: pd.DataFrame, folder: Path, *, config: PlotConfig) -> OrderedDict[str, Path]:
    paths: OrderedDict[str, Path] = OrderedDict()
    for column, label, key, filename in (
        (
            "sim_r_c",
            r"Coupler radius $r_c$ [CST parameter]",
            "coupler_beta_vs_coupler_radius",
            "coupler_beta_vs_coupler_radius.png",
        ),
        (
            "sim_w_c",
            r"Coupler width $w_c$ [CST parameter]",
            "coupler_beta_vs_coupler_width",
            "coupler_beta_vs_coupler_width.png",
        ),
    ):
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.dropna().nunique() < 2:
            continue
        paths[key] = _plot_coupler_beta_vs_geometry_axis(
            table.assign(_geometry_value=values),
            folder / filename,
            xlabel=label,
            series_column="sim_w_c" if column == "sim_r_c" and "sim_w_c" in table else "sim_r_c",
            config=config,
        )
    return paths


def _plot_coupler_beta_vs_geometry_axis(
    table: pd.DataFrame,
    output_path: Path,
    *,
    xlabel: str,
    series_column: str,
    config: PlotConfig,
) -> Path:
    plot_table = table.dropna(subset=["_geometry_value"]).copy()
    plot_table = plot_table.sort_values("_geometry_value", kind="mergesort")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    if series_column not in plot_table or pd.to_numeric(plot_table[series_column], errors="coerce").nunique() < 2:
        group_columns = ["cpl_pos_basis"]
    else:
        group_columns = ["cpl_pos_basis", series_column]
    plot_table = _average_duplicate_geometry_points(plot_table, group_columns)
    for group_values, group in plot_table.groupby(group_columns, dropna=False, sort=False):
        label = _format_geometry_legend_label(group_values, group_columns)
        ax.plot(
            group["_geometry_value"].astype(float),
            group["beta"].astype(float),
            marker="o",
            linewidth=2.0,
            label=label,
        )
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, label=r"matched $\beta=1$")
    apply_axis_text_style(
        ax,
        xlabel=xlabel,
        ylabel=r"Coupling coefficient $\beta$",
        title=r"Coupler coupling coefficient versus geometry",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _average_duplicate_geometry_points(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    columns = [*group_columns, "_geometry_value"]
    aggregated = (
        table.groupby(columns, dropna=False, as_index=False)
        .agg({"beta": "mean"})
        .sort_values([*group_columns, "_geometry_value"], kind="mergesort")
    )
    return aggregated


def _format_x_axis(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table["plot_label"], rotation=35, ha="right")
    ax.tick_params(axis="x", labelrotation=35)


def _apply_positive_log_scale_when_useful(ax: plt.Axes, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(dtype=float)) & (finite > 0.0)]
    if finite.empty:
        return
    if float(finite.max()) / float(finite.min()) > 20.0:
        ax.set_yscale("log")


def _figure_size(table: pd.DataFrame) -> tuple[float, float]:
    width = min(max(8.4, 0.55 * len(table) + 5.8), 17.0)
    return width, 5.2


def _row_label(row: pd.Series) -> str:
    if "cpl_pos_basis" in row and pd.notna(row["cpl_pos_basis"]):
        basis = ""
        basis = str(row["cpl_pos_basis"]).replace("_", " ")
        if "tune_position" in row and pd.notna(row["tune_position"]):
            return f"{basis}\nposition {_format_number(row['tune_position'])}"
        return basis
    if "tune_position" in row and pd.notna(row["tune_position"]):
        return f"tune {_format_number(row['tune_position'])}"
    if "run_id" in row and pd.notna(row["run_id"]):
        return f"run {_format_number(row['run_id'])}"
    return str(row["source_file"])


def _format_basis_label(value: object) -> str:
    if pd.isna(value):
        return "coupler"
    return str(value).replace("_", " ")


def _format_geometry_legend_label(group_values: object, group_columns: list[str]) -> str:
    values = group_values if isinstance(group_values, tuple) else (group_values,)
    for column, value in zip(group_columns, values, strict=True):
        if column == "sim_w_c":
            return rf"width $w_c$ = {_format_number(value)}"
        if column == "sim_r_c":
            return rf"radius $r_c$ = {_format_number(value)}"
    for column, value in zip(group_columns, values, strict=True):
        if column == "cpl_pos_basis":
            return _format_basis_label(value)
    return "geometry sweep"


def _format_number(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return f"{number:.1f}"
    return f"{number:g}"
