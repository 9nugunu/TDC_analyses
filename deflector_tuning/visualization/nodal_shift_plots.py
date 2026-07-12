"""Bar and grid-map plots for nodal-shift phase target errors."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.analysis.nodal_shift import MARKER_ORDER
from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.grid_scan_spacing_maps import _default_design_point, _draw_design_crosshair
from deflector_tuning.visualization.plot_config import (
    BEST_MARKER_COLOR,
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    contour_contrast_color,
    save_figure,
)
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.progress import progress_iter

REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "from_tune_position",
    "to_tune_position",
    "phase_error_from_target_deg",
    "abs_phase_error_from_target_deg",
)
REGULAR_NODAL_FAMILIES: tuple[str, ...] = ("cell", "iris")


def plot_nodal_shift(
    nodal_shift: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write nodal-shift bar charts and optional grid-scan objective maps."""

    if nodal_shift.empty:
        raise ValueError("nodal_shift is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in nodal_shift]
    if missing:
        raise ValueError(f"nodal_shift is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = _prepare_table(nodal_shift)

    paths: OrderedDict[str, Path] = OrderedDict()
    families = _position_family_order(table)
    for family in progress_iter(families, desc="Rendering nodal bar figures", total=len(families)):
        family_table = table[table["position_family"] == family].copy()
        if family_table.empty:
            continue
        paths[f"{family}_bar"] = _plot_family_bar(
            family_table,
            folder / f"{family}_bar.png",
            family=family,
            config=config,
        )

    for regular_family in REGULAR_NODAL_FAMILIES:
        regular_table = _regular_family_table(table, regular_family)
        if regular_table.empty:
            continue
        regular_table.drop(columns=["_from_sort", "_to_sort"], errors="ignore").to_csv(
            folder / f"regular_{regular_family}_nodal_shift.csv",
            index=False,
        )
        for marker in MARKER_ORDER:
            marker_table = regular_table[regular_table["marker_name"] == marker].copy()
            if marker_table.empty:
                continue
            paths[f"regular_{regular_family}_{marker}_signed_error"] = _plot_marker_bar(
                marker_table,
                folder / f"regular_{regular_family}_{marker}_signed_error.png",
                marker=marker,
                family=regular_family,
                value_column="phase_error_from_target_deg",
                ylabel="Phase error from target [deg]",
                title=f"Regular-{regular_family} {MARKER_LABELS.get(marker, marker)} nodal-shift error",
                reference_line=0.0,
                config=config,
            )
            paths[f"regular_{regular_family}_{marker}_phase_movement"] = _plot_marker_bar(
                marker_table,
                folder / f"regular_{regular_family}_{marker}_phase_movement.png",
                marker=marker,
                family=regular_family,
                value_column="phase_advance_0to360_deg",
                ylabel="Phase movement [deg]",
                title=f"Regular-{regular_family} {MARKER_LABELS.get(marker, marker)} phase movement",
                reference_line=_target_phase_advance(marker_table),
                config=config,
            )
            paths[f"regular_{regular_family}_{marker}_cumulative_phase"] = _plot_marker_cumulative_phase(
                marker_table,
                folder / f"regular_{regular_family}_{marker}_cumulative_phase.png",
                marker=marker,
                family=regular_family,
                config=config,
            )
            paths[f"regular_{regular_family}_{marker}_cumulative_error"] = _plot_marker_cumulative_error(
                marker_table,
                folder / f"regular_{regular_family}_{marker}_cumulative_error.png",
                marker=marker,
                family=regular_family,
                config=config,
            )

    if {"sim_r_c", "sim_w_c"}.issubset(table.columns):
        grid_table = _grid_objective_table(table)
        if not grid_table.empty:
            grid_table.to_csv(folder / "nodal_shift_grid_objective.csv", index=False)
            for column, label in (
                ("f_2pi3_abs_error_deg", r"$f_{2\pi/3}$ target error [deg]"),
                ("f_pi2_abs_error_deg", r"$f_{\pi/2}$ target error [deg]"),
                ("combined_abs_error_deg", "Combined nodal target error [deg]"),
            ):
                if column in grid_table:
                    paths[column.removesuffix("_deg")] = _plot_grid_map(
                        grid_table,
                        folder / f"{column.removesuffix('_deg')}.png",
                        value_column=column,
                        value_label=label,
                        config=config,
                    )
    return paths


def _prepare_table(nodal_shift: pd.DataFrame) -> pd.DataFrame:
    table = nodal_shift[nodal_shift["marker_name"].isin(MARKER_ORDER)].copy()
    if "position_family" not in table:
        table["position_family"] = table["from_tune_position"].map(_position_family)
    table["_from_sort"] = pd.to_numeric(table["from_tune_position"], errors="coerce")
    table["_to_sort"] = pd.to_numeric(table["to_tune_position"], errors="coerce")
    table["transition_label"] = table.apply(
        lambda row: f"{_format_position(row['from_tune_position'])}->{_format_position(row['to_tune_position'])}",
        axis=1,
    )
    return table.sort_values(["position_family", "_from_sort", "_to_sort", "marker_name"], kind="mergesort")


def _plot_family_bar(table: pd.DataFrame, output_path: Path, *, family: str, config: PlotConfig) -> Path:
    labels = list(dict.fromkeys(table["transition_label"].tolist()))
    x = np.arange(len(labels), dtype=float)
    marker_order = [marker for marker in MARKER_ORDER if marker in set(table["marker_name"])]
    width = min(0.34, 0.72 / max(len(marker_order), 1))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for marker_index, marker in enumerate(marker_order):
        marker_table = (
            table[table["marker_name"] == marker]
            .groupby("transition_label", sort=False, as_index=False)["phase_error_from_target_deg"]
            .mean()
        )
        values_by_label = dict(
            zip(
                marker_table["transition_label"],
                marker_table["phase_error_from_target_deg"],
                strict=True,
            )
        )
        values = [float(values_by_label[label]) if label in values_by_label else np.nan for label in labels]
        offsets = x + (marker_index - (len(marker_order) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            label=MARKER_LABELS.get(marker, marker),
            color=MARKER_COLORS.get(marker),
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    apply_axis_text_style(
        ax,
        xlabel="Transition",
        ylabel="Phase error from target [deg]",
        title=f"{family.title()} nodal-shift target error",
        config=config,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    legend = ax.legend(frameon=False, loc="best", fontsize=config.legend_size)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _regular_family_table(table: pd.DataFrame, family: str) -> pd.DataFrame:
    family_table = table[table["position_family"] == family].copy()
    if family_table.empty:
        return family_table

    group_columns = [
        column
        for column in (
            "dataset_id",
            "data_kind",
            "data_layer",
            "marker_role",
            "port_side",
            "s_name",
            "sim_r_c",
            "sim_w_c",
        )
        if column in family_table
    ]
    if not group_columns:
        min_from = family_table["_from_sort"].min()
        max_to = family_table["_to_sort"].max()
        return family_table[(family_table["_from_sort"] > min_from) & (family_table["_to_sort"] < max_to)].copy()

    regular_groups = []
    for _, group in family_table.groupby(group_columns, dropna=False, sort=False):
        min_from = group["_from_sort"].min()
        max_to = group["_to_sort"].max()
        regular = group[(group["_from_sort"] > min_from) & (group["_to_sort"] < max_to)].copy()
        if not regular.empty:
            regular_groups.append(regular)
    if not regular_groups:
        return family_table.iloc[0:0].copy()
    return pd.concat(regular_groups, ignore_index=True).sort_values(
        ["_from_sort", "_to_sort", "marker_name"],
        kind="mergesort",
    )


def _plot_marker_bar(
    table: pd.DataFrame,
    output_path: Path,
    *,
    marker: str,
    family: str,
    value_column: str,
    ylabel: str,
    title: str,
    reference_line: float | None,
    config: PlotConfig,
) -> Path:
    labels = list(dict.fromkeys(table["transition_label"].tolist()))
    x = np.arange(len(labels), dtype=float)
    grouped = table.groupby("transition_label", sort=False, as_index=False)[value_column].mean()
    values_by_label = dict(zip(grouped["transition_label"], grouped[value_column], strict=True))
    values = [float(values_by_label[label]) if label in values_by_label else np.nan for label in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(
        x,
        values,
        width=0.58,
        color=MARKER_COLORS.get(marker),
        edgecolor="black",
        linewidth=0.6,
    )
    if reference_line is not None:
        ax.axhline(reference_line, color="0.25", linestyle="--", linewidth=1.0)
    apply_axis_text_style(ax, xlabel=f"Regular-{family} transition", ylabel=ylabel, title=title, config=config)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_marker_cumulative_phase(
    table: pd.DataFrame,
    output_path: Path,
    *,
    marker: str,
    family: str,
    config: PlotConfig,
) -> Path:
    cumulative = _cumulative_phase_table(table)
    labels = cumulative["transition_label"].tolist()
    steps = cumulative["step_index"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(
        steps,
        cumulative["measured_cumulative_phase_deg"],
        marker="o",
        linewidth=1.8,
        color=MARKER_COLORS.get(marker),
        label="Measured",
    )
    if "target_cumulative_phase_deg" in cumulative:
        ax.plot(
            steps,
            cumulative["target_cumulative_phase_deg"],
            linestyle="--",
            linewidth=1.4,
            color="0.25",
            label="Target",
        )
    apply_axis_text_style(
        ax,
        xlabel=f"Regular-{family} transition",
        ylabel="Cumulative phase advance [deg]",
        title=f"Regular-{family} {MARKER_LABELS.get(marker, marker)} cumulative phase advance",
        config=config,
    )
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, color="0.88", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.legend_size, framealpha=0.72)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _plot_marker_cumulative_error(
    table: pd.DataFrame,
    output_path: Path,
    *,
    marker: str,
    family: str,
    config: PlotConfig,
) -> Path:
    cumulative = _cumulative_phase_table(table)
    labels = cumulative["transition_label"].tolist()
    steps = cumulative["step_index"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(
        steps,
        cumulative["target_minus_measured_cumulative_phase_deg"],
        marker="o",
        linewidth=1.8,
        color=MARKER_COLORS.get(marker),
    )
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    apply_axis_text_style(
        ax,
        xlabel=f"Regular-{family} transition",
        ylabel="Error [deg]",
        title=f"Regular-{family} {MARKER_LABELS.get(marker, marker)} cumulative phase error",
        config=config,
    )
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _cumulative_phase_table(table: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        table.groupby("transition_label", sort=False, as_index=False)
        .agg(
            phase_advance_0to360_deg=("phase_advance_0to360_deg", "mean"),
            _from_sort=("_from_sort", "min"),
            _to_sort=("_to_sort", "min"),
        )
        .sort_values(["_from_sort", "_to_sort"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["step_index"] = np.arange(1, len(grouped) + 1, dtype=int)
    grouped["measured_cumulative_phase_deg"] = grouped["phase_advance_0to360_deg"].astype(float).cumsum()
    target = _target_phase_advance(table)
    if target is not None:
        grouped["target_cumulative_phase_deg"] = grouped["step_index"].astype(float) * target
        grouped["target_minus_measured_cumulative_phase_deg"] = (
            grouped["target_cumulative_phase_deg"] - grouped["measured_cumulative_phase_deg"]
        )
    return grouped


def _target_phase_advance(table: pd.DataFrame) -> float | None:
    if "target_phase_advance_deg" not in table:
        return None
    targets = pd.to_numeric(table["target_phase_advance_deg"], errors="coerce").dropna().unique()
    if len(targets) == 0:
        return None
    return float(targets[0])


def _grid_objective_table(table: pd.DataFrame) -> pd.DataFrame:
    required = ("sim_r_c", "sim_w_c", "abs_phase_error_from_target_deg")
    if any(column not in table for column in required):
        return pd.DataFrame()
    require_finite_plot_columns(
        table,
        columns=("sim_r_c", "sim_w_c", "abs_phase_error_from_target_deg"),
        context="nodal_shift",
        id_columns=("dataset_id", "source_file", "from_source_file", "to_source_file", "sim_r_c", "sim_w_c"),
    )
    grouped = (
        table.groupby(["sim_r_c", "sim_w_c", "marker_name"], dropna=False, as_index=False)[
            "abs_phase_error_from_target_deg"
        ]
        .mean()
    )
    pivot = grouped.pivot_table(
        index=["sim_r_c", "sim_w_c"],
        columns="marker_name",
        values="abs_phase_error_from_target_deg",
        aggfunc="mean",
    )
    rows = pivot.reset_index()
    rename = {
        "f_2pi3": "f_2pi3_abs_error_deg",
        "f_pi2": "f_pi2_abs_error_deg",
    }
    rows = rows.rename(columns=rename)
    metric_columns = [column for column in rename.values() if column in rows]
    if metric_columns:
        rows["combined_abs_error_deg"] = rows[metric_columns].sum(axis=1)
    return rows.sort_values(["sim_r_c", "sim_w_c"], kind="mergesort").reset_index(drop=True)


def _plot_grid_map(
    table: pd.DataFrame,
    output_path: Path,
    *,
    value_column: str,
    value_label: str,
    config: PlotConfig,
) -> Path:
    x_column = "sim_r_c"
    y_column = "sim_w_c"
    x_values = np.array(sorted(table[x_column].dropna().unique()), dtype=float)
    y_values = np.array(sorted(table[y_column].dropna().unique()), dtype=float)
    pivot = table.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean").reindex(index=y_values, columns=x_values)
    X, Y = np.meshgrid(x_values, y_values)
    Z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    if len(x_values) >= 2 and len(y_values) >= 2:
        cmap = config.contour_error_cmap
        contour_color = contour_contrast_color(cmap, config)
        cf = ax.contourf(X, Y, Z, levels=12, cmap=cmap)
        cs = ax.contour(
            X,
            Y,
            Z,
            levels=8,
            colors=contour_color,
            linewidths=config.contour_line_width,
            alpha=config.contour_line_alpha,
        )
        contour_labels = ax.clabel(cs, inline=True, fontsize=config.contour_label_size, fmt="%.1f", colors=contour_color)
        for label in contour_labels:
            label.set_fontweight(config.contour_label_weight)
    else:
        cf = ax.scatter(table[x_column], table[y_column], c=table[value_column], cmap=config.contour_error_cmap, s=90, edgecolor="black")
    best = table.loc[table[value_column].idxmin()]
    ax.scatter([best[x_column]], [best[y_column]], marker="*", s=190, c=BEST_MARKER_COLOR, edgecolor="black", linewidth=0.8, zorder=6)
    design_point = _default_design_point(x_column, y_column, config=config)
    if design_point is not None:
        _draw_design_crosshair(ax, design_point)
    apply_axis_text_style(ax, xlabel=r"$r_c$ [mm]", ylabel=r"$w_c$ [mm]", title=value_label, config=config)
    colorbar = fig.colorbar(cf, ax=ax, shrink=0.92)
    colorbar.set_label(value_label, fontsize=config.label_size, fontweight=config.label_weight, rotation=-90, labelpad=30)
    colorbar.ax.tick_params(labelsize=config.tick_size)
    ax.grid(True, color="0.85", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _format_position(position: object) -> str:
    try:
        value = float(position)
    except (TypeError, ValueError):
        return str(position)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _position_family(tune_position: object) -> str:
    if pd.isna(tune_position):
        return "unknown"
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{fractional:g}"


def _position_family_order(table: pd.DataFrame) -> list[str]:
    family_order = [family for family in ("cell", "iris") if family in set(table["position_family"])]
    family_order.extend(family for family in table["position_family"].dropna().unique() if family not in family_order)
    return family_order
