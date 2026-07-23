"""Plots that use direct CST Y11 values without impedance normalization."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_plot_style,
    save_figure,
)

MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
REQUIRED_Y_COLUMNS: tuple[str, ...] = (
    "source_file",
    "freq_ghz",
    "y_re_siemens",
    "y_im_siemens",
)
REQUIRED_MARKER_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "y_re_siemens",
    "y_im_siemens",
)


def plot_y11_raw_frequency_with_markers(
    y11_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Plot direct Re(Y11) and Im(Y11) traces with raw marker points."""

    _require_columns(y11_table, REQUIRED_Y_COLUMNS, "y11_table")
    _require_columns(marker_points, REQUIRED_MARKER_COLUMNS, "marker_points")
    if y11_table.empty:
        raise ValueError("y11_table is empty")
    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: OrderedDict[str, Path] = OrderedDict()
    for key, group in _sweep_groups(y11_table):
        marker_group = _matching_group(marker_points, group)
        path = folder / f"{key}.png"
        fig, axes = plt.subplots(2, 1, figsize=config.figure_size, sharex=True)
        for axis, column, label in zip(
            axes,
            ("y_re_siemens", "y_im_siemens"),
            (r"Re($Y_{11}$)", r"Im($Y_{11}$)"),
            strict=True,
        ):
            axis.plot(
                group["freq_ghz"],
                group[column].astype(float) * 1e3,
                color="#1565c0",
                linewidth=config.line_width,
                label=label,
            )
            for marker_name in MARKER_ORDER:
                point = marker_group[marker_group["marker_name"] == marker_name]
                if point.empty:
                    continue
                color = MARKER_COLORS.get(marker_name, "#444444")
                axis.axvline(
                    float(point["freq_target_ghz"].iloc[0]),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                )
                axis.scatter(
                    point["freq_ghz"],
                    point[column].astype(float) * 1e3,
                    color=color,
                    s=config.marker_size,
                    zorder=3,
                    label=MARKER_LABELS.get(marker_name, marker_name),
                )
            axis.set_ylabel(f"{label} [mS]")
            axis.grid(True, color="0.90", linewidth=0.7)
        axes[-1].set_xlabel("Freq. (GHz)")
        axes[0].set_title(f"Raw Y11 | {_group_title(group)}")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="best")
        save_figure(fig, path, config)
        plt.close(fig)
        paths[key] = path
    return paths


def plot_y11_raw_polar_views(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Plot raw |Y11| (mS) and angle(Y11) for each sweep point."""

    _require_columns(marker_points, REQUIRED_MARKER_COLUMNS, "marker_points")
    if marker_points.empty:
        raise ValueError("marker_points is empty")
    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: OrderedDict[str, Path] = OrderedDict()
    for key, group in _sweep_groups(marker_points):
        values = group.set_index("marker_name").reindex(MARKER_ORDER).dropna(
            subset=["y_re_siemens", "y_im_siemens"]
        )
        if values.empty:
            continue
        complex_values = values["y_re_siemens"].astype(float).to_numpy() + 1j * values[
            "y_im_siemens"
        ].astype(float).to_numpy()
        theta = np.angle(complex_values)
        radius = np.abs(complex_values) * 1e3
        color = [MARKER_COLORS.get(name, "#444444") for name in values.index]
        fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw={"projection": "polar"})
        ax.plot(theta, radius, color="#777777", linewidth=config.line_width, alpha=0.8)
        ax.scatter(theta, radius, c=color, s=config.marker_size * 1.3, zorder=3)
        for angle, radial, marker_name, marker_color in zip(theta, radius, values.index, color, strict=True):
            ax.annotate(
                MARKER_LABELS.get(marker_name, marker_name),
                xy=(angle, radial),
                xytext=(6, 6),
                textcoords="offset points",
                color=marker_color,
            )
        ax.set_title(f"Raw Y11 polar | {_group_title(group)}")
        ax.set_ylabel("|Y11| [mS]")
        ax.grid(True, color="0.90", linewidth=0.7)
        path = folder / f"{key}.png"
        save_figure(fig, path, config)
        plt.close(fig)
        paths[key] = path
    return paths


def plot_y11_raw_grid(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Plot raw Re(Y11) and Im(Y11) marker sweeps against r_c."""

    _require_columns(marker_points, REQUIRED_MARKER_COLUMNS, "marker_points")
    if "sim_r_c" not in marker_points:
        raise ValueError("marker_points requires sim_r_c for a raw Y11 grid plot")
    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = marker_points.copy()
    table["sim_r_c"] = pd.to_numeric(table["sim_r_c"], errors="coerce")
    table = table.dropna(subset=["sim_r_c"]).sort_values("sim_r_c", kind="mergesort")
    if table.empty:
        return OrderedDict()
    fig, axes = plt.subplots(2, 1, figsize=config.figure_size, sharex=True)
    for axis, column, label in zip(
        axes,
        ("y_re_siemens", "y_im_siemens"),
        (r"Re($Y_{11}$)", r"Im($Y_{11}$)"),
        strict=True,
    ):
        for marker_name in MARKER_ORDER:
            group = table[table["marker_name"] == marker_name]
            if group.empty:
                continue
            axis.plot(
                group["sim_r_c"],
                group[column].astype(float) * 1e3,
                color=MARKER_COLORS.get(marker_name, "#444444"),
                linewidth=config.line_width,
                label=MARKER_LABELS.get(marker_name, marker_name),
            )
        axis.set_ylabel(f"{label} [mS]")
        axis.grid(True, color="0.90", linewidth=0.7)
    axes[-1].set_xlabel(r"$r_c$ [mm]")
    axes[0].set_title("Raw Y11 marker sweep")
    axes[0].legend(loc="best")
    path = folder / "y11_raw_rc_scan.png"
    save_figure(fig, path, config)
    plt.close(fig)
    return OrderedDict((("y11_raw_rc_scan", path),))


def plot_y11_raw_complex_sweep(
    marker_points: pd.DataFrame,
    output_path: str | Path,
    *,
    config: PlotConfig | None = None,
) -> Path:
    """Plot the direct raw complex Y11 trajectories at the three markers."""

    _require_columns(marker_points, REQUIRED_MARKER_COLUMNS, "marker_points")
    config = config or PlotConfig()
    apply_plot_style(config)
    fig, ax = plt.subplots(figsize=config.figure_size)
    for marker_name in MARKER_ORDER:
        group = marker_points[marker_points["marker_name"] == marker_name].copy()
        if group.empty:
            continue
        group = group.sort_values("sim_r_c" if "sim_r_c" in group else "source_file")
        ax.plot(
            group["y_re_siemens"].astype(float) * 1e3,
            group["y_im_siemens"].astype(float) * 1e3,
            color=MARKER_COLORS.get(marker_name, "#444444"),
            linewidth=config.line_width,
            label=MARKER_LABELS.get(marker_name, marker_name),
        )
    ax.axhline(0.0, color="0.70", linewidth=0.8)
    ax.axvline(0.0, color="0.70", linewidth=0.8)
    ax.set_xlabel(r"Re($Y_{11}$) [mS]")
    ax.set_ylabel(r"Im($Y_{11}$) [mS]")
    ax.set_title("Raw CST Y11 complex sweep")
    ax.legend(loc="best")
    ax.grid(True, color="0.90", linewidth=0.7)
    path = Path(output_path)
    save_figure(fig, path, config)
    plt.close(fig)
    return path


def _sweep_groups(table: pd.DataFrame):
    if "sim_r_c" in table and table["sim_r_c"].notna().any():
        groups = table.copy()
        groups["_r_sort"] = pd.to_numeric(groups["sim_r_c"], errors="coerce")
        for value, group in groups.sort_values("_r_sort", kind="mergesort").groupby(
            "sim_r_c", sort=False, dropna=False
        ):
            yield _number_token(value), group.drop(columns="_r_sort")
        return
    for source_file, group in table.groupby("source_file", sort=True, dropna=False):
        yield Path(str(source_file)).stem, group


def _matching_group(table: pd.DataFrame, reference_group: pd.DataFrame) -> pd.DataFrame:
    if "sim_r_c" in reference_group and reference_group["sim_r_c"].notna().any():
        target = float(pd.to_numeric(reference_group["sim_r_c"], errors="coerce").dropna().iloc[0])
        numeric = pd.to_numeric(table["sim_r_c"], errors="coerce")
        return table[numeric == target]
    source_file = str(reference_group["source_file"].iloc[0])
    return table[table["source_file"].astype(str) == source_file]


def _group_title(group: pd.DataFrame) -> str:
    if "sim_r_c" in group and group["sim_r_c"].notna().any():
        return f"r_c={float(group['sim_r_c'].dropna().iloc[0]):.2f} mm"
    return str(group["source_file"].iloc[0])


def _number_token(value: object) -> str:
    try:
        return f"r_c_{float(value):.2f}".replace(".", "p")
    except (TypeError, ValueError):
        return Path(str(value)).stem


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
