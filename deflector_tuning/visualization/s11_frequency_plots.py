"""S11 frequency-domain plots with marker-frequency point overlays."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style, save_figure

REQUIRED_SPARAMETER_COLUMNS: tuple[str, ...] = ("source_file", "tune_position", "freq_ghz", "s_db", "s_phase_deg")
REQUIRED_MARKER_COLUMNS: tuple[str, ...] = (
    "source_file",
    "tune_position",
    "marker_name",
    "freq_ghz",
    "s_db",
    "s_phase_deg",
)
MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
MARKER_LABELS: dict[str, str] = {
    "f_2pi3": r"$f_{2\pi/3}$",
    "f_mean": r"$f_{mean}$",
    "f_pi2": r"$f_{\pi/2}$",
}
MARKER_COLORS: dict[str, str] = {
    "f_2pi3": "#2e7d32",
    "f_mean": "#ff6f00",
    "f_pi2": "#1565c0",
}


def plot_s11_with_markers(
    sparameter_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_by_position: bool = True,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write S11 frequency plots with marker point overlays.

    The style follows the existing KYHL S11 frequency views: a blue S11 trace,
    colored marker vertical guides/points, and compact marker annotations.
    """

    if sparameter_table.empty:
        raise ValueError("sparameter_table is empty")
    missing_s = [column for column in REQUIRED_SPARAMETER_COLUMNS if column not in sparameter_table]
    if missing_s:
        raise ValueError(f"sparameter_table is missing required columns: {missing_s}")
    missing_m = [column for column in REQUIRED_MARKER_COLUMNS if column not in marker_points]
    if missing_m:
        raise ValueError(f"marker_points is missing required columns: {missing_m}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    s_table = sparameter_table.copy()
    m_table = marker_points.copy()
    paths: OrderedDict[str, Path] = OrderedDict()
    paths["overview"] = _plot_one(
        s_table,
        m_table,
        folder / "s11_with_markers.png",
        title="S11 magnitude with marker points",
        config=config,
    )
    if split_by_position:
        for tune_position, group in s_table.groupby("tune_position", dropna=False, sort=True):
            marker_group = m_table[m_table["tune_position"] == tune_position]
            key = f"position_{_format_position_key(tune_position)}"
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"s11_position_{_format_position_key(tune_position)}.png",
                title=f"Position {_format_position(tune_position)}: S11 magnitude",
                config=config,
            )
    return paths


def _plot_one(
    s_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    config: PlotConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for source_file, group in s_table.groupby("source_file", sort=False):
        group = group.sort_values("freq_ghz")
        label = _source_label(source_file)
        ax.plot(group["freq_ghz"], group["s_db"], color="#1565c0", linewidth=2.4, label=label)

    if not marker_points.empty:
        y_min, y_max = _axis_marker_bounds(s_table["s_db"])
        for _, point in marker_points.sort_values(["source_file", "freq_ghz", "marker_name"]).iterrows():
            marker_name = point["marker_name"]
            color = MARKER_COLORS.get(marker_name, "#333333")
            ax.axvline(point["freq_ghz"], color=color, linestyle="--", linewidth=1.0, alpha=0.45)
            ax.scatter(
                [point["freq_ghz"]],
                [point["s_db"]],
                s=72,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
            ax.annotate(
                _marker_annotation(point),
                xy=(point["freq_ghz"], point["s_db"]),
                xytext=(8, 16 if marker_name != "f_mean" else -42),
                textcoords="offset points",
                color=color,
                fontsize=config.annotation_size,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.92},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.9},
            )
        ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=config.title_size, fontweight="bold")
    ax.set_xlabel("Freq. (GHz)", fontsize=config.label_size, fontweight="bold")
    ax.set_ylabel(r"S11 (dB)", fontsize=config.label_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=max(config.label_size - 1, 1))
    ax.grid(True, which="major", color="0.78", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="0.90", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.legend(frameon=True, loc="best", fontsize=config.label_size)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _axis_marker_bounds(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * 0.25, 3.0)
    return low - padding, high + padding


def _marker_annotation(point: pd.Series) -> str:
    label = MARKER_LABELS.get(str(point["marker_name"]), str(point["marker_name"]))
    return f"{label}\n{float(point['s_phase_deg']):+.1f}°"


def _source_label(source_file: object) -> str:
    return Path(str(source_file)).stem.replace("_processed", "")


def _format_position(position: object) -> str:
    value = float(position)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def _format_position_key(position: object) -> str:
    return _format_position(position).replace(".", "p")
