from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.analysis.grid_rc_line_scan import (
    DEFAULT_GRID_RC_LINE_SCAN_W_C,
    extract_fixed_width_rc_line_scan,
)
from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    DESIGN_REFERENCE_LINEWIDTH,
    PlotConfig,
    REFERENCE_GUIDE_ALPHA,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)

PHASE_LINE_SCAN_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "sim_r_c",
    "sim_w_c",
    "marker_name",
    "s_phase_deg",
)
PHASE_LINE_SCAN_MARKER_ORDER: Final[tuple[str, ...]] = ("f_2pi3", "f_mean", "f_pi2")
PHASE_RC_MAP_MARKERS: Final[dict[str, str]] = {
    "f_2pi3": "o",
    "f_mean": "s",
    "f_pi2": "^",
}


def plot_grid_scan_sparameter_phase_r_c_line_scan(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    if marker_points.empty:
        raise ValueError("marker_points is empty")
    missing = [column for column in PHASE_LINE_SCAN_REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    file_stem = _sparameter_phase_line_scan_file_stem(fixed_w_c)
    line_scan = _sparameter_phase_line_scan_table(marker_points, fixed_w_c=fixed_w_c)
    if line_scan.empty:
        raise ValueError(f"no S-parameter phases found at fixed w_c={fixed_w_c:g} mm")
    require_finite_plot_columns(
        line_scan,
        columns=("sim_r_c", "s_phase_deg"),
        context="grid_scan S-parameter phase r_c line scan",
        id_columns=("marker_name", "sim_r_c", "sim_w_c", "source_file", "run_id"),
    )

    _write_phase_line_scan_csvs(line_scan, folder, file_stem)
    path = _plot_sparameter_phase_line_scan(
        line_scan,
        folder / f"{file_stem}.png",
        fixed_w_c=fixed_w_c,
        config=config,
    )
    return OrderedDict([(file_stem, path)])


def plot_phase_rc_map(
    line_scan: pd.DataFrame,
    rc_fit: pd.DataFrame,
    output_path: str | Path,
    *,
    fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C,
    config: PlotConfig | None = None,
) -> Path:
    """Plot a phase line scan with fitted baseline/current and design guides."""

    missing = [column for column in ("sim_r_c", "marker_name", "s_phase_deg") if column not in line_scan]
    if missing:
        raise ValueError(f"line_scan is missing required columns: {missing}")
    fit_missing = [column for column in ("state", "r_c_mm") if column not in rc_fit]
    if fit_missing:
        raise ValueError(f"rc_fit is missing required columns: {fit_missing}")
    radii = rc_fit.set_index("state")["r_c_mm"]
    missing_states = [state for state in ("baseline", "current", "design") if state not in radii]
    if missing_states:
        raise ValueError(f"rc_fit is missing required states: {missing_states}")
    config = config or PlotConfig()
    apply_plot_style(config)
    require_finite_plot_columns(
        line_scan,
        columns=("sim_r_c", "s_phase_deg"),
        context="S-parameter phase r_c candidate plot",
        id_columns=("marker_name", "sim_r_c", "sim_w_c", "source_file", "run_id"),
    )
    return _plot_sparameter_phase_line_scan(
        line_scan,
        Path(output_path),
        fixed_w_c=fixed_w_c,
        config=config,
        candidate_r_c=float(radii["current"]),
        candidate_label=rf"Torque 13.5: $r_c={float(radii['current']):.3f}$ mm",
        before_r_c=float(radii["baseline"]),
        before_label=rf"Before tuning: $r_c={float(radii['baseline']):.3f}$ mm",
        target_r_c=float(radii["design"]),
        target_label=rf"Design: $r_c={float(radii['design']):.3f}$ mm",
        show_design_reference=False,
        annotate_design_phase_values=True,
        design_r_c=float(radii["design"]),
        sparse_markers=True,
    )


def plot_phase_rc_states(
    line_scan: pd.DataFrame,
    output_path: str | Path,
    *,
    before_r_c_mm: float,
    current_r_c_mm: float,
    s003_r_c_mm: float,
    design_r_c_mm: float,
    f_mean_zero_r_c_mm: float,
    fixed_w_c: float = DEFAULT_GRID_RC_LINE_SCAN_W_C,
    config: PlotConfig | None = None,
) -> Path:
    """Plot the named tuning states on one simulated ``r_c`` scan."""

    config = config or PlotConfig()
    apply_plot_style(config)
    require_finite_plot_columns(
        line_scan,
        columns=("sim_r_c", "s_phase_deg"),
        context="S-parameter phase r_c state plot",
        id_columns=("marker_name", "sim_r_c", "sim_w_c", "source_file", "run_id"),
    )
    return _plot_sparameter_phase_line_scan(
        line_scan,
        Path(output_path),
        fixed_w_c=fixed_w_c,
        config=config,
        candidate_r_c=float(current_r_c_mm),
        candidate_label=rf"s002 (13.5 N·m): $r_c={float(current_r_c_mm):.3f}$ mm",
        s003_r_c=float(s003_r_c_mm),
        s003_label=rf"s003 tuning: $r_c={float(s003_r_c_mm):.3f}$ mm",
        before_r_c=float(before_r_c_mm),
        before_label=rf"Initial: $r_c={float(before_r_c_mm):.3f}$ mm",
        target_r_c=float(f_mean_zero_r_c_mm),
        target_label="f_mean phase = 0°",
        show_design_reference=True,
        design_r_c=float(design_r_c_mm),
        design_label=None,
        annotate_design_phase_values=True,
        sparse_markers=False,
    )


def _write_phase_line_scan_csvs(line_scan: pd.DataFrame, folder: Path, file_stem: str) -> None:
    line_scan.to_csv(folder / f"{file_stem}.csv", index=False)
    for marker_name, marker_table in line_scan.groupby("marker_name", sort=False, dropna=False):
        marker_token = _filename_token(str(marker_name))
        marker_table.to_csv(folder / f"{file_stem}__{marker_token}.csv", index=False)


def _sparameter_phase_line_scan_table(marker_points: pd.DataFrame, *, fixed_w_c: float) -> pd.DataFrame:
    line_scan = extract_fixed_width_rc_line_scan(marker_points, fixed_w_c=fixed_w_c)
    if line_scan.empty:
        return line_scan
    numeric = line_scan.assign(
        sim_r_c=pd.to_numeric(line_scan["sim_r_c"], errors="coerce"),
        sim_w_c=pd.to_numeric(line_scan["sim_w_c"], errors="coerce"),
        s_phase_deg=pd.to_numeric(line_scan["s_phase_deg"], errors="coerce"),
    ).dropna(subset=["sim_r_c", "sim_w_c", "s_phase_deg"])
    if numeric.empty:
        return numeric
    sorted_table = numeric.assign(
        _marker_order=numeric["marker_name"].map(_marker_sort_key),
    )
    sort_columns = ["_marker_order", "sim_r_c"]
    for optional_column in ("sim_NumDepth", "run_id", "source_file"):
        if optional_column in sorted_table:
            sort_columns.append(optional_column)
    deduplicated = (
        sorted_table.sort_values(sort_columns, kind="mergesort")
        .drop_duplicates(subset=["marker_name", "sim_r_c"], keep="last")
        .drop(columns=["_marker_order"])
        .reset_index(drop=True)
    )
    return deduplicated


def _plot_sparameter_phase_line_scan(
    line_scan: pd.DataFrame,
    output_path: Path,
    *,
    fixed_w_c: float,
    config: PlotConfig,
    candidate_r_c: float | None = None,
    candidate_label: str | None = None,
    s003_r_c: float | None = None,
    s003_label: str | None = None,
    before_r_c: float | None = None,
    before_label: str | None = None,
    target_r_c: float | None = None,
    target_label: str | None = None,
    show_design_reference: bool = True,
    annotate_design_phase_values: bool = True,
    design_r_c: float | None = None,
    design_label: str | None = None,
    sparse_markers: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for marker_name, marker_table in line_scan.groupby("marker_name", sort=False, dropna=False):
        marker_key = str(marker_name)
        plot_kwargs: dict[str, object] = {
            "linewidth": config.line_width,
            "markersize": 6.0,
            "color": MARKER_COLORS.get(marker_key, "#444444"),
            "label": MARKER_LABELS.get(marker_key, marker_key),
        }
        if sparse_markers:
            plot_kwargs.update(
                {
                    "marker": PHASE_RC_MAP_MARKERS.get(marker_key, "o"),
                    "markevery": max(1, (len(marker_table) + 11) // 12),
                    "markerfacecolor": "white",
                    "markeredgewidth": 1.2,
                }
            )
        else:
            plot_kwargs["marker"] = "o"
        ax.plot(
            marker_table["sim_r_c"].tolist(),
            marker_table["s_phase_deg"].tolist(),
            **plot_kwargs,
        )
    reference_r_c = design_r_c
    if reference_r_c is None:
        reference_r_c = config.design_point_by_axis.get("sim_r_c")
    if show_design_reference and reference_r_c is not None:
        ax.axvline(
            reference_r_c,
            color=REFERENCE_GUIDE_COLOR,
            linestyle=REFERENCE_GUIDE_LINESTYLE,
            linewidth=DESIGN_REFERENCE_LINEWIDTH,
            alpha=REFERENCE_GUIDE_ALPHA,
        )
        if design_label is not None:
            ax.annotate(
                design_label,
                xy=(reference_r_c, 0.75),
                xycoords=("data", "axes fraction"),
                xytext=(7, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=9.0,
                fontweight=config.legend_weight,
                color=REFERENCE_GUIDE_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": REFERENCE_GUIDE_COLOR,
                    "alpha": 0.92,
                },
                zorder=7,
            )
    if annotate_design_phase_values and reference_r_c is not None:
        _annotate_design_phase_values(ax, line_scan, design_r_c=reference_r_c, config=config)
    if candidate_r_c is not None:
        ax.axvline(
            candidate_r_c,
            color="#111111",
            linestyle="--",
            linewidth=1.7,
            zorder=5,
        )
        ax.annotate(
            candidate_label or rf"candidate $r_c={candidate_r_c:.3f}$ mm",
            xy=(candidate_r_c, 0.99),
            xycoords=("data", "axes fraction"),
            xytext=(7, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight=config.legend_weight,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#111111", "alpha": 0.92},
            zorder=7,
        )
    if before_r_c is not None:
        ax.axvline(
            before_r_c,
            color="#8c510a",
            linestyle="-.",
            linewidth=1.7,
            zorder=5,
        )
        ax.annotate(
            before_label or rf"before $r_c={before_r_c:.3f}$ mm",
            xy=(before_r_c, 0.91),
            xycoords=("data", "axes fraction"),
            xytext=(7, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight=config.legend_weight,
            color="#8c510a",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#8c510a", "alpha": 0.92},
            zorder=7,
        )
    if s003_r_c is not None:
        ax.axvline(
            s003_r_c,
            color="#2b8cbe",
            linestyle=(0, (5, 2)),
            linewidth=1.7,
            zorder=5,
        )
        ax.annotate(
            s003_label or rf"S003 tuning $r_c={s003_r_c:.3f}$ mm",
            xy=(s003_r_c, 0.60),
            xycoords=("data", "axes fraction"),
            xytext=(-7, -5),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9.0,
            fontweight=config.legend_weight,
            color="#2b8cbe",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#2b8cbe", "alpha": 0.92},
            zorder=7,
        )
    if target_r_c is not None:
        ax.axvline(
            target_r_c,
            color="#c51b7d",
            linestyle=":",
            linewidth=2.0,
            zorder=5,
        )
        ax.annotate(
            target_label or rf"target $r_c={target_r_c:.3f}$ mm",
            xy=(target_r_c, 0.83),
            xycoords=("data", "axes fraction"),
            xytext=(7, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight=config.legend_weight,
            color="#c51b7d",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c51b7d", "alpha": 0.92},
            zorder=7,
        )
    apply_axis_text_style(
        ax,
        xlabel=r"$r_c$ [mm]",
        ylabel="S-parameter phase [deg]",
        title=rf"S-parameter phase vs $r_c$ ($w_c={fixed_w_c:g}$ mm)",
        config=config,
        compact=True,
    )
    ax.margins(y=0.08)
    legend = ax.legend(frameon=False, fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    ax.grid(True, color="0.86", linewidth=0.8)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _annotate_design_phase_values(
    ax,
    line_scan: pd.DataFrame,
    *,
    design_r_c: float,
    config: PlotConfig,
) -> None:
    design_rows = line_scan[(line_scan["sim_r_c"] - design_r_c).abs() <= 1e-9]
    offsets = {"f_2pi3": (10, -22), "f_mean": (10, -22), "f_pi2": (10, -20)}
    for row in design_rows.itertuples(index=False):
        marker_key = str(getattr(row, "marker_name"))
        phase = float(getattr(row, "s_phase_deg"))
        ax.annotate(
            f"{phase:.1f} deg",
            xy=(design_r_c, phase),
            xytext=offsets.get(marker_key, (8, 0)),
            textcoords="offset points",
            ha="left",
            va="center",
            color=MARKER_COLORS.get(marker_key, "#444444"),
            fontsize=config.compact_annotation_size,
            fontweight=config.legend_weight,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": MARKER_COLORS.get(marker_key, "#444444"),
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": MARKER_COLORS.get(marker_key, "#444444"),
                "linewidth": 0.8,
                "alpha": 0.9,
            },
            clip_on=False,
            zorder=6,
        )


def _sparameter_phase_line_scan_file_stem(fixed_w_c: float) -> str:
    return f"sparameter_phase_vs_r_c_at_w_c_{_number_token(fixed_w_c)}"


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _filename_token(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_")


def _marker_sort_key(marker_name: object) -> int:
    marker = str(marker_name)
    if marker in PHASE_LINE_SCAN_MARKER_ORDER:
        return PHASE_LINE_SCAN_MARKER_ORDER.index(marker)
    return len(PHASE_LINE_SCAN_MARKER_ORDER)
