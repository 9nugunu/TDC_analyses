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
    DEFAULT_DESIGN_POINT_BY_AXIS,
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
        columns=("sim_r_c", "s_phase_deg_unwrapped"),
        context="grid_scan S-parameter phase r_c line scan",
        id_columns=("marker_name", "sim_r_c", "sim_w_c", "source_file", "run_id"),
    )

    line_scan.to_csv(folder / f"{file_stem}.csv", index=False)
    path = _plot_sparameter_phase_line_scan(
        line_scan,
        folder / f"{file_stem}.png",
        fixed_w_c=fixed_w_c,
        config=config,
    )
    return OrderedDict([(file_stem, path)])


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
        s_phase_deg_0_360=lambda table: table["s_phase_deg"].mod(360.0),
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
    return _add_unwrapped_phase_by_marker(deduplicated)


def _add_unwrapped_phase_by_marker(line_scan: pd.DataFrame) -> pd.DataFrame:
    unwrapped_groups: list[pd.DataFrame] = []
    for _, marker_table in line_scan.groupby("marker_name", sort=False, dropna=False):
        unwrapped_groups.append(
            marker_table.assign(
                s_phase_deg_unwrapped=_unwrap_phase_deg(marker_table["s_phase_deg"]),
            )
        )
    return pd.concat(unwrapped_groups, ignore_index=True)


def _unwrap_phase_deg(phases: pd.Series) -> list[float]:
    values = [float(phase) for phase in phases]
    if not values:
        return []
    unwrapped = [values[0]]
    for phase in values[1:]:
        previous = unwrapped[-1]
        delta = ((phase - previous + 180.0) % 360.0) - 180.0
        unwrapped.append(previous + delta)
    return unwrapped


def _plot_sparameter_phase_line_scan(
    line_scan: pd.DataFrame,
    output_path: Path,
    *,
    fixed_w_c: float,
    config: PlotConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for marker_name, marker_table in line_scan.groupby("marker_name", sort=False, dropna=False):
        marker_key = str(marker_name)
        ax.plot(
            marker_table["sim_r_c"].tolist(),
            marker_table["s_phase_deg_unwrapped"].tolist(),
            marker="o",
            linewidth=config.line_width,
            markersize=6.0,
            color=MARKER_COLORS.get(marker_key, "#444444"),
            label=MARKER_LABELS.get(marker_key, marker_key),
        )
    design_r_c = DEFAULT_DESIGN_POINT_BY_AXIS.get("sim_r_c")
    if design_r_c is not None:
        ax.axvline(
            design_r_c,
            color=REFERENCE_GUIDE_COLOR,
            linestyle=REFERENCE_GUIDE_LINESTYLE,
            linewidth=DESIGN_REFERENCE_LINEWIDTH,
            alpha=REFERENCE_GUIDE_ALPHA,
        )
        _annotate_design_phase_values(ax, line_scan, design_r_c=design_r_c, config=config)
    apply_axis_text_style(
        ax,
        xlabel=rf"$r_c$ [mm]",
        ylabel="S-parameter phase [deg, unwrapped]",
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
    offsets = {"f_2pi3": (8, 8), "f_mean": (8, -18), "f_pi2": (8, -6)}
    for row in design_rows.itertuples(index=False):
        marker_key = str(getattr(row, "marker_name"))
        phase = float(getattr(row, "s_phase_deg_unwrapped"))
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
        )


def _sparameter_phase_line_scan_file_stem(fixed_w_c: float) -> str:
    return f"sparameter_phase_vs_r_c_at_w_c_{_number_token(fixed_w_c)}"


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _marker_sort_key(marker_name: object) -> int:
    marker = str(marker_name)
    if marker in PHASE_LINE_SCAN_MARKER_ORDER:
        return PHASE_LINE_SCAN_MARKER_ORDER.index(marker)
    return len(PHASE_LINE_SCAN_MARKER_ORDER)
