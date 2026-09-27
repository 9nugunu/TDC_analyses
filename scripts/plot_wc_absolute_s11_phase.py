"""Plot absolute S11 phase versus coupler width for a fixed-radius iris scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.marker_styles import MARKER_COLORS
from deflector_tuning.visualization.plot_config import (
    DESIGN_REFERENCE_LINEWIDTH,
    REFERENCE_GUIDE_ALPHA,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)

MARKER_NAME = "f_2pi3"
IRIS_LABEL = r"$f_{2\pi/3}$ Iris"
REQUIRED_COLUMNS = (
    "marker_name",
    "sim_NumDepth",
    "sim_r_c",
    "sim_w_c",
    "s_phase_deg",
)


def build_wc_absolute_phase_table(
    marker_points: pd.DataFrame,
    *,
    fixed_r_c_mm: float,
    num_depth: float = 2.0,
) -> pd.DataFrame:
    """Select the absolute f_2pi3 S11 phase for one fixed-radius iris scan."""

    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker table is missing required columns: {missing}")

    numeric = marker_points.copy()
    for column in ("sim_NumDepth", "sim_r_c", "sim_w_c", "s_phase_deg"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    selected = numeric[
        (numeric["marker_name"] == MARKER_NAME)
        & numeric["sim_NumDepth"].sub(num_depth).abs().le(1e-9)
        & numeric["sim_r_c"].sub(fixed_r_c_mm).abs().le(1e-9)
    ].dropna(subset=["sim_w_c", "s_phase_deg"])
    selected = selected.sort_values("sim_w_c", kind="mergesort")
    selected = selected.drop_duplicates(subset=["sim_w_c"], keep="last").reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"no {MARKER_NAME} iris rows found at r_c={fixed_r_c_mm:g} mm "
            f"and NumDepth={num_depth:g}"
        )
    return selected


def draw_wc_absolute_phase(
    phase_table: pd.DataFrame,
    *,
    fixed_r_c_mm: float,
    reference_w_c_mm: float | None = None,
    config: PlotConfig | None = None,
):
    """Draw the absolute S11 phase curve and return its figure and axis."""

    config = config or PlotConfig()
    apply_plot_style(config)
    if reference_w_c_mm is None and "delta_phase_reference_w_c_mm" in phase_table:
        references = pd.to_numeric(
            phase_table["delta_phase_reference_w_c_mm"],
            errors="coerce",
        ).dropna()
        if not references.empty:
            reference_w_c_mm = float(references.iloc[0])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    color = MARKER_COLORS[MARKER_NAME]
    ax.plot(
        phase_table["sim_w_c"],
        phase_table["s_phase_deg"],
        color=color,
        linewidth=config.line_width,
        marker="o",
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.3,
        label=IRIS_LABEL,
    )
    if reference_w_c_mm is not None:
        ax.axvline(
            reference_w_c_mm,
            color=REFERENCE_GUIDE_COLOR,
            linestyle=REFERENCE_GUIDE_LINESTYLE,
            linewidth=DESIGN_REFERENCE_LINEWIDTH,
            alpha=REFERENCE_GUIDE_ALPHA,
        )

    ax.set_xticks(phase_table["sim_w_c"].tolist())
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, color="0.85", linewidth=0.9)
    ax.set_axisbelow(True)
    apply_axis_text_style(
        ax,
        xlabel=r"$w_c$ [mm]",
        ylabel=r"$\phi_{S_{11}}$ [deg]",
        title=rf"$S_{{11}}$ phase vs $w_c$ ($r_c={fixed_r_c_mm:.2f}$ mm)",
        config=config,
    )
    legend = ax.legend(loc="upper right", frameon=False)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    return fig, ax


def plot_wc_absolute_phase(
    phase_table: pd.DataFrame,
    output_path: str | Path,
    *,
    fixed_r_c_mm: float,
    reference_w_c_mm: float | None = None,
    config: PlotConfig | None = None,
) -> Path:
    """Write the absolute S11 phase curve to a PNG file."""

    config = config or PlotConfig()
    fig, _ = draw_wc_absolute_phase(
        phase_table,
        fixed_r_c_mm=fixed_r_c_mm,
        reference_w_c_mm=reference_w_c_mm,
        config=config,
    )
    try:
        return save_figure(fig, output_path, config)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r-c-mm", type=float, default=56.59)
    parser.add_argument("--num-depth", type=float, default=2.0)
    parser.add_argument("--reference-w-c-mm", type=float)
    args = parser.parse_args()

    marker_points = pd.read_csv(args.input_csv)
    phase_table = build_wc_absolute_phase_table(
        marker_points,
        fixed_r_c_mm=args.r_c_mm,
        num_depth=args.num_depth,
    )
    output_path = plot_wc_absolute_phase(
        phase_table,
        args.output,
        fixed_r_c_mm=args.r_c_mm,
        reference_w_c_mm=args.reference_w_c_mm,
    )
    print(output_path)


if __name__ == "__main__":
    main()
