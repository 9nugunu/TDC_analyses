"""Before/after raw-measurement phase comparisons at common port settings."""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
    match_legend_text_colors_to_handles,
    save_figure,
)


REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("tune_position", "marker_name", "s_phase_deg")
MARKER_ORDER: Final[tuple[str, ...]] = ("f_2pi3", "f_mean", "f_pi2")
FILE_STEM: Final[str] = "sparameter_phase_before_after_position_scan"


def plot_before_after_sparameter_phase_position_scan(
    before_marker_points: pd.DataFrame,
    after_marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    before_label: str = "Before tuning",
    after_label: str = "After tuning",
    config: PlotConfig | None = None,
) -> Path:
    """Plot raw S-parameter phase at positions shared by two measurement states.

    Rows without a finite ``tune_position`` are intentionally excluded.  This
    keeps aggregate files such as ``Fullstructure.S2P`` out of a port-setting
    line scan.
    """

    comparison = build_before_after_phase_comparison_table(before_marker_points, after_marker_points)
    config = config or PlotConfig()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(folder / f"{FILE_STEM}.csv", index=False)

    apply_plot_style(config)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for marker_name, marker_table in comparison.groupby("marker_name", sort=False):
        marker_key = str(marker_name)
        color = MARKER_COLORS.get(marker_key, "#444444")
        marker_label = MARKER_LABELS.get(marker_key, marker_key)
        for tune_state, state_label, linestyle, marker in (
            ("before", before_label, "-", "o"),
            ("after", after_label, "--", "s"),
        ):
            state_table = marker_table[marker_table["tune_state"] == tune_state]
            ax.plot(
                state_table["tune_position"].tolist(),
                state_table["s_phase_deg"].tolist(),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=config.line_width,
                markersize=6.0,
                label=f"{marker_label} ({state_label})",
            )

    apply_axis_text_style(
        ax,
        xlabel="Iris-port E setting",
        ylabel="S-parameter phase [deg]",
        title="S-parameter phase vs iris-port E setting (before vs after tuning)",
        config=config,
        compact=True,
    )
    ax.set_ylim(-198.0, 198.0)
    ax.set_xticks(sorted(comparison["tune_position"].unique()))
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=False, fontsize=10.0, ncol=2)
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    fig.tight_layout()
    path = save_figure(fig, folder / f"{FILE_STEM}.png", config)
    plt.close(fig)
    return path


def build_before_after_phase_comparison_table(
    before_marker_points: pd.DataFrame,
    after_marker_points: pd.DataFrame,
) -> pd.DataFrame:
    """Return phase rows only at position-marker pairs present in both states."""

    before = _clean_phase_points(before_marker_points, tune_state="before")
    after = _clean_phase_points(after_marker_points, tune_state="after")
    common_pairs = before[["marker_name", "tune_position"]].merge(
        after[["marker_name", "tune_position"]],
        on=["marker_name", "tune_position"],
        how="inner",
    ).drop_duplicates()
    if common_pairs.empty:
        raise ValueError("before and after marker tables have no common finite position-marker pairs")

    comparison = pd.concat(
        [
            before.merge(common_pairs, on=["marker_name", "tune_position"], how="inner"),
            after.merge(common_pairs, on=["marker_name", "tune_position"], how="inner"),
        ],
        ignore_index=True,
    )
    comparison["_marker_order"] = comparison["marker_name"].map(_marker_sort_key)
    comparison["_state_order"] = comparison["tune_state"].map({"before": 0, "after": 1})
    return (
        comparison.sort_values(["_marker_order", "_state_order", "tune_position"], kind="mergesort")
        .drop(columns=["_marker_order", "_state_order"])
        .reset_index(drop=True)
    )


def _clean_phase_points(marker_points: pd.DataFrame, *, tune_state: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")
    provenance_columns = [
        column
        for column in ("dataset_id", "source_file", "freq_target_ghz", "freq_ghz", "s_db")
        if column in marker_points
    ]
    table = marker_points.loc[:, [*REQUIRED_COLUMNS, *provenance_columns]].copy()
    table["tune_position"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table["s_phase_deg"] = pd.to_numeric(table["s_phase_deg"], errors="coerce")
    table["marker_name"] = table["marker_name"].astype(str)
    table = table[np.isfinite(table["tune_position"]) & np.isfinite(table["s_phase_deg"])].copy()
    if table.empty:
        raise ValueError(f"{tune_state} marker_points has no finite position-phase rows")
    table = table.drop_duplicates(subset=["marker_name", "tune_position"], keep="last")
    table["tune_state"] = tune_state
    return table


def _marker_sort_key(marker_name: object) -> int:
    marker = str(marker_name)
    if marker in MARKER_ORDER:
        return MARKER_ORDER.index(marker)
    return len(MARKER_ORDER)
