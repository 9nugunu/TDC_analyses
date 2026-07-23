"""Plot f_2pi3 same-family transitions versus regular-cell radius."""

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

from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)

MARKER_NAME = "f_2pi3"
TARGET_PHASE_ADVANCE_DEG = 240.0
FAMILY_SPECS: tuple[tuple[str, float, float, str, str], ...] = (
    ("cell", 3.5, 4.5, "#0072B2", "Cell 3.5→4.5"),
    ("iris", 4.0, 5.0, "#D55E00", "Iris 4.0→5.0"),
)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "marker_name",
    "tune_position",
    "s_phase_deg",
    "source_file",
)


def build_transition_sweep(
    reference_points: pd.DataFrame,
    sweep_points: pd.DataFrame,
    *,
    base_radius_mm: float,
) -> pd.DataFrame:
    """Combine fixed upstream references with fine downstream radius-sweep points."""

    _require_columns(reference_points, REQUIRED_COLUMNS, table_name="reference marker table")
    _require_columns(
        sweep_points,
        (*REQUIRED_COLUMNS, "sim_offset_cell_03"),
        table_name="fine-sweep marker table",
    )
    reference = reference_points[reference_points["marker_name"] == MARKER_NAME].copy()
    sweep = sweep_points[sweep_points["marker_name"] == MARKER_NAME].copy()
    reference["_position"] = pd.to_numeric(reference["tune_position"], errors="coerce")
    sweep["_position"] = pd.to_numeric(sweep["tune_position"], errors="coerce")
    sweep["_offset"] = pd.to_numeric(sweep["sim_offset_cell_03"], errors="coerce")
    sweep = sweep[sweep["_offset"].notna()].copy()

    rows: list[dict[str, object]] = []
    for family, pos_from, pos_to, _, _ in FAMILY_SPECS:
        from_row = _unique_position_row(reference, pos_from, context=f"{family} reference")
        to_rows = sweep[(sweep["_position"] - pos_to).abs() < 1e-9].sort_values(
            "_offset",
            kind="mergesort",
        )
        if to_rows.empty:
            raise ValueError(f"fine-sweep marker table has no {family} rows at position {pos_to:g}")
        if to_rows["_offset"].duplicated().any():
            duplicates = to_rows.loc[to_rows["_offset"].duplicated(keep=False), "_offset"].tolist()
            raise ValueError(f"fine-sweep marker table has duplicate {family} offsets: {duplicates}")

        phase_from = float(from_row["s_phase_deg"])
        for _, to_row in to_rows.iterrows():
            offset = float(to_row["_offset"])
            phase_to = float(to_row["s_phase_deg"])
            phase_advance = (phase_to - phase_from) % 360.0
            rows.append(
                {
                    "marker_name": MARKER_NAME,
                    "family": family,
                    "pos_from": pos_from,
                    "pos_to": pos_to,
                    "offset_cell_03_mm": offset,
                    "r_c_mm": float(base_radius_mm) + offset,
                    "phase_from_deg": phase_from,
                    "phase_to_deg": phase_to,
                    "phase_adv_deg": phase_advance,
                    "phase_err_240_deg": phase_advance - TARGET_PHASE_ADVANCE_DEG,
                    "reference_dataset_id": from_row["dataset_id"],
                    "sweep_dataset_id": to_row["dataset_id"],
                    "reference_source_file": from_row["source_file"],
                    "sweep_source_file": to_row["source_file"],
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["offset_cell_03_mm", "family"], kind="mergesort")
        .reset_index(drop=True)
    )


def plot_transition_sweep(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Write the same-family transition-radius figure."""

    config = PlotConfig(dpi=dpi)
    fig, _ = _draw_transition_sweep(table, config=config)
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _draw_transition_sweep(
    table: pd.DataFrame,
    *,
    config: PlotConfig | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = config or PlotConfig()
    apply_plot_style(config)
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for family, pos_from, pos_to, color, label in FAMILY_SPECS:
        family_table = table[table["family"] == family].sort_values("r_c_mm", kind="mergesort")
        if family_table.empty:
            raise ValueError(f"transition sweep table has no {family} rows")
        ax.plot(
            family_table["r_c_mm"],
            family_table["phase_adv_deg"],
            marker="o" if family == "cell" else "s",
            linestyle="-" if family == "cell" else "--",
            linewidth=2.2,
            color=color,
            label=label,
        )
        zero_rows = family_table[family_table["offset_cell_03_mm"].abs() < 1e-12]
        if not zero_rows.empty:
            zero = zero_rows.iloc[0]
            value = float(zero["phase_adv_deg"])
            ax.annotate(
                f"{family.title()}: {value:.1f}°",
                xy=(float(zero["r_c_mm"]), value),
                xytext=(10, 14 if family == "cell" else -18),
                textcoords="offset points",
                color=color,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            )

    ax.axhline(
        TARGET_PHASE_ADVANCE_DEG,
        color="0.35",
        linestyle=":",
        linewidth=1.4,
        label="Ideal 240°",
    )
    apply_axis_text_style(
        ax,
        xlabel="Regular-cell radius $r_c$ [mm]",
        ylabel="$f_{2\\pi/3}$ reflection phase advance [deg]",
        title="$f_{2\\pi/3}$ same-family transition vs regular-cell radius",
        config=config,
    )
    ax.grid(True, color="0.86", linewidth=0.8)
    legend = ax.legend(frameon=True, loc="best", fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    fig.tight_layout()
    return fig, ax


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], *, table_name: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _unique_position_row(table: pd.DataFrame, position: float, *, context: str) -> pd.Series:
    rows = table[(table["_position"] - position).abs() < 1e-9]
    if len(rows) != 1:
        raise ValueError(f"{context} requires exactly one row at position {position:g}; found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_marker_csv", type=Path)
    parser.add_argument("fine_marker_csv", type=Path)
    parser.add_argument("--base-radius-mm", required=True, type=float)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    args = parser.parse_args()

    table = build_transition_sweep(
        pd.read_csv(args.reference_marker_csv),
        pd.read_csv(args.fine_marker_csv),
        base_radius_mm=args.base_radius_mm,
    )
    table_path = args.analysis_dir / "tables" / "f_2pi3_same_family_transition_vs_radius.csv"
    figure_path = (
        args.analysis_dir
        / "figures"
        / "nodal_shift"
        / "f_2pi3_same_family_transition_vs_radius.png"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_transition_sweep(table, figure_path)
    print(table_path)
    print(figure_path)


if __name__ == "__main__":
    main()
