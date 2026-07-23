"""Campaign-level S11 phase-shift summaries at fixed tuning positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.tuning_campaign import TuningCampaign
from deflector_tuning.visualization.marker_styles import MARKER_LABELS
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)


MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
POSITION_COLORS: Mapping[str, str] = {
    "Iris 2.0": "#f58518",
    "Cell 1.5": "#4c78a8",
}


@dataclass(frozen=True)
class TuningPhaseShiftOutputs:
    """One campaign summary table and its three phase-shift figures."""

    table_path: Path
    combined_figure_path: Path
    iris_figure_path: Path
    cell_figure_path: Path


def build_phase_shift_table(
    state_points: Mapping[str, pd.DataFrame],
    *,
    state_order: Sequence[str],
    positions: Mapping[str, float],
) -> pd.DataFrame:
    """Return wrapped S11 phase shifts relative to each position's first state."""

    rows: list[dict[str, object]] = []
    for position_label, position in positions.items():
        for marker_name in MARKER_ORDER:
            observed: list[tuple[str, pd.Series]] = []
            for state_id in state_order:
                table = state_points.get(state_id)
                if table is None:
                    continue
                marker_row = _select_marker_row(table, position=position, marker_name=marker_name)
                if marker_row is not None:
                    observed.append((state_id, marker_row))
            if not observed:
                continue
            baseline_state, baseline = observed[0]
            baseline_phase = float(baseline["s_phase_deg"])
            for state_id, row in observed:
                phase = float(row["s_phase_deg"])
                rows.append(
                    {
                        "state_id": state_id,
                        "position_label": position_label,
                        "position": position,
                        "source_file": str(row["source_file"]),
                        "marker_name": marker_name,
                        "s_phase_deg": phase,
                        "baseline_state": baseline_state,
                        "baseline_phase_deg": baseline_phase,
                        "delta_phase_deg": _wrap_phase_deg(phase - baseline_phase),
                    }
                )
    return pd.DataFrame(rows)


def run_tuning_campaign_phase_shifts(
    campaign: TuningCampaign,
    *,
    analysis_root: str | Path,
    output_dir: str | Path,
    current_state_id: str | None = None,
    config: PlotConfig | None = None,
) -> TuningPhaseShiftOutputs | None:
    """Render the phase summaries from s002 onward through the current state."""

    all_state_ids = tuple(
        state_id
        for state_id, state in campaign.states.items()
        if state_id != campaign.baseline_state_id and state.dataset is not None
    )
    if current_state_id is not None:
        if current_state_id not in all_state_ids:
            return None
        state_order = all_state_ids[: all_state_ids.index(current_state_id) + 1]
    else:
        state_order = all_state_ids
    if len(state_order) < 2:
        return None
    analysis_root = Path(analysis_root)
    marker_paths = {
        state_id: analysis_root / str(campaign.states[state_id].dataset) / "tables" / "marker_pts.csv"
        for state_id in state_order
    }
    if not all(path.is_file() for path in marker_paths.values()):
        return None

    positions = {
        _position_label(family, reference.experiment_positions[-1]): reference.experiment_positions[-1]
        for family, reference in campaign.simulation_references.items()
        if reference.experiment_positions
    }
    required = {"Iris 2.0", "Cell 1.5"}
    if not required.issubset(positions):
        return None
    state_points = {state_id: pd.read_csv(path) for state_id, path in marker_paths.items()}
    shifts = build_phase_shift_table(state_points, state_order=state_order, positions=positions)
    if shifts.empty:
        return None

    config = config or PlotConfig()
    output_dir = Path(output_dir)
    table_path = output_dir / "tables" / "tuning_marker_phase_shift_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    shifts.to_csv(table_path, index=False)
    figure_dir = output_dir / "figures" / "tuning_phase_shifts"
    display_states = state_order[1:]
    combined = _plot_combined(shifts, figure_dir / "tuning_marker_phase_shifts_grouped_bar.png", display_states, config)
    iris = _plot_one_position(
        shifts,
        figure_dir / "tuning_iris_2p0_marker_phase_shifts_bar.png",
        "Iris 2.0",
        display_states,
        config,
    )
    cell = _plot_one_position(
        shifts,
        figure_dir / "tuning_cell_1p5_marker_phase_shifts_bar.png",
        "Cell 1.5",
        display_states,
        config,
    )
    return TuningPhaseShiftOutputs(table_path, combined, iris, cell)


def _select_marker_row(table: pd.DataFrame, *, position: float, marker_name: str) -> pd.Series | None:
    required = {"source_file", "tune_position", "marker_name", "s_phase_deg"}
    if not required.issubset(table.columns):
        raise ValueError(f"marker_pts is missing required columns: {sorted(required.difference(table.columns))}")
    rows = table.loc[
        ((pd.to_numeric(table["tune_position"], errors="coerce") - position).abs() < 1e-9)
        & table["marker_name"].astype(str).eq(marker_name)
    ].copy()
    rows["s_phase_deg"] = pd.to_numeric(rows["s_phase_deg"], errors="coerce")
    rows = rows[np.isfinite(rows["s_phase_deg"])].copy()
    if rows.empty:
        return None
    port_extended = rows.loc[
        rows["source_file"].astype(str).str.contains("portE", case=False, na=False)
        & ~rows["source_file"].astype(str).str.contains("noportE", case=False, na=False)
    ]
    if len(port_extended) == 1:
        return port_extended.iloc[0]
    if len(rows) == 1:
        return rows.iloc[0]
    raise ValueError(f"Ambiguous marker samples at position={position:g}, marker={marker_name}")


def _wrap_phase_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _position_label(family: str, position: float) -> str:
    return f"{family.title()} {position:.1f}"


def _plot_combined(
    shifts: pd.DataFrame,
    output_path: Path,
    display_states: Sequence[str],
    config: PlotConfig,
) -> Path:
    apply_plot_style(config)
    figure, axis = plt.subplots(figsize=(10.4, 5.8))
    marker_x, centers, state_width = _state_marker_coordinates(display_states)
    offsets = (-0.19, 0.19)
    for offset, label in zip(offsets, ("Iris 2.0", "Cell 1.5"), strict=True):
        values = _phase_values(shifts, label, display_states)
        bars = axis.bar(
            marker_x + offset,
            values,
            width=0.34,
            color=POSITION_COLORS[label],
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        axis.bar_label(bars, fmt="%+.1f", padding=3, fontsize=config.compact_annotation_size, color=POSITION_COLORS[label])
    _style_phase_axis(axis, marker_x, centers, state_width, display_states, config)
    legend = axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=config.compact_legend_size)
    apply_legend_text_style(legend, config)
    apply_axis_text_style(
        axis,
        xlabel="Frequency mode",
        ylabel=r"$\Delta$ S11 phase [deg]",
        title="S11 phase shift during tuning at fixed positions",
        config=config,
        compact=True,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _plot_one_position(
    shifts: pd.DataFrame,
    output_path: Path,
    position_label: str,
    display_states: Sequence[str],
    config: PlotConfig,
) -> Path:
    apply_plot_style(config)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    marker_x, centers, state_width = _state_marker_coordinates(display_states)
    bars = axis.bar(
        marker_x,
        _phase_values(shifts, position_label, display_states),
        width=0.58,
        color=POSITION_COLORS[position_label],
        edgecolor="white",
        linewidth=0.8,
    )
    axis.bar_label(bars, fmt="%+.1f", padding=3, fontsize=config.compact_annotation_size, color=POSITION_COLORS[position_label])
    _style_phase_axis(axis, marker_x, centers, state_width, display_states, config)
    apply_axis_text_style(
        axis,
        xlabel="Frequency mode",
        ylabel=r"$\Delta$ S11 phase [deg]",
        title=f"{position_label}: S11 phase shift",
        config=config,
        compact=True,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path, config)
    plt.close(figure)
    return path


def _state_marker_coordinates(display_states: Sequence[str]) -> tuple[np.ndarray, np.ndarray, float]:
    marker_count = len(MARKER_ORDER)
    state_width = marker_count + 0.85
    marker_x = np.concatenate(
        [np.arange(marker_count, dtype=float) + index * state_width for index in range(len(display_states))]
    )
    centers = np.arange(len(display_states), dtype=float) * state_width + (marker_count - 1) / 2.0
    return marker_x, centers, state_width


def _phase_values(shifts: pd.DataFrame, position_label: str, display_states: Sequence[str]) -> list[float]:
    values: list[float] = []
    for state_id in display_states:
        for marker_name in MARKER_ORDER:
            rows = shifts.loc[
                shifts["state_id"].eq(state_id)
                & shifts["position_label"].eq(position_label)
                & shifts["marker_name"].eq(marker_name)
            ]
            values.append(float(rows.iloc[0]["delta_phase_deg"]) if not rows.empty else np.nan)
    return values


def _style_phase_axis(
    axis,
    marker_x: np.ndarray,
    centers: np.ndarray,
    state_width: float,
    display_states: Sequence[str],
    config: PlotConfig,
) -> None:
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    for boundary in (np.arange(1, len(display_states), dtype=float) * state_width - 0.425):
        axis.axvline(boundary, color="0.87", linewidth=0.8, zorder=0)
    axis.set_xticks(marker_x, [MARKER_LABELS[marker] for _ in display_states for marker in MARKER_ORDER])
    axis.tick_params(axis="x", pad=7)
    top_axis = axis.secondary_xaxis("top")
    top_axis.set_xticks(centers, display_states)
    top_axis.tick_params(axis="x", labelsize=config.compact_tick_size)
    for tick in top_axis.get_xticklabels():
        tick.set_fontweight(config.tick_weight)
    axis.grid(True, axis="y", color="0.86", linewidth=0.8)
    axis.set_axisbelow(True)
