"""Load campaign tables and orchestrate phase-summary products."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import pandas as pd
from deflector_tuning.tuning_campaign import TuningCampaign
from deflector_tuning.visualization.plot_config import PlotConfig
# Reexports preserve existing callers while calculations and rendering live separately.
from deflector_tuning.analysis.tuning_phase_shifts import (
    MARKER_ORDER, build_phase_shift_table, _select_marker_row, _wrap_phase_deg, _position_label,
)
from deflector_tuning.visualization.tuning_phase_shift_plots import (
    _plot_combined, _plot_one_position, _plot_iris_absolute_phase,
    _absolute_phase_values, _absolute_phase_colors, _phase_values, _state_marker_coordinates,
)

@dataclass(frozen=True)
class TuningPhaseShiftOutputs:
    """One campaign summary table and its phase-shift and phase-value figures."""

    table_path: Path
    combined_figure_path: Path
    iris_figure_path: Path | None
    cell_figure_path: Path | None
    iris_absolute_figure_path: Path | None
    position_figure_paths: Mapping[str, Path] = field(default_factory=dict)
    absolute_figure_paths: Mapping[str, Path] = field(default_factory=dict)


def run_tuning_campaign_phase_shifts(
    campaign: TuningCampaign,
    *,
    analysis_root: str | Path,
    output_dir: str | Path,
    current_state_id: str | None = None,
    config: PlotConfig | None = None,
) -> TuningPhaseShiftOutputs | None:
    """Render configured position summaries through the current state."""

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
    if not positions:
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
    figure_paths = {}
    absolute_paths = {}
    for family, reference in campaign.simulation_references.items():
        if not reference.experiment_positions:
            continue
        position = reference.experiment_positions[-1]
        label = _position_label(family, position)
        token = str(float(position)).replace("-", "m").replace(".", "p")
        figure_paths[family] = _plot_one_position(
            shifts, figure_dir / f"tuning_{family}_{token}_marker_phase_shifts_bar.png",
            label, display_states, config,
        )
        absolute_paths[family] = _plot_iris_absolute_phase(
            shifts, figure_dir / f"{family}_{token}_phase.png", state_order, config,
            position_label=label,
        )
    return TuningPhaseShiftOutputs(
        table_path, combined, figure_paths.get("iris"), figure_paths.get("cell"),
        absolute_paths.get("iris"), figure_paths, absolute_paths,
    )
