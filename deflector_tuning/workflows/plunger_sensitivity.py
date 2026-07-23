"""Campaign-configured experimental phase response to plunger offset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from deflector_tuning.analysis.phase_offset_response import build_phase_offset_response
from deflector_tuning.project_defaults import ProjectDefaults
from deflector_tuning.tuning_campaign import TuningCampaignMatch
from deflector_tuning.visualization.phase_offset_plots import plot_phase_offset_response
from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.workflows.manifest import register_phase_offset_response
from deflector_tuning.workflows.models import RunResult


@dataclass(frozen=True)
class PlungerSensitivityResult:
    """Paths written for one plunger-offset phase response."""

    table_path: Path
    figure_path: Path | None


def run_plunger_sensitivity(
    campaign_match: TuningCampaignMatch,
    *,
    current_result: RunResult,
    render_figure: bool,
    project_defaults: ProjectDefaults,
) -> PlungerSensitivityResult:
    """Build phase change versus offset for an auxiliary campaign measurement."""

    sensitivity = campaign_match.phase_offset_sensitivity
    if sensitivity is None:
        raise ValueError("Campaign match does not define plunger-offset sensitivity")
    marker_points = pd.read_csv(current_result.tables["marker_pts"])
    response = build_phase_offset_response(
        marker_points,
        reference_offset_mm=sensitivity.reference_offset_mm,
    )
    table_path = current_result.output_dir / "tables" / "phase_vs_plunger_offset.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(table_path, index=False)
    figure_path = (
        plot_phase_offset_response(
            response,
            current_result.output_dir / "figures" / "plunger_sensitivity",
            config=PlotConfig(
                design_point_by_axis=dict(project_defaults.design_point_by_axis),
                ideal_phase_guide_angles_deg=project_defaults.ideal_phase_guide_angles_deg,
            ),
        )
        if render_figure
        else None
    )
    register_phase_offset_response(
        current_result.manifest_path,
        table_path=table_path,
        figure_path=figure_path,
        reference_offset_mm=sensitivity.reference_offset_mm,
    )
    return PlungerSensitivityResult(table_path=table_path, figure_path=figure_path)
