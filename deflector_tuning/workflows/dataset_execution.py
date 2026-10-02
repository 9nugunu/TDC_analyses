"""Shared dataset analysis and tuning-campaign postprocessing policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS, ProjectDefaults
from deflector_tuning.runner import run_folder_analysis
from deflector_tuning.tuning_campaign import TuningCampaignMatch
from deflector_tuning.workflows.models import RunResult
from deflector_tuning.workflows.plunger_sensitivity import (
    PlungerSensitivityResult,
    run_plunger_sensitivity,
)
from deflector_tuning.workflows.tuning_campaign import (
    register_matching_tuning_campaign,
    resolve_tuning_marker_correction,
)
from deflector_tuning.workflows.tuning_phase_shifts import (
    TuningPhaseShiftOutputs,
    run_tuning_campaign_phase_shifts,
)
from deflector_tuning.workflows.tuning_simulation_comparison import (
    TuningCmpResult,
    run_tuning_cmp,
)


@dataclass(frozen=True)
class DatasetExecutionResult:
    """Analysis outputs and optional campaign products for one dataset."""

    analysis: RunResult
    campaign_match: TuningCampaignMatch | None
    plunger_sensitivity: PlungerSensitivityResult | None = None
    tuning_cmp: TuningCmpResult | None = None
    phase_shifts: TuningPhaseShiftOutputs | None = None


def run_dataset_analysis(
    *,
    dataset_id: str,
    sparameter_path: str | Path,
    output_dir: str | Path,
    marker_role: str,
    dispersion_path: str | Path | None = None,
    data_root: str | Path = Path("data"),
    file_workers: int = 1,
    plot_workers: int = 1,
    tables_only: bool = False,
    project_defaults: ProjectDefaults = DEFAULT_PROJECT_DEFAULTS,
    geometry_sweep_axis: str | None = None,
    geometry_sweep_base: float | None = None,
) -> DatasetExecutionResult:
    """Run a resolved dataset with the common single/batch scientific policy.

    ``dataset_id`` identifies the already-resolved input for campaign lookup.
    Scheduling, input discovery, CLI output, and batch table checks belong to
    the callers; the lower-level runner remains usable without campaign work.
    """

    marker_correction = (
        resolve_tuning_marker_correction(dataset_id, data_root=data_root)
        if marker_role == "exp"
        else None
    )
    result = run_folder_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        output_dir=output_dir,
        marker_role=marker_role,
        data_root=data_root,
        file_workers=file_workers,
        plot_workers=plot_workers,
        tables_only=tables_only,
        project_defaults=project_defaults,
        marker_correction=marker_correction,
        geometry_sweep_axis=geometry_sweep_axis,
        geometry_sweep_base=geometry_sweep_base,
    )
    campaign_match = register_matching_tuning_campaign(
        dataset_id,
        manifest_path=result.manifest_path,
        data_root=data_root,
    )
    plunger_sensitivity = None
    tuning_cmp = None
    phase_shifts = None
    if campaign_match is not None:
        if campaign_match.phase_offset_sensitivity is not None:
            plunger_sensitivity = run_plunger_sensitivity(
                campaign_match,
                current_result=result,
                render_figure=not tables_only,
                project_defaults=project_defaults,
            )
        if campaign_match.comparison_enabled and not tables_only:
            tuning_cmp = run_tuning_cmp(
                campaign_match,
                current_result=result,
                data_root=data_root,
                file_workers=file_workers,
                plot_workers=plot_workers,
                project_defaults=project_defaults,
            )
        if campaign_match.measurement_kind == "state" and not tables_only:
            phase_shifts = run_tuning_campaign_phase_shifts(
                campaign_match.campaign,
                analysis_root=Path(result.output_dir).parent,
                output_dir=result.output_dir,
                current_state_id=campaign_match.state_id,
            )
    return DatasetExecutionResult(
        analysis=result,
        campaign_match=campaign_match,
        plunger_sensitivity=plunger_sensitivity,
        tuning_cmp=tuning_cmp,
        phase_shifts=phase_shifts,
    )
