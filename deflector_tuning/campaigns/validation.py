"""Cross-record relationships and recorded campaign dataset locations."""

from __future__ import annotations

from pathlib import Path

from .models import TuningCampaign


def _validate_relationships(campaign: TuningCampaign) -> None:
    baseline_ids = [
        state_id
        for state_id, state in campaign.states.items()
        if state.role == "baseline"
    ]
    if len(baseline_ids) != 1:
        raise ValueError(
            f"Expected exactly one baseline state in {campaign.source_path}; "
            f"found {baseline_ids}"
        )

    for state_id, state in campaign.states.items():
        if (
            state.previous_state_id is not None
            and state.previous_state_id not in campaign.states
        ):
            raise ValueError(
                f"State {state_id} prev references missing state "
                f"{state.previous_state_id!r}"
            )
    _validate_previous_state_cycles(campaign)

    completed_datasets = [
        state.dataset
        for state in campaign.states.values()
        if state.measurement_status == "done" and state.dataset is not None
    ]
    if len(set(completed_datasets)) != len(completed_datasets):
        raise ValueError(
            f"Campaign {campaign.campaign_id} has a duplicate completed-state dataset"
        )

    state_datasets = set(completed_datasets)
    for dataset, measurement in campaign.auxiliary_measurements.items():
        if measurement.state_id not in campaign.states:
            raise ValueError(
                f"Auxiliary measurement {dataset} references missing state "
                f"{measurement.state_id!r}"
            )
        if dataset in state_datasets:
            raise ValueError(
                f"Campaign {campaign.campaign_id} uses {dataset!r} as both state and auxiliary data"
            )

    for issue_id, issue in campaign.issues.items():
        if issue.after_state_id not in campaign.states:
            raise ValueError(
                f"Issue {issue_id} after references missing state "
                f"{issue.after_state_id!r}"
            )
        if (
            issue.verification_state_id is not None
            and issue.verification_state_id not in campaign.states
        ):
            raise ValueError(
                f"Issue {issue_id} verify references missing state "
                f"{issue.verification_state_id!r}"
            )


def _validate_previous_state_cycles(campaign: TuningCampaign) -> None:
    for starting_id in campaign.states:
        visited: set[str] = set()
        current_id: str | None = starting_id
        while current_id is not None:
            if current_id in visited:
                raise ValueError(
                    f"Campaign {campaign.campaign_id} contains a prev cycle at {current_id}"
                )
            visited.add(current_id)
            current_id = campaign.states[current_id].previous_state_id


def _validate_dataset_paths(campaign: TuningCampaign, data_root: Path) -> None:
    for reference in campaign.simulation_references.values():
        simulation_path = data_root / "sim" / reference.dataset
        if not simulation_path.is_dir():
            raise FileNotFoundError(
                f"Missing data/sim campaign dataset {reference.dataset}: "
                f"{simulation_path}"
            )
    for state_id, state in campaign.states.items():
        if state.measurement_status != "done" or state.dataset is None:
            continue
        dataset_path = data_root / "raw" / state.dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Missing data/raw dataset for completed state {state_id}: {dataset_path}"
            )
    for dataset in campaign.auxiliary_measurements:
        dataset_path = data_root / "raw" / dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Missing data/raw dataset for auxiliary measurement {dataset}: {dataset_path}"
            )
