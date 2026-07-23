"""Normalized views and optional exports for tuning-campaign metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deflector_tuning.tuning_campaign import (
    TuningCampaign,
    TuningCampaignMatch,
    find_tuning_campaign,
    is_tuning_dataset_id,
    tuning_torque_nm_from_dataset_id,
)
from deflector_tuning.workflows.manifest import register_tuning_campaign


DEFAULT_TUNING_CAMPAIGN_DIR = (
    Path(__file__).resolve().parents[2] / "config" / "tuning_campaigns"
)


def register_matching_tuning_campaign(
    dataset_id: str,
    *,
    manifest_path: str | Path,
    campaign_dir: str | Path = DEFAULT_TUNING_CAMPAIGN_DIR,
    data_root: str | Path | None = None,
) -> TuningCampaignMatch | None:
    """Register metadata only for datasets explicitly named with ``_tune_``."""

    if not is_tuning_dataset_id(dataset_id):
        return None
    match = find_tuning_campaign(
        dataset_id,
        campaign_dir,
        data_root=data_root,
    )
    if match is None:
        raise ValueError(
            f"tuning dataset {dataset_id!r} is not registered in {Path(campaign_dir)}"
        )
    if match.state_id is None:
        raise ValueError(
            f"tuning dataset {dataset_id!r} matched campaign "
            f"{match.campaign.campaign_id!r} but no state"
        )
    state = match.campaign.states[match.state_id]
    folder_torque_nm = tuning_torque_nm_from_dataset_id(dataset_id)
    if (
        match.measurement_kind == "state"
        and
        folder_torque_nm is not None
        and state.torque_nm is not None
        and abs(folder_torque_nm - state.torque_nm) > 1e-9
    ):
        raise ValueError(
            f"Dataset {dataset_id!r} folder torque {folder_torque_nm} does not match "
            f"YAML torque {state.torque_nm} for {match.state_id}"
        )
    register_tuning_campaign(
        Path(manifest_path),
        config_path=match.config_path,
        campaign=match.campaign,
        state_id=match.state_id,
        measurement_kind=match.measurement_kind,
    )
    return match


def campaign_state_table(campaign: TuningCampaign) -> pd.DataFrame:
    """Return one normalized row per state without modifying YAML metadata."""

    rows: list[dict[str, object]] = []
    for sequence, (state_id, state) in enumerate(campaign.states.items()):
        rows.append(
            {
                "state_id": state_id,
                "seq": sequence,
                "role": state.role,
                "prev": state.previous_state_id,
                "data": state.dataset,
                "torque_nm": state.torque_nm,
                "ang_deg": state.angle_deg,
                "meas": state.measurement_status,
                "change": state.change,
                "flag": state.quality_flag,
                "unc": (
                    json.dumps(dict(state.uncertainty), sort_keys=True)
                    if state.uncertainty
                    else None
                ),
                "note": state.note,
            }
        )
    return pd.DataFrame(rows)


def campaign_issue_table(campaign: TuningCampaign) -> pd.DataFrame:
    """Return one row per issue with compact defaults made explicit."""

    rows = [
        {
            "issue_id": issue_id,
            "after": issue.after_state_id,
            "verify": issue.verification_state_id,
            "tag": issue.tag,
            "check": ";".join(issue.check),
            "stat": issue.status,
            "sev": issue.severity,
            "action": issue.action,
            "note": issue.note,
        }
        for issue_id, issue in campaign.issues.items()
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "issue_id",
            "after",
            "verify",
            "tag",
            "check",
            "stat",
            "sev",
            "action",
            "note",
        ),
    )


def export_campaign_tables(
    campaign: TuningCampaign, output_dir: str | Path
) -> dict[str, Path]:
    """Write optional derived CSV views while leaving YAML as source of truth."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "states": directory / "states.csv",
        "issues": directory / "issues.csv",
    }
    campaign_state_table(campaign).to_csv(outputs["states"], index=False)
    campaign_issue_table(campaign).to_csv(outputs["issues"], index=False)
    return outputs
