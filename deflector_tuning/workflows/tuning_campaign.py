"""Normalized views and optional exports for tuning-campaign metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection
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


def resolve_tuning_marker_correction(
    dataset_id: str,
    *,
    campaign_dir: str | Path = DEFAULT_TUNING_CAMPAIGN_DIR,
    data_root: str | Path | None = None,
) -> TemperatureHumidityCorrection | None:
    """Return the state-recorded marker correction for one dataset, if any."""

    match = find_tuning_campaign(
        dataset_id,
        campaign_dir,
        data_root=data_root,
    )
    if match is None or match.state_id is None:
        return None
    return match.campaign.marker_correction_for(match.state_id)


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
    if match.match_mode == "auto":
        _persist_automatic_measurement(match, dataset_id)
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
        match_mode=match.match_mode,
    )
    return match


def _persist_automatic_measurement(
    match: TuningCampaignMatch,
    dataset_id: str,
) -> None:
    """Persist one inferred state or auxiliary measurement in central YAML."""

    config_path = match.config_path
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping YAML root in {config_path}")
    state_id = match.state_id
    if state_id is None:
        raise ValueError(f"Automatic campaign match for {dataset_id!r} has no state")
    if match.measurement_kind == "state":
        states = raw.setdefault("states", {})
        if not isinstance(states, dict):
            raise ValueError(f"Expected states mapping in {config_path}")
        state = match.campaign.states[state_id]
        entry = {
            "prev": state.previous_state_id,
            "data": dataset_id,
            "meas": state.measurement_status,
            "flag": state.quality_flag,
        }
        if state.torque_nm is not None:
            entry["torque_nm"] = state.torque_nm
        existing = states.get(state_id)
        if existing is not None and existing != entry:
            raise ValueError(
                f"Automatic state {state_id!r} already has a different record in {config_path}"
            )
        states.setdefault(state_id, entry)
    elif match.measurement_kind == "aux":
        auxiliary = raw.setdefault("aux", {})
        if not isinstance(auxiliary, dict):
            raise ValueError(f"Expected aux mapping in {config_path}")
        entry = {
            "state": state_id,
            "cmp": False,
            "sens": {"axis": "plunger_offset_mm", "ref_mm": 0},
        }
        existing = auxiliary.get(dataset_id)
        if existing is not None and existing != entry:
            raise ValueError(
                f"Automatic auxiliary dataset {dataset_id!r} already has a different record "
                f"in {config_path}"
            )
        auxiliary.setdefault(dataset_id, entry)
    else:
        raise ValueError(
            f"Automatic campaign match for {dataset_id!r} has unknown measurement kind "
            f"{match.measurement_kind!r}"
        )
    temporary = config_path.with_name(f".{config_path.name}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(config_path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
