"""Match explicit dataset references and infer opted-in tuning states."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import yaml

from .config import load_tuning_campaign
from .models import (
    PhaseOffsetSensitivity,
    TuningAuxMeasurement,
    TuningCampaign,
    TuningCampaignMatch,
    TuningState,
)
from .validation import _validate_dataset_paths


TUNING_DATASET_TOKEN_PATTERN = re.compile(r"(?:^|_)tune(?:_|$)", re.IGNORECASE)
TUNING_TORQUE_PATTERN = re.compile(
    r"(?:^|_)torque(?P<value>\d+(?:p\d+)?)(?:_|$)",
    re.IGNORECASE,
)
TUNING_STATE_PATTERN = re.compile(
    r"(?:^|_)tune_(?P<state_id>s(?P<index>\d+))(?:_|$)",
    re.IGNORECASE,
)
TUNING_AUXILIARY_PATTERN = re.compile(
    r"(?:^|_)tune_(?P<state_id>s(?P<index>\d+))_aux_(?P<kind>[a-z0-9_]+)$",
    re.IGNORECASE,
)
STATE_ID_PATTERN = re.compile(r"^s(?P<index>\d+)$", re.IGNORECASE)


def is_tuning_dataset_id(dataset_id: str) -> bool:
    """Return whether a dataset explicitly opts into tuning via ``_tune_``."""

    return TUNING_DATASET_TOKEN_PATTERN.search(dataset_id) is not None


def tuning_torque_nm_from_dataset_id(dataset_id: str) -> float | None:
    """Parse a readable ``torque13p5`` token without treating it as authority."""

    match = TUNING_TORQUE_PATTERN.search(dataset_id)
    if match is None:
        return None
    return float(match.group("value").lower().replace("p", "."))


def find_tuning_campaign(
    dataset_id: str,
    campaign_dir: str | Path,
    *,
    data_root: str | Path | None = None,
) -> TuningCampaignMatch | None:
    """Find the unique campaign and optional state associated with a dataset."""

    directory = Path(campaign_dir)
    if not directory.is_dir():
        return None
    matches: list[TuningCampaignMatch] = []
    paths = sorted((*directory.glob("*.yaml"), *directory.glob("*.yml")))
    for path in paths:
        if not _campaign_mentions_dataset(path, dataset_id):
            continue
        campaign = load_tuning_campaign(path)
        state_id = next(
            (
                candidate_id
                for candidate_id, state in campaign.states.items()
                if state.dataset == dataset_id
            ),
            None,
        )
        auxiliary_measurement = campaign.auxiliary_measurements.get(dataset_id)
        simulation_match = any(
            reference.dataset == dataset_id
            for reference in campaign.simulation_references.values()
        )
        if state_id is not None:
            measurement_kind = "state"
            comparison_enabled = True
            phase_offset_sensitivity = None
        elif auxiliary_measurement is not None:
            state_id = auxiliary_measurement.state_id
            measurement_kind = "aux"
            comparison_enabled = auxiliary_measurement.comparison_enabled
            phase_offset_sensitivity = auxiliary_measurement.phase_offset_sensitivity
        elif simulation_match:
            measurement_kind = "simulation"
            comparison_enabled = False
            phase_offset_sensitivity = None
        else:
            continue
        if state_id is not None or simulation_match:
            matches.append(
                TuningCampaignMatch(
                    campaign=campaign,
                    config_path=path,
                    state_id=state_id,
                    measurement_kind=measurement_kind,
                    comparison_enabled=comparison_enabled,
                    phase_offset_sensitivity=phase_offset_sensitivity,
                    match_mode="explicit",
                )
            )
    if len(matches) > 1:
        names = ", ".join(str(match.config_path) for match in matches)
        raise ValueError(
            f"Dataset {dataset_id!r} matches multiple tuning campaigns: {names}"
        )
    if matches:
        match = matches[0]
        if data_root is not None:
            _validate_dataset_paths(match.campaign, Path(data_root))
        return match

    inferred = _find_automatic_tuning_campaign(dataset_id, paths)
    if inferred is None:
        return None
    if data_root is not None:
        _validate_dataset_paths(inferred.campaign, Path(data_root))
    return inferred


def _find_automatic_tuning_campaign(
    dataset_id: str,
    paths: tuple[Path, ...],
) -> TuningCampaignMatch | None:
    if not is_tuning_dataset_id(dataset_id):
        return None
    candidates: list[tuple[Path, TuningCampaign]] = []
    for path in paths:
        if not _campaign_auto_states_enabled(path):
            continue
        candidates.append((path, load_tuning_campaign(path)))
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(str(path) for path, _ in candidates)
        raise ValueError(
            f"Dataset {dataset_id!r} matches multiple automatic tuning campaigns: {names}"
        )
    config_path, campaign = candidates[0]
    auxiliary_match = TUNING_AUXILIARY_PATTERN.search(dataset_id)
    if auxiliary_match is not None:
        return _infer_auxiliary_match(
            dataset_id,
            campaign=campaign,
            config_path=config_path,
            state_id=auxiliary_match.group("state_id").lower(),
            kind=auxiliary_match.group("kind").lower(),
        )
    state_match = TUNING_STATE_PATTERN.search(dataset_id)
    if state_match is None:
        return None
    return _infer_primary_state_match(
        dataset_id,
        campaign=campaign,
        config_path=config_path,
        state_id=state_match.group("state_id").lower(),
        state_index=state_match.group("index"),
    )


def _campaign_auto_states_enabled(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        return False
    campaign = raw.get("campaign")
    return isinstance(campaign, Mapping) and campaign.get("auto_states") is True


def _infer_primary_state_match(
    dataset_id: str,
    *,
    campaign: TuningCampaign,
    config_path: Path,
    state_id: str,
    state_index: str,
) -> TuningCampaignMatch:
    if state_id in campaign.states:
        raise ValueError(
            f"Automatic tuning state {state_id!r} in {dataset_id!r} already exists in "
            f"{config_path}; register it explicitly or use an _aux_ dataset name"
        )
    expected_index = _next_numeric_state_index(campaign)
    requested_index = int(state_index)
    if requested_index != expected_index:
        raise ValueError(
            f"Automatic tuning dataset {dataset_id!r} must be next state s{expected_index:0{len(state_index)}d}, "
            f"not {state_id}"
        )
    previous_state_id = f"s{expected_index - 1:0{len(state_index)}d}"
    if previous_state_id not in campaign.states:
        raise ValueError(
            f"Automatic tuning dataset {dataset_id!r} requires previous state "
            f"{previous_state_id!r} in {config_path}"
        )
    state = TuningState(
        state_id=state_id,
        measurement_status="done",
        previous_state_id=previous_state_id,
        dataset=dataset_id,
        torque_nm=tuning_torque_nm_from_dataset_id(dataset_id),
        quality_flag="auto_pending",
    )
    states = dict(campaign.states)
    states[state_id] = state
    inferred_campaign = replace(campaign, states=states)
    return TuningCampaignMatch(
        campaign=inferred_campaign,
        config_path=config_path,
        state_id=state_id,
        measurement_kind="state",
        comparison_enabled=True,
        match_mode="auto",
    )


def _infer_auxiliary_match(
    dataset_id: str,
    *,
    campaign: TuningCampaign,
    config_path: Path,
    state_id: str,
    kind: str,
) -> TuningCampaignMatch:
    if kind != "plunger":
        raise ValueError(
            f"Automatic auxiliary dataset {dataset_id!r} must use _aux_plunger"
        )
    if state_id not in campaign.states:
        raise ValueError(
            f"Automatic auxiliary dataset {dataset_id!r} requires recorded state "
            f"{state_id!r} in {config_path}"
        )
    auxiliary = TuningAuxMeasurement(
        dataset=dataset_id,
        state_id=state_id,
        comparison_enabled=False,
        phase_offset_sensitivity=PhaseOffsetSensitivity(),
    )
    auxiliary_measurements = dict(campaign.auxiliary_measurements)
    auxiliary_measurements[dataset_id] = auxiliary
    inferred_campaign = replace(
        campaign,
        auxiliary_measurements=auxiliary_measurements,
    )
    return TuningCampaignMatch(
        campaign=inferred_campaign,
        config_path=config_path,
        state_id=state_id,
        measurement_kind="aux",
        comparison_enabled=False,
        phase_offset_sensitivity=auxiliary.phase_offset_sensitivity,
        match_mode="auto",
    )


def _next_numeric_state_index(campaign: TuningCampaign) -> int:
    indices = [
        int(match.group("index"))
        for state_id in campaign.states
        if (match := STATE_ID_PATTERN.fullmatch(state_id)) is not None
    ]
    if not indices:
        raise ValueError(
            f"Automatic state registration requires numeric state IDs in {campaign.source_path}"
        )
    return max(indices) + 1


def _campaign_mentions_dataset(path: Path, dataset_id: str) -> bool:
    """Peek only at dataset references before fully validating a campaign."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        return False
    campaign = raw.get("campaign")
    if isinstance(campaign, Mapping):
        if campaign.get("sim") == dataset_id:
            return True
        simulations = campaign.get("sims")
        if isinstance(simulations, Mapping) and any(
            isinstance(reference, Mapping) and reference.get("data") == dataset_id
            for reference in simulations.values()
        ):
            return True
    states = raw.get("states")
    if isinstance(states, Mapping) and any(
        isinstance(state, Mapping) and state.get("data") == dataset_id
        for state in states.values()
    ):
        return True
    auxiliary_measurements = raw.get("aux")
    return isinstance(auxiliary_measurements, Mapping) and dataset_id in auxiliary_measurements
