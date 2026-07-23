"""Compact YAML metadata for multi-state mechanical tuning campaigns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, cast

import yaml


SUPPORTED_SCHEMA_VERSION = 1
MEASUREMENT_STATUSES = frozenset({"pending", "done"})
ISSUE_STATUSES = frozenset({"open", "verify_pending", "closed"})
ISSUE_SEVERITIES = frozenset({"info", "warn", "critical"})
ISSUE_ACTIONS = frozenset({"include", "flag", "exclude"})
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


@dataclass(frozen=True)
class PhaseReference:
    """Phase-reference provenance recorded for one campaign."""

    port_extension: str | None = None
    extension_mm: float | None = None
    verified: bool = False
    anchor_marker: str = "f_mean"
    anchor_phase_deg: float = 180.0
    anchor_tolerance_deg: float = 5.0


@dataclass(frozen=True)
class SimulationReference:
    """One measurement-family reference to a one-dimensional ``r_c`` sweep."""

    family: str
    dataset: str
    experiment_positions: tuple[float, ...]
    axis: str = "r_c"


@dataclass(frozen=True)
class TuningState:
    """One ordered mechanical configuration and its measurement reference."""

    state_id: str
    measurement_status: str
    role: str | None = None
    previous_state_id: str | None = None
    dataset: str | None = None
    torque_nm: float | None = None
    angle_deg: float | None = None
    change: str | None = None
    quality_flag: str | None = None
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class PhaseOffsetSensitivity:
    """Requested phase response versus a measured plunger offset."""

    reference_offset_mm: float = 0.0


@dataclass(frozen=True)
class TuningAuxMeasurement:
    """A measurement attached to an existing state, not a new state."""

    dataset: str
    state_id: str
    comparison_enabled: bool = False
    phase_offset_sensitivity: PhaseOffsetSensitivity | None = None


@dataclass(frozen=True)
class TuningIssue:
    """A sparse experimental issue linked to discovery and verification states."""

    issue_id: str
    after_state_id: str
    tag: str
    verification_state_id: str | None = None
    check: tuple[str, ...] = ()
    status: str = "open"
    severity: str = "warn"
    action: str = "flag"
    note: str | None = None


@dataclass(frozen=True)
class TuningCampaign:
    """Validated campaign metadata loaded from one YAML source file."""

    schema_version: int
    campaign_id: str
    auto_states_enabled: bool
    simulation_references: Mapping[str, SimulationReference]
    design_r_c_mm: float
    phase: PhaseReference
    states: Mapping[str, TuningState]
    auxiliary_measurements: Mapping[str, TuningAuxMeasurement]
    issues: Mapping[str, TuningIssue]
    source_path: Path

    @property
    def baseline_state_id(self) -> str:
        """Return the unique RF baseline state ID guaranteed by validation."""

        return next(
            state_id
            for state_id, state in self.states.items()
            if state.role == "baseline"
        )

    @property
    def simulation_dataset(self) -> str:
        """Return the first simulation dataset for legacy single-sweep callers."""

        return next(iter(self.simulation_references.values())).dataset


@dataclass(frozen=True)
class TuningCampaignMatch:
    """One unambiguous campaign match for a dataset ID."""

    campaign: TuningCampaign
    config_path: Path
    state_id: str | None
    measurement_kind: str
    comparison_enabled: bool
    phase_offset_sensitivity: PhaseOffsetSensitivity | None = None
    match_mode: str = "explicit"


def is_tuning_dataset_id(dataset_id: str) -> bool:
    """Return whether a dataset explicitly opts into tuning via ``_tune_``."""

    return TUNING_DATASET_TOKEN_PATTERN.search(dataset_id) is not None


def tuning_torque_nm_from_dataset_id(dataset_id: str) -> float | None:
    """Parse a readable ``torque13p5`` token without treating it as authority."""

    match = TUNING_TORQUE_PATTERN.search(dataset_id)
    if match is None:
        return None
    return float(match.group("value").lower().replace("p", "."))


def load_tuning_campaign(
    path: str | Path, *, data_root: str | Path | None = None
) -> TuningCampaign:
    """Load and validate one compact tuning-campaign YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "YAML root", config_path)
    schema = _integer(root.get("schema"), "schema", config_path)
    if schema != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported tuning campaign schema {schema!r} in {config_path}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    campaign_values = _mapping(root.get("campaign"), "campaign", config_path)
    campaign_id = _text(campaign_values.get("id"), "campaign.id", config_path)
    auto_states_enabled = campaign_values.get("auto_states", False)
    if not isinstance(auto_states_enabled, bool):
        raise ValueError(
            f"Expected boolean campaign.auto_states in {config_path}"
        )
    simulation_references = _parse_simulation_references(
        campaign_values,
        config_path,
    )
    design_r_c_mm = _number(
        campaign_values.get("rc_design_mm"), "campaign.rc_design_mm", config_path
    )
    phase = _parse_phase(campaign_values.get("phase"), config_path)
    states = _parse_states(root.get("states"), config_path)
    auxiliary_measurements = _parse_auxiliary_measurements(root.get("aux", {}), config_path)
    issues = _parse_issues(root.get("issues", {}), config_path)

    campaign = TuningCampaign(
        schema_version=schema,
        campaign_id=campaign_id,
        auto_states_enabled=auto_states_enabled,
        simulation_references=simulation_references,
        design_r_c_mm=design_r_c_mm,
        phase=phase,
        states=states,
        auxiliary_measurements=auxiliary_measurements,
        issues=issues,
        source_path=config_path,
    )
    _validate_relationships(campaign)
    if data_root is not None:
        _validate_dataset_paths(campaign, Path(data_root))
    return campaign


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


def _parse_phase(value: object, config_path: Path) -> PhaseReference:
    if value is None:
        return PhaseReference()
    phase = _mapping(value, "campaign.phase", config_path)
    verified = phase.get("verified", False)
    if not isinstance(verified, bool):
        raise ValueError(
            f"Expected boolean campaign.phase.verified in {config_path}"
        )
    extension = phase.get("ext_mm")
    anchor_phase_deg = _optional_number(
        phase.get("anchor_deg"),
        "campaign.phase.anchor_deg",
        config_path,
    )
    anchor_tolerance_deg = _optional_number(
        phase.get("tol_deg"),
        "campaign.phase.tol_deg",
        config_path,
    )
    return PhaseReference(
        port_extension=_optional_text(
            phase.get("port_ext"), "campaign.phase.port_ext", config_path
        ),
        extension_mm=(
            None
            if extension is None
            else _number(extension, "campaign.phase.ext_mm", config_path)
        ),
        verified=verified,
        anchor_marker=_optional_text(
            phase.get("anchor_marker"),
            "campaign.phase.anchor_marker",
            config_path,
        )
        or "f_mean",
        anchor_phase_deg=(
            180.0 if anchor_phase_deg is None else anchor_phase_deg
        ),
        anchor_tolerance_deg=(
            5.0 if anchor_tolerance_deg is None else anchor_tolerance_deg
        ),
    )


def _parse_simulation_references(
    campaign_values: Mapping[object, object],
    config_path: Path,
) -> dict[str, SimulationReference]:
    legacy = campaign_values.get("sim")
    raw_references = campaign_values.get("sims")
    if legacy is not None and raw_references is not None:
        raise ValueError(f"Use only one of campaign.sim or campaign.sims in {config_path}")
    if raw_references is None:
        dataset = _text(legacy, "campaign.sim", config_path)
        return {
            "iris": SimulationReference(
                family="iris",
                dataset=dataset,
                experiment_positions=(1.0, 2.0),
            )
        }

    references = _mapping(raw_references, "campaign.sims", config_path)
    if not references:
        raise ValueError(f"Expected at least one campaign.sims entry in {config_path}")
    parsed: dict[str, SimulationReference] = {}
    for raw_family, raw_reference in references.items():
        family = _mapping_key(raw_family, "campaign.sims", config_path)
        reference = _mapping(
            raw_reference,
            f"campaign.sims.{family}",
            config_path,
        )
        axis = _text(
            reference.get("axis", "r_c"),
            f"campaign.sims.{family}.axis",
            config_path,
        )
        if axis != "r_c":
            raise ValueError(
                f"Expected campaign.sims.{family}.axis to be r_c in {config_path}"
            )
        raw_positions = reference.get("exp_pos")
        if not isinstance(raw_positions, list) or len(raw_positions) < 2:
            raise ValueError(
                f"Expected campaign.sims.{family}.exp_pos to contain at least two positions "
                f"in {config_path}"
            )
        positions = tuple(
            _number(
                position,
                f"campaign.sims.{family}.exp_pos",
                config_path,
            )
            for position in raw_positions
        )
        parsed[family] = SimulationReference(
            family=family,
            dataset=_text(
                reference.get("data"),
                f"campaign.sims.{family}.data",
                config_path,
            ),
            experiment_positions=positions,
            axis=axis,
        )
    return parsed


def _parse_states(value: object, config_path: Path) -> dict[str, TuningState]:
    raw_states = _mapping(value, "states", config_path)
    if not raw_states:
        raise ValueError(f"Expected at least one state in {config_path}")
    states: dict[str, TuningState] = {}
    for raw_state_id, raw_values in raw_states.items():
        state_id = _mapping_key(raw_state_id, "state", config_path)
        values = _mapping(raw_values, f"states.{state_id}", config_path)
        status = _text(values.get("meas"), f"states.{state_id}.meas", config_path)
        if status not in MEASUREMENT_STATUSES:
            raise ValueError(
                f"Expected states.{state_id}.meas to be pending or done in {config_path}"
            )
        dataset = _optional_text(
            values.get("data"), f"states.{state_id}.data", config_path
        )
        if status == "done" and dataset is None:
            raise ValueError(
                f"Completed state {state_id} requires data in {config_path}"
            )
        uncertainty_values = values.get("unc", {})
        uncertainty_mapping = _mapping(
            uncertainty_values, f"states.{state_id}.unc", config_path
        )
        uncertainty = {
            _mapping_key(name, f"states.{state_id}.unc", config_path): _number(
                amount, f"states.{state_id}.unc.{name}", config_path
            )
            for name, amount in uncertainty_mapping.items()
        }
        states[state_id] = TuningState(
            state_id=state_id,
            measurement_status=status,
            role=_optional_text(
                values.get("role"), f"states.{state_id}.role", config_path
            ),
            previous_state_id=_optional_text(
                values.get("prev"), f"states.{state_id}.prev", config_path
            ),
            dataset=dataset,
            torque_nm=_optional_number(
                values.get("torque_nm"), f"states.{state_id}.torque_nm", config_path
            ),
            angle_deg=_optional_number(
                values.get("ang_deg"), f"states.{state_id}.ang_deg", config_path
            ),
            change=_optional_text(
                values.get("change"), f"states.{state_id}.change", config_path
            ),
            quality_flag=_optional_text(
                values.get("flag"), f"states.{state_id}.flag", config_path
            ),
            uncertainty=uncertainty,
            note=_optional_text(
                values.get("note"), f"states.{state_id}.note", config_path
            ),
        )
    return states


def _parse_auxiliary_measurements(
    value: object,
    config_path: Path,
) -> dict[str, TuningAuxMeasurement]:
    raw_measurements = _mapping(value, "aux", config_path)
    measurements: dict[str, TuningAuxMeasurement] = {}
    for raw_dataset, raw_values in raw_measurements.items():
        dataset = _mapping_key(raw_dataset, "aux", config_path)
        values = _mapping(raw_values, f"aux.{dataset}", config_path)
        comparison_enabled = values.get("cmp", False)
        if not isinstance(comparison_enabled, bool):
            raise ValueError(f"Expected boolean aux.{dataset}.cmp in {config_path}")
        sensitivity_values = values.get("sens")
        sensitivity = None
        if sensitivity_values is not None:
            sensitivity_mapping = _mapping(
                sensitivity_values,
                f"aux.{dataset}.sens",
                config_path,
            )
            axis = _text(
                sensitivity_mapping.get("axis"),
                f"aux.{dataset}.sens.axis",
                config_path,
            )
            if axis != "plunger_offset_mm":
                raise ValueError(
                    f"Expected aux.{dataset}.sens.axis to be plunger_offset_mm "
                    f"in {config_path}"
                )
            sensitivity = PhaseOffsetSensitivity(
                reference_offset_mm=_number(
                    sensitivity_mapping.get("ref_mm", 0.0),
                    f"aux.{dataset}.sens.ref_mm",
                    config_path,
                )
            )
        measurements[dataset] = TuningAuxMeasurement(
            dataset=dataset,
            state_id=_text(values.get("state"), f"aux.{dataset}.state", config_path),
            comparison_enabled=comparison_enabled,
            phase_offset_sensitivity=sensitivity,
        )
    return measurements


def _parse_issues(value: object, config_path: Path) -> dict[str, TuningIssue]:
    raw_issues = _mapping(value, "issues", config_path)
    issues: dict[str, TuningIssue] = {}
    for raw_issue_id, raw_values in raw_issues.items():
        issue_id = _mapping_key(raw_issue_id, "issue", config_path)
        values = _mapping(raw_values, f"issues.{issue_id}", config_path)
        check_value = values.get("check", [])
        if not isinstance(check_value, list) or not all(
            isinstance(item, str) and item.strip() for item in check_value
        ):
            raise ValueError(
                f"Expected issues.{issue_id}.check to be a text list in {config_path}"
            )
        status = _text(
            values.get("stat", "open"), f"issues.{issue_id}.stat", config_path
        )
        severity = _text(
            values.get("sev", "warn"), f"issues.{issue_id}.sev", config_path
        )
        action = _text(
            values.get("action", "flag"), f"issues.{issue_id}.action", config_path
        )
        if status not in ISSUE_STATUSES:
            raise ValueError(f"Unknown issue status {status!r} in {config_path}")
        if severity not in ISSUE_SEVERITIES:
            raise ValueError(f"Unknown issue severity {severity!r} in {config_path}")
        if action not in ISSUE_ACTIONS:
            raise ValueError(f"Unknown issue action {action!r} in {config_path}")
        issues[issue_id] = TuningIssue(
            issue_id=issue_id,
            after_state_id=_text(
                values.get("after"), f"issues.{issue_id}.after", config_path
            ),
            verification_state_id=_optional_text(
                values.get("verify"), f"issues.{issue_id}.verify", config_path
            ),
            tag=_text(values.get("tag"), f"issues.{issue_id}.tag", config_path),
            check=tuple(item.strip() for item in cast(list[str], check_value)),
            status=status,
            severity=severity,
            action=action,
            note=_optional_text(
                values.get("note"), f"issues.{issue_id}.note", config_path
            ),
        )
    return issues


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


def _mapping(value: object, name: str, config_path: Path) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {name} mapping in {config_path}")
    return cast(Mapping[object, object], value)


def _mapping_key(value: object, name: str, config_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty text {name} ID in {config_path}")
    return value.strip()


def _text(value: object, name: str, config_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty text {name} in {config_path}")
    return value.strip()


def _optional_text(value: object, name: str, config_path: Path) -> str | None:
    if value is None:
        return None
    return _text(value, name, config_path)


def _integer(value: object, name: str, config_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer {name} in {config_path}")
    return value


def _number(value: object, name: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {name} in {config_path}")
    return float(value)


def _optional_number(value: object, name: str, config_path: Path) -> float | None:
    if value is None:
        return None
    return _number(value, name, config_path)
