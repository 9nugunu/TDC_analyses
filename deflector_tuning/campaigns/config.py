"""Read compact campaign YAML and validate typed metadata."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Mapping, cast

import yaml

from .models import (
    MarkerCorrectionReference,
    PhaseOffsetSensitivity,
    PhaseReference,
    SimulationReference,
    TuningAuxMeasurement,
    TuningCampaign,
    TuningIssue,
    TuningState,
)
from .validation import _validate_dataset_paths, _validate_relationships


SUPPORTED_SCHEMA_VERSION = 1
MEASUREMENT_STATUSES = frozenset({"pending", "done"})
ISSUE_STATUSES = frozenset({"open", "verify_pending", "closed"})
ISSUE_SEVERITIES = frozenset({"info", "warn", "critical"})
ISSUE_ACTIONS = frozenset({"include", "flag", "exclude"})


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
    marker_correction = _parse_marker_correction(
        campaign_values.get("marker_correction"),
        config_path,
    )
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
        marker_correction=marker_correction,
        states=states,
        auxiliary_measurements=auxiliary_measurements,
        issues=issues,
        source_path=config_path,
    )
    _validate_relationships(campaign)
    if data_root is not None:
        _validate_dataset_paths(campaign, Path(data_root))
    return campaign


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


def _parse_marker_correction(
    value: object,
    config_path: Path,
) -> MarkerCorrectionReference:
    """Parse optional campaign-wide experimental marker-correction inputs."""

    if value is None:
        return MarkerCorrectionReference()
    correction = _mapping(value, "campaign.marker_correction", config_path)
    defaults = MarkerCorrectionReference()
    return MarkerCorrectionReference(
        design_temp_C=_finite_number(
            correction.get("design_temp_C", defaults.design_temp_C),
            "campaign.marker_correction.design_temp_C",
            config_path,
        ),
        humidity_fraction=_finite_number(
            correction.get("humidity_fraction", defaults.humidity_fraction),
            "campaign.marker_correction.humidity_fraction",
            config_path,
        ),
        thermal_alpha_per_C=_finite_number(
            correction.get("thermal_alpha_per_C", defaults.thermal_alpha_per_C),
            "campaign.marker_correction.thermal_alpha_per_C",
            config_path,
        ),
        eps_air_humid=_finite_number(
            correction.get("eps_air_humid", defaults.eps_air_humid),
            "campaign.marker_correction.eps_air_humid",
            config_path,
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
            measurement_temp_C=_optional_finite_number(
                values.get("temp_meas_C"),
                f"states.{state_id}.temp_meas_C",
                config_path,
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


def _finite_number(value: object, name: str, config_path: Path) -> float:
    number = _number(value, name, config_path)
    if not isfinite(number):
        raise ValueError(f"Expected finite numeric {name} in {config_path}")
    return number


def _optional_finite_number(
    value: object,
    name: str,
    config_path: Path,
) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name, config_path)
