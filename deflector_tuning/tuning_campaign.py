"""Public compatibility facade for compact tuning-campaign metadata.

Metadata models, YAML parsing/validation, and dataset discovery live in the
``campaigns`` package. These imports retain the original public import paths and
refer to the same class and function objects as their implementation modules.
"""

from .campaigns.config import (
    SUPPORTED_SCHEMA_VERSION,
    MEASUREMENT_STATUSES,
    ISSUE_STATUSES,
    ISSUE_SEVERITIES,
    ISSUE_ACTIONS,
    load_tuning_campaign,
)
from .campaigns.discovery import (
    TUNING_DATASET_TOKEN_PATTERN,
    TUNING_TORQUE_PATTERN,
    TUNING_STATE_PATTERN,
    TUNING_AUXILIARY_PATTERN,
    STATE_ID_PATTERN,
    find_tuning_campaign,
    is_tuning_dataset_id,
    tuning_torque_nm_from_dataset_id,
)
from .campaigns.models import (
    PhaseReference,
    MarkerCorrectionReference,
    SimulationReference,
    TuningState,
    PhaseOffsetSensitivity,
    TuningAuxMeasurement,
    TuningIssue,
    TuningCampaign,
    TuningCampaignMatch,
)


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "MEASUREMENT_STATUSES",
    "ISSUE_STATUSES",
    "ISSUE_SEVERITIES",
    "ISSUE_ACTIONS",
    "TUNING_DATASET_TOKEN_PATTERN",
    "TUNING_TORQUE_PATTERN",
    "TUNING_STATE_PATTERN",
    "TUNING_AUXILIARY_PATTERN",
    "STATE_ID_PATTERN",
    "PhaseReference",
    "MarkerCorrectionReference",
    "SimulationReference",
    "TuningState",
    "PhaseOffsetSensitivity",
    "TuningAuxMeasurement",
    "TuningIssue",
    "TuningCampaign",
    "TuningCampaignMatch",
    "load_tuning_campaign",
    "find_tuning_campaign",
    "is_tuning_dataset_id",
    "tuning_torque_nm_from_dataset_id",
]
