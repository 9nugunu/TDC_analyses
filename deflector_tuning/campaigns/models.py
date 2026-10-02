"""Immutable metadata records shared by campaign loading and discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from deflector_tuning.markers.frequency_markers import TemperatureHumidityCorrection


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
class MarkerCorrectionReference:
    """Campaign-wide inputs for experimental marker-frequency correction."""

    design_temp_C: float = 20.0
    humidity_fraction: float = 0.65
    thermal_alpha_per_C: float = 1.68e-5
    eps_air_humid: float = 1.000712754221782


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
    measurement_temp_C: float | None = None
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
    marker_correction: MarkerCorrectionReference
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

    def marker_correction_for(
        self,
        state_id: str,
    ) -> TemperatureHumidityCorrection | None:
        """Return the experimental marker correction recorded for one state."""

        measurement_temp_C = self.states[state_id].measurement_temp_C
        if measurement_temp_C is None:
            return None
        return TemperatureHumidityCorrection(
            temp_op_C=self.marker_correction.design_temp_C,
            temp_meas_C=measurement_temp_C,
            humidity_fraction=self.marker_correction.humidity_fraction,
            thermal_alpha_per_C=self.marker_correction.thermal_alpha_per_C,
            eps_air_humid=self.marker_correction.eps_air_humid,
        )


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
