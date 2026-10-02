"""Check directed phase, exported-sample selection, and pair compatibility."""

import numpy as np
import pytest

from scripts.build_paper_cst_bundle import (
    nearest_sample_index,
    phase_pair_metrics,
    radius_phase_response,
    validate_pair_metadata,
)


@pytest.mark.parametrize(
    "first_deg,second_deg,target_deg,delta_deg,residual_deg",
    [(170, -70, 120, 120, 0), (-179, 179, 0, 358, -2),
     (0, -120, 240, 240, 0), (0, 180, 0, 180, -180),
     (30, 29, 120, 359, -121)],
)
def test_directed_phase_and_target_branch_wrap(
    first_deg, second_deg, target_deg, delta_deg, residual_deg
):
    first = 0.8 * np.exp(1j * np.deg2rad(first_deg))
    second = 1.1 * np.exp(1j * np.deg2rad(second_deg))
    result = phase_pair_metrics(first, second, target_deg)
    assert result["delta_deg"] == pytest.approx(delta_deg)
    assert result["signed_target_residual_deg"] == pytest.approx(residual_deg)
    assert result["first_modulus"] == pytest.approx(0.8)
    assert result["second_modulus"] == pytest.approx(1.1)


@pytest.mark.parametrize("first,second", [(0j, 1j), (1j, 0j), (complex(np.nan, 0), 1j)])
def test_undefined_phase_is_rejected(first, second):
    with pytest.raises(ValueError, match="finite and nonzero"):
        phase_pair_metrics(first, second, 180)


def test_nearest_exported_sample_and_deterministic_tie():
    assert nearest_sample_index(np.array([2.85, 2.86, 2.87]), 2.8571160335532) == 1
    assert nearest_sample_index(np.array([1.0, 3.0]), 2.0) == 0


@pytest.mark.parametrize("frequencies,target", [([2, 1], 1.5), ([1, 2], 3), ([], 1), ([1, np.nan], 1)])
def test_invalid_frequency_grid_or_out_of_range_marker_is_rejected(frequencies, target):
    with pytest.raises(ValueError):
        nearest_sample_index(np.array(frequencies), target)


def _metadata(depth, plunger):
    return {"project": "case.cst", "option_line": "# GHz S RI R 0",
            "port_assignments": ["! Touchstone port 1 = CST MWS port 2 (\"\")"],
            "parameters": {"NumDepth": depth, "DepthPlunger": plunger,
                           "R_plunger": 8, "NumCell": 9, "Nmesh": 180}}


def test_pair_metadata_allows_only_declared_short_state_changes():
    result = validate_pair_metadata(_metadata(0.5, 100), _metadata(1.5, 65),
                                    ["NumDepth", "DepthPlunger"])
    assert result["consistent"] is True
    assert set(result["varying_short_state_fields"]) == {"NumDepth", "DepthPlunger"}


@pytest.mark.parametrize("changed_field", ["R_plunger", "Nmesh", "new_geometry_parameter"])
def test_pair_metadata_exposes_geometry_mismatch(changed_field):
    second = _metadata(1.5, 65)
    second["parameters"][changed_field] = 999
    with pytest.raises(ValueError, match=changed_field):
        validate_pair_metadata(_metadata(0.5, 100), second, ["NumDepth", "DepthPlunger"])


def test_pair_metadata_exposes_changed_port_assignment():
    second = _metadata(1.5, 65)
    second["port_assignments"] = ["! Touchstone port 1 = CST MWS port 1 (\"\")"]
    with pytest.raises(ValueError, match="port_assignments"):
        validate_pair_metadata(_metadata(0.5, 100), second, ["NumDepth", "DepthPlunger"])


def test_radius_unwrap_preserves_nominal_raw_phase_and_centered_slope():
    result = radius_phase_response(np.array([56.58, 56.59, 56.60]),
                                   np.array([170.0, 179.0, -172.0]), 56.59)
    assert result["unwrapped_phase_deg"] == pytest.approx([170, 179, 188])
    assert result["nominal_central_slope_deg_per_mm"] == pytest.approx(900)
    assert result["branch_turns"] == [0, 0, 1]


def test_radius_unwrap_can_anchor_negative_nominal_branch():
    result = radius_phase_response(np.array([1, 2, 3]), np.array([179, -172, -163]), 2)
    assert result["unwrapped_phase_deg"] == pytest.approx([-181, -172, -163])
    assert result["nominal_central_slope_deg_per_mm"] == pytest.approx(9)


def test_radius_requires_a_centered_nominal_point():
    with pytest.raises(ValueError, match="interior"):
        radius_phase_response(np.array([1, 2, 3]), np.array([0, 5, 10]), 1)


def test_nominal_geometry_link_can_record_different_project_names_explicitly():
    first, second = _metadata(2, 37.908), _metadata(2, 37.908)
    second["project"] = "radius_scan.cst"
    result = validate_pair_metadata(first, second, [], require_same_project=False)
    assert result["consistent"]
    assert result["different_project_names"] == ["case.cst", "radius_scan.cst"]
