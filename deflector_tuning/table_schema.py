"""Canonical names for standard marker-analysis tables and columns."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class TableNameSpec:
    """Canonical and former filenames for one managed analysis table."""

    filename: str
    legacy_filename: str | None


STANDARD_TABLE_SPECS: OrderedDict[str, TableNameSpec] = OrderedDict(
    [
        ("markers", TableNameSpec("markers.csv", "markers.csv")),
        ("marker_pts", TableNameSpec("marker_pts.csv", "marker_points.csv")),
        ("phase_polar", TableNameSpec("phase_polar.csv", "marker_phase_polar.csv")),
        (
            "kyhl_admit_audit",
            TableNameSpec(
                "kyhl_admit_audit.csv",
                "kyhl_f2pi3_normalized_admittance_audit.csv",
            ),
        ),
        (
            "kyhl_admit_pts",
            TableNameSpec("kyhl_admit_pts.csv", "kyhl_admittance_points.csv"),
        ),
        (
            "kyhl_admit_steps",
            TableNameSpec("kyhl_admit_steps.csv", "kyhl_admittance_transitions.csv"),
        ),
        (
            "cell_iris_cmp",
            TableNameSpec("cell_iris_cmp.csv", "cell_iris_response_comparison.csv"),
        ),
        (
            "coupler_params",
            TableNameSpec("coupler_params.csv", "coupler_cavity_parameter_estimates.csv"),
        ),
        ("rc_line", TableNameSpec("rc_line.csv", "grid_rc_line_scan.csv")),
        ("phase_adv", TableNameSpec("phase_adv.csv", "phase_advance.csv")),
        ("phase_stats", TableNameSpec("phase_stats.csv", "phase_summary.csv")),
        ("nodal_shift", TableNameSpec("nodal_shift.csv", "nodal_shift.csv")),
        (
            "geom_phase",
            TableNameSpec("geom_phase.csv", "geometry_phase_response.csv"),
        ),
    ]
)

Y11_TABLE_SPECS: OrderedDict[str, TableNameSpec] = OrderedDict(
    [
        ("markers", TableNameSpec("markers.csv", "markers.csv")),
        ("y11_pts", TableNameSpec("y11_pts.csv", "y11_marker_points.csv")),
    ]
)

Z11_TABLE_SPECS: OrderedDict[str, TableNameSpec] = OrderedDict(
    [
        ("markers", TableNameSpec("markers.csv", "markers.csv")),
        ("z11_pts", TableNameSpec("z11_pts.csv", None)),
    ]
)

TABLE_CONTRACTS: dict[str, OrderedDict[str, TableNameSpec]] = {
    "standard": STANDARD_TABLE_SPECS,
    "y11": Y11_TABLE_SPECS,
    "z11": Z11_TABLE_SPECS,
}

SOURCE_COLUMN_ALIASES: dict[str, str] = {
    "sim_tuner_insertion_depth": "sim_tuner_depth",
    "sim_coupler_path_bot2_width": "sim_cpl_bot2_w",
}

FORBIDDEN_LEGACY_COLUMNS: frozenset[str] = frozenset(
    {
        "admittance_real",
        "admittance_imag",
        "kyhl_operation_real",
        "kyhl_operation_imag",
        "kyhl_operation_angle_deg",
        "phase_advance_0to360_deg",
        "phase_error_from_target_deg",
        "abs_phase_error_from_target_deg",
        "y_real_siemens",
        "y_imag_siemens",
        "target_freq_ghz",
    }
)


def table_specs(contract_name: str) -> OrderedDict[str, TableNameSpec]:
    """Return the ordered table specification for a named output contract."""

    try:
        return TABLE_CONTRACTS[contract_name]
    except KeyError as exc:
        raise KeyError(f"Unknown table contract: {contract_name}") from exc


def canonical_source_column(name: str) -> str:
    """Return the global canonical name for one propagated source column."""

    return SOURCE_COLUMN_ALIASES.get(name, name)
