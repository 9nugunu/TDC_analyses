from __future__ import annotations

from deflector_tuning.table_schema import (
    FORBIDDEN_LEGACY_COLUMNS,
    SOURCE_COLUMN_ALIASES,
    STANDARD_TABLE_SPECS,
    TABLE_CONTRACTS,
    canonical_source_column,
)


def test_standard_table_specs_define_canonical_and_legacy_filenames() -> None:
    assert list(STANDARD_TABLE_SPECS) == [
        "markers",
        "marker_pts",
        "phase_polar",
        "kyhl_admit_audit",
        "kyhl_admit_pts",
        "kyhl_admit_steps",
        "cell_iris_cmp",
        "coupler_params",
        "rc_line",
        "phase_adv",
        "phase_stats",
        "nodal_shift",
        "geom_phase",
    ]
    assert STANDARD_TABLE_SPECS["marker_pts"].filename == "marker_pts.csv"
    assert STANDARD_TABLE_SPECS["marker_pts"].legacy_filename == "marker_points.csv"
    assert STANDARD_TABLE_SPECS["markers"].legacy_filename == "markers.csv"


def test_direct_matrix_contracts_use_compact_selected_names() -> None:
    assert list(TABLE_CONTRACTS["y11"]) == ["markers", "y11_pts"]
    assert TABLE_CONTRACTS["y11"]["y11_pts"].filename == "y11_pts.csv"
    assert (
        TABLE_CONTRACTS["y11"]["y11_pts"].legacy_filename
        == "y11_marker_points.csv"
    )
    assert list(TABLE_CONTRACTS["z11"]) == ["markers", "z11_pts"]
    assert TABLE_CONTRACTS["z11"]["z11_pts"].filename == "z11_pts.csv"
    assert TABLE_CONTRACTS["z11"]["z11_pts"].legacy_filename is None


def test_source_aliases_are_global_column_rules() -> None:
    assert SOURCE_COLUMN_ALIASES == {
        "sim_tuner_insertion_depth": "sim_tuner_depth",
        "sim_coupler_path_bot2_width": "sim_cpl_bot2_w",
    }
    assert canonical_source_column("sim_tuner_insertion_depth") == "sim_tuner_depth"
    assert canonical_source_column("sim_r_c") == "sim_r_c"


def test_forbidden_legacy_columns_include_duplicate_kyhl_aliases() -> None:
    assert "admittance_real" in FORBIDDEN_LEGACY_COLUMNS
    assert "kyhl_operation_real" in FORBIDDEN_LEGACY_COLUMNS
    assert "phase_advance_0to360_deg" in FORBIDDEN_LEGACY_COLUMNS
    assert "y_real_siemens" in FORBIDDEN_LEGACY_COLUMNS
    assert "y_imag_siemens" in FORBIDDEN_LEGACY_COLUMNS
    assert "target_freq_ghz" in FORBIDDEN_LEGACY_COLUMNS
