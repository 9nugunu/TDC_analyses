from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.visualization.coupler_cavity_parameter_plots import (
    plot_coupler_cavity_parameters,
)


def _parameter_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file": "run_1.s1p",
                "tune_position": 0.5,
                "coupler_transition_pair": "0.5_to_1.5",
                "coupler_position_basis": "cell_center",
                "coupler_pair_start_tune_position": 0.5,
                "coupler_pair_end_tune_position": 1.5,
                "coupler_frequency_ghz": 2.860,
                "matching_frequency_ghz": 2.856,
                "delta_frequency_mhz": 4.0,
                "external_quality_factor": 85.0,
                "target_external_quality_factor": 92.0,
                "coupling_k": 0.025,
                "coupling_k_source": "marker_frequency_ratio_abs",
                "coupling_beta": 0.92,
                "coupling_beta_status": "ok",
                "operation_mode_deg": 120.0,
                "is_valid": True,
            },
            {
                "source_file": "run_2.s1p",
                "tune_position": 1.5,
                "coupler_transition_pair": "0.5_to_1.5",
                "coupler_position_basis": "cell_center",
                "coupler_pair_start_tune_position": 0.5,
                "coupler_pair_end_tune_position": 1.5,
                "coupler_frequency_ghz": 2.854,
                "matching_frequency_ghz": 2.856,
                "delta_frequency_mhz": -2.0,
                "external_quality_factor": 98.0,
                "target_external_quality_factor": 92.0,
                "coupling_k": 0.025,
                "coupling_k_source": "marker_frequency_ratio_abs",
                "coupling_beta": 1.05,
                "coupling_beta_status": "ok",
                "operation_mode_deg": 120.0,
                "is_valid": True,
            },
        ]
    )


def _radius_sweep_parameter_table() -> pd.DataFrame:
    table = _parameter_table().copy()
    table["sim_r_c"] = [54.0, 55.0]
    table["sim_w_c"] = [19.3224, 19.3224]
    table["coupler_position_basis"] = "geometry_sweep"
    return table


def test_plot_coupler_cavity_parameters_writes_named_outputs(tmp_path: Path) -> None:
    paths = plot_coupler_cavity_parameters(_parameter_table(), tmp_path)

    assert list(paths) == [
        "coupler_frequency_shift",
        "external_quality_factor",
        "coupling_beta",
    ]
    assert [path.name for path in paths.values()] == [
        "coupler_frequency_shift.png",
        "external_quality_factor.png",
        "coupling_beta.png",
    ]
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_coupler_cavity_parameters_writes_beta_vs_radius_for_geometry_sweep(tmp_path: Path) -> None:
    paths = plot_coupler_cavity_parameters(_radius_sweep_parameter_table(), tmp_path)

    assert "coupler_beta_vs_coupler_radius" in paths
    assert paths["coupler_beta_vs_coupler_radius"].name == "coupler_beta_vs_coupler_radius.png"
    assert paths["coupler_beta_vs_coupler_radius"].exists()
    assert paths["coupler_beta_vs_coupler_radius"].stat().st_size > 0


def test_plot_coupler_cavity_parameters_rejects_missing_columns(tmp_path: Path) -> None:
    table = _parameter_table().drop(columns=["delta_frequency_mhz"])

    with pytest.raises(ValueError, match="coupler_cavity_parameter_estimates is missing required columns"):
        plot_coupler_cavity_parameters(table, tmp_path)
