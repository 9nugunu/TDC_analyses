from pathlib import Path

import pandas as pd

import deflector_tuning.runner as runner
from runner_helpers import (
    _tables,
    _cell_iris_response_comparison_table,
    _coupler_cavity_parameter_estimates_table,
)


def test_detect_analysis_modes_skips_sim_260526_grid_scan_for_experiment_marker_points() -> None:
    tables = _tables()
    tables["marker_pts"] = tables["marker_pts"].assign(data_kind="experiment")

    detected = runner.detect_analysis_modes(tables)

    assert "marker_analysis" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_uses_dataset_category_as_grid_gate() -> None:
    assert "grid_scan_spacing" in runner.detect_analysis_modes(_tables(), dataset_category="grid")
    assert "grid_scan_spacing" not in runner.detect_analysis_modes(_tables(), dataset_category="sweep")


def test_detect_analysis_modes_enables_geometry_phase_response_when_table_has_rows() -> None:
    tables = _tables()
    tables["geom_phase"] = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "sweep_axis": "sim_offset_cell_03",
                "sweep_value": 0.0,
                "phase_delta_shift_from_baseline_deg": 0.0,
            }
        ]
    )

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "geometry_phase_response" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_enables_cell_iris_response_when_comparison_table_has_rows() -> None:
    tables = _tables()
    tables["cell_iris_cmp"] = _cell_iris_response_comparison_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "cell_iris_response" in detected
    assert "grid_scan_spacing" not in detected


def test_detect_analysis_modes_enables_coupler_cavity_parameters_when_table_has_rows() -> None:
    tables = _tables()
    tables["coupler_params"] = _coupler_cavity_parameter_estimates_table()

    detected = runner.detect_analysis_modes(tables, dataset_category="sweep")

    assert "coupler_cavity_parameters" in detected
    assert "grid_scan_spacing" not in detected


def test_resolve_input_paths_uses_data_root_and_default_dispersion(
    tmp_path: Path,
) -> None:
    sparameter_path, dispersion_path = runner.resolve_input_paths(
        "prepro/prepro_sweep_260415_sample_prepro",
        data_root=tmp_path / "data",
    )

    assert sparameter_path == tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    assert dispersion_path == tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"


def test_resolve_input_paths_accepts_configured_default_dispersion(tmp_path: Path) -> None:
    _, dispersion_path = runner.resolve_input_paths(
        "sim/sim_sweep_260519_scan_dataset",
        data_root=tmp_path / "data",
        default_dispersion_subpath=Path("sim") / "custom_dispersion",
    )

    assert dispersion_path == tmp_path / "data" / "sim" / "custom_dispersion"
