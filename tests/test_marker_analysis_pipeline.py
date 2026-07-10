from pathlib import Path

import pandas as pd

import deflector_tuning.analysis.marker_pipeline as marker_pipeline
from deflector_tuning.analysis.marker_pipeline import (
    build_marker_analysis,
    build_marker_phase_polar_table,
    save_marker_analysis,
)


def _write_dispersion_summary(folder: Path) -> None:
    processed = folder / "processed"
    processed.mkdir(parents=True)
    (processed / "dispersion_summary.csv").write_text(
        "mode_index,freq_90_GHz,freq_120_GHz\n1,2.8778045932946,2.8574021866048\n",
        encoding="utf-8",
    )


def _write_prepro_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    for tune_position, phases in [
        (0.5, [10.0, 20.0, 30.0]),
        (1.5, [-110.0, -100.0, -90.0]),
    ]:
        filename = folder / f"{tune_position}_processed.csv"
        filename.write_text(
            "freq[Hz],Magnitude,Phase_deg\n"
            f"2855880000,-1.0,{phases[0]}\n"
            f"2866050000,-2.0,{phases[1]}\n"
            f"2876210000,-3.0,{phases[2]}\n",
            encoding="utf-8",
        )


def _write_grid_sim_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\t"w_c"\n"1"\t"54.59"\t"18.3224"\n"2"\t"55.59"\t"19.3224"\n',
        encoding="utf-8",
    )
    for run_id, phases in [(1, [10.0, 20.0, 30.0]), (2, [-110.0, -100.0, -90.0])]:
        (folder / f"run_{run_id}.s1p").write_text(
            f"# GHz S DB R 50\n2.85588 -1.0 {phases[0]}\n2.86605 -2.0 {phases[1]}\n2.87621 -3.0 {phases[2]}\n",
            encoding="utf-8",
        )


def _write_rc_line_grid_sim_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\t"w_c"\n"1"\t"55.59"\t"19.3224"\n"2"\t"54.59"\t"19.3224"\n"3"\t"56.59"\t"18.3224"\n',
        encoding="utf-8",
    )
    for run_id, phases in [
        (1, [40.0, 50.0, 60.0]),
        (2, [10.0, 20.0, 30.0]),
        (3, [70.0, 80.0, 90.0]),
    ]:
        (folder / f"run_{run_id}.s1p").write_text(
            f"# GHz S DB R 50\n2.85588 -1.0 {phases[0]}\n2.86605 -2.0 {phases[1]}\n2.87621 -3.0 {phases[2]}\n",
            encoding="utf-8",
        )


def test_build_marker_analysis_processes_one_folder_into_marker_phase_tables(
    tmp_path: Path,
) -> None:
    sparameter_path = tmp_path / "data" / "prepro" / "prepro_sweep_260415_sample_prepro"
    dispersion_path = tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"
    _write_prepro_dataset(sparameter_path)
    _write_dispersion_summary(dispersion_path)

    result = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role="exp",
    )

    assert list(result.keys()) == [
        "markers",
        "marker_points",
        "marker_phase_polar",
        "kyhl_f2pi3_normalized_admittance_audit",
        "kyhl_admittance_points",
        "kyhl_admittance_transitions",
        "cell_iris_response_comparison",
        "coupler_cavity_parameter_estimates",
        "grid_rc_line_scan",
        "phase_advance",
        "phase_summary",
        "nodal_shift",
        "geometry_phase_response",
    ]
    assert len(result["markers"]) == 3
    assert len(result["marker_points"]) == 6
    assert len(result["kyhl_f2pi3_normalized_admittance_audit"]) == 2
    assert result["kyhl_f2pi3_normalized_admittance_audit"]["marker_name"].unique().tolist() == ["f_2pi3"]
    assert "line_normalized_admittance_real" in result["kyhl_f2pi3_normalized_admittance_audit"]
    assert "mode_reflection_angle_deg" in result["kyhl_f2pi3_normalized_admittance_audit"]
    assert len(result["kyhl_admittance_points"]) == 6
    assert len(result["kyhl_admittance_transitions"]) == 3
    assert result["cell_iris_response_comparison"].empty
    assert len(result["coupler_cavity_parameter_estimates"]) == 2
    assert result["coupler_cavity_parameter_estimates"]["coupling_beta_status"].unique().tolist() == ["ok"]
    assert result["coupler_cavity_parameter_estimates"]["coupling_k_source"].unique().tolist() == [
        "marker_frequency_ratio_abs"
    ]
    assert result["grid_rc_line_scan"].empty
    assert len(result["phase_advance"]) == 3
    assert len(result["phase_summary"]) == 3
    assert len(result["nodal_shift"]) == 2
    assert result["geometry_phase_response"].empty
    assert set(result["marker_points"]["source_file"]) == {
        "0.5_processed.csv",
        "1.5_processed.csv",
    }
    assert "s_phase_deg" in result["marker_points"].columns
    assert result["marker_phase_polar"]["f_2pi3_phase_deg"].tolist() == [10.0, -110.0]
    assert result["marker_phase_polar"]["f_mean_phase_deg"].tolist() == [20.0, -100.0]
    assert result["marker_phase_polar"]["f_pi2_phase_deg"].tolist() == [30.0, -90.0]
    first_phase = result["phase_advance"].sort_values("marker_name").iloc[0]
    assert first_phase["from_tune_position"] == 0.5
    assert first_phase["to_tune_position"] == 1.5
    assert first_phase["position_family"] == "cell"


def test_build_marker_analysis_skips_transition_phase_advance_for_sim_260526_grid_scan(
    tmp_path: Path,
) -> None:
    sparameter_path = tmp_path / "data" / "sim" / "sim_grid_260526_scan"
    dispersion_path = tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"
    _write_grid_sim_dataset(sparameter_path)
    _write_dispersion_summary(dispersion_path)

    result = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role="sim",
    )

    assert result["marker_points"]["scan_type"].unique().tolist() == ["grid_2d"]
    assert result["marker_points"]["sim_r_c"].dropna().nunique() == 2
    assert result["phase_advance"].empty
    assert result["kyhl_admittance_transitions"].empty
    assert result["cell_iris_response_comparison"].empty
    assert result["coupler_cavity_parameter_estimates"].empty
    assert result["phase_summary"].empty
    assert result["nodal_shift"].empty
    assert "phase_advance_0to360_deg" in result["phase_advance"].columns
    assert "transition_count" in result["phase_summary"].columns
    assert "phase_error_from_target_deg" in result["nodal_shift"].columns


def test_build_marker_analysis_extracts_fixed_width_rc_line_scan_for_grid_scan(
    tmp_path: Path,
) -> None:
    sparameter_path = tmp_path / "data" / "sim" / "sim_grid_260526_scan"
    dispersion_path = tmp_path / "data" / "sim" / "sim_dispersion_260505_single_cell_step1"
    _write_rc_line_grid_sim_dataset(sparameter_path)
    _write_dispersion_summary(dispersion_path)

    result = build_marker_analysis(
        sparameter_path=sparameter_path,
        dispersion_path=dispersion_path,
        marker_role="sim",
    )

    line_scan = result["grid_rc_line_scan"]
    assert line_scan["sim_w_c"].tolist() == [19.3224, 19.3224]
    assert line_scan["sim_r_c"].tolist() == [54.59, 55.59]
    assert line_scan["source_file"].tolist() == ["run_2.s1p", "run_1.s1p"]
    assert line_scan["f_2pi3_phase_deg"].tolist() == [10.0, 40.0]


def test_build_marker_phase_polar_table_preserves_offset_cell_sweep_column() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "source_file": "run_1.s2p",
                "run_id": 1,
                "tune_position": 5.0,
                "s_name": "S11",
                "marker_role": "sim",
                "scan_type": "tune_position",
                "sim_NumDepth": 5.0,
                "sim_offset_cell_03": 0.0,
                "marker_name": "f_2pi3",
                "s_phase_deg": 10.0,
            },
            {
                "dataset_id": "dataset",
                "source_file": "run_1.s2p",
                "run_id": 1,
                "tune_position": 5.0,
                "s_name": "S11",
                "marker_role": "sim",
                "scan_type": "tune_position",
                "sim_NumDepth": 5.0,
                "sim_offset_cell_03": 0.0,
                "marker_name": "f_mean",
                "s_phase_deg": 20.0,
            },
            {
                "dataset_id": "dataset",
                "source_file": "run_1.s2p",
                "run_id": 1,
                "tune_position": 5.0,
                "s_name": "S11",
                "marker_role": "sim",
                "scan_type": "tune_position",
                "sim_NumDepth": 5.0,
                "sim_offset_cell_03": 0.0,
                "marker_name": "f_pi2",
                "s_phase_deg": 30.0,
            },
        ]
    )

    table = build_marker_phase_polar_table(marker_points)

    assert table.loc[0, "sim_offset_cell_03"] == 0.0


def test_save_marker_analysis_writes_csv_tables(tmp_path: Path) -> None:
    result = {
        "markers": pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
        "marker_points": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "s_phase_deg": 10.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "marker_phase_polar": pd.DataFrame([{"source_file": "run1.s2p", "f_2pi3_phase_deg": 10.0}]),
        "kyhl_f2pi3_normalized_admittance_audit": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "line_normalized_admittance_real": 1.0,
                    "mode_reflection_angle_deg": 240.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "kyhl_admittance_points": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "operation_scaled_admittance_angle_deg": 60.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "kyhl_admittance_transitions": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "operation_scaled_admittance_angle_deg": 60.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "cell_iris_response_comparison": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "operation_scaled_admittance_response_ratio_iris_over_cell": 2.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "coupler_cavity_parameter_estimates": pd.DataFrame(
            [
                {
                    "source_file": "run1.s2p",
                    "coupler_frequency_ghz": 2.866,
                    "external_quality_factor": 50.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "grid_rc_line_scan": pd.DataFrame([{"source_file": "run2.s2p", "sim_r_c": 55.59, "sim_w_c": 19.3224}]),
        "phase_advance": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "phase_error_from_240_deg": 0.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "phase_summary": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "transition_count": 1,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "nodal_shift": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "phase_error_from_target_deg": 0.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
        "geometry_phase_response": pd.DataFrame(
            [
                {
                    "marker_name": "f_2pi3",
                    "sweep_axis": "sim_offset_cell_03",
                    "sweep_value": 0.0,
                    "phase_delta_shift_from_baseline_deg": 0.0,
                    "data_kind": "experiment",
                    "port_side": pd.NA,
                }
            ]
        ),
    }
    output_dir = tmp_path / "analysis_outputs"

    paths = save_marker_analysis(result, output_dir)

    assert list(paths.keys()) == [
        "markers",
        "marker_points",
        "marker_phase_polar",
        "kyhl_f2pi3_normalized_admittance_audit",
        "kyhl_admittance_points",
        "kyhl_admittance_transitions",
        "cell_iris_response_comparison",
        "coupler_cavity_parameter_estimates",
        "grid_rc_line_scan",
        "phase_advance",
        "phase_summary",
        "nodal_shift",
        "geometry_phase_response",
    ]
    for path in paths.values():
        assert path.exists()
    saved_marker_points = pd.read_csv(paths["marker_points"])
    assert saved_marker_points.loc[0, "s_phase_deg"] == 10.0
    for table_name in [
        "marker_points",
        "marker_phase_polar",
        "kyhl_f2pi3_normalized_admittance_audit",
        "kyhl_admittance_points",
        "kyhl_admittance_transitions",
        "cell_iris_response_comparison",
        "coupler_cavity_parameter_estimates",
        "grid_rc_line_scan",
        "phase_advance",
        "phase_summary",
        "nodal_shift",
        "geometry_phase_response",
    ]:
        saved = pd.read_csv(paths[table_name])
        assert "data_kind" not in saved.columns
        assert "port_side" not in saved.columns


def test_build_marker_phase_polar_table_writes_one_row_per_run_with_marker_phase_columns() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "source_file": "run_2.s1p",
                "run_id": 2,
                "marker_name": "f_2pi3",
                "s_phase_deg": -110.0,
            },
            {
                "source_file": "run_2.s1p",
                "run_id": 2,
                "marker_name": "f_mean",
                "s_phase_deg": -100.0,
            },
            {
                "source_file": "run_2.s1p",
                "run_id": 2,
                "marker_name": "f_pi2",
                "s_phase_deg": -90.0,
            },
            {
                "source_file": "run_1.s1p",
                "run_id": 1,
                "marker_name": "f_2pi3",
                "s_phase_deg": 10.0,
            },
            {
                "source_file": "run_1.s1p",
                "run_id": 1,
                "marker_name": "f_mean",
                "s_phase_deg": 20.0,
            },
            {
                "source_file": "run_1.s1p",
                "run_id": 1,
                "marker_name": "f_pi2",
                "s_phase_deg": 30.0,
            },
        ]
    )

    result = build_marker_phase_polar_table(marker_points)

    assert list(result.columns) == [
        "source_file",
        "run_id",
        "f_2pi3_phase_deg",
        "f_mean_phase_deg",
        "f_pi2_phase_deg",
    ]
    assert result["run_id"].tolist() == [1, 2]
    assert result["f_2pi3_phase_deg"].tolist() == [10.0, -110.0]
    assert result["f_mean_phase_deg"].tolist() == [20.0, -100.0]
    assert result["f_pi2_phase_deg"].tolist() == [30.0, -90.0]


def test_build_marker_analysis_samples_only_s11_rows(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, pd.DataFrame] = {}

    class FakeLoader:
        def load(self, path):
            return pd.DataFrame(
                [
                    {"source_file": "0.5.s2p", "s_name": "S11", "freq_ghz": 2.856},
                    {"source_file": "0.5.s2p", "s_name": "S21", "freq_ghz": 2.856},
                ]
            )

    def fake_sample_nearest_markers(sparameter_table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
        captured["sparameter_table"] = sparameter_table
        return pd.DataFrame(
            [
                {
                    "dataset_id": "dataset",
                    "data_kind": "experiment",
                    "data_layer": "prepro",
                    "source_file": "0.5.s2p",
                    "tune_position": 0.5,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "exp",
                    "target_freq_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "s_db": -1.0,
                    "s_phase_deg": 10.0,
                },
                {
                    "dataset_id": "dataset",
                    "data_kind": "experiment",
                    "data_layer": "prepro",
                    "source_file": "1.5.s2p",
                    "tune_position": 1.5,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "exp",
                    "target_freq_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "s_db": -2.0,
                    "s_phase_deg": -110.0,
                },
            ]
        )

    monkeypatch.setattr(
        marker_pipeline,
        "extract_marker_frequencies",
        lambda *args, **kwargs: pd.DataFrame([{"marker_name": "f_2pi3", "freq_ghz": 2.856}]),
    )
    monkeypatch.setattr(marker_pipeline, "sample_nearest_markers", fake_sample_nearest_markers)

    build_marker_analysis(
        sparameter_path=tmp_path / "data" / "prepro" / "dataset",
        dispersion_path=tmp_path / "data" / "sim" / "dispersion",
        marker_role="exp",
        loader=FakeLoader(),
    )

    assert captured["sparameter_table"]["s_name"].tolist() == ["S11"]
