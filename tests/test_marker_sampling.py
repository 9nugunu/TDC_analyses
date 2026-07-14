import pandas as pd
import pytest

from deflector_tuning.markers.sampling import sample_nearest_markers


def test_sample_nearest_markers_returns_long_form_rows_per_group_and_marker() -> None:
    s_table = pd.DataFrame(
        [
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "freq_ghz": 2.855,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "freq_ghz": 2.8561,
                "s_db": -2.0,
                "s_phase_deg": 20.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "freq_ghz": 2.8662,
                "s_db": -3.0,
                "s_phase_deg": 30.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.5_processed.csv",
                "tune_position": 1.5,
                "port_side": None,
                "s_name": "S11",
                "freq_ghz": 2.8559,
                "s_db": -4.0,
                "s_phase_deg": 40.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.5_processed.csv",
                "tune_position": 1.5,
                "port_side": None,
                "s_name": "S11",
                "freq_ghz": 2.8661,
                "s_db": -5.0,
                "s_phase_deg": 50.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )
    markers = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "freq_ghz": 2.856,
                "marker_role": "exp",
                "marker_source": "dispersion_mode_1_temp_humidity_corrected",
                "uncorrected_freq_ghz": 2.857,
                "frequency_scale_factor": 0.99957,
                "temp_op_C": 20.0,
                "temp_meas_C": 24.4,
                "humidity_fraction": 0.65,
            },
            {
                "marker_name": "f_mean",
                "freq_ghz": 2.866,
                "marker_role": "exp",
                "marker_source": "dispersion_mode_1_temp_humidity_corrected",
                "uncorrected_freq_ghz": 2.867,
                "frequency_scale_factor": 0.99957,
                "temp_op_C": 20.0,
                "temp_meas_C": 24.4,
                "humidity_fraction": 0.65,
            },
        ]
    )

    sampled = sample_nearest_markers(s_table, markers)

    assert list(sampled.columns) == [
        "dataset_id",
        "data_kind",
        "data_layer",
        "source_file",
        "tune_position",
        "port_side",
        "s_name",
        "marker_name",
        "marker_role",
        "marker_source",
        "freq_target_ghz",
        "freq_ghz",
        "freq_error_ghz",
        "s_db",
        "s_phase_deg",
        "source_format",
        "uncorrected_freq_ghz",
        "frequency_scale_factor",
        "temp_op_C",
        "temp_meas_C",
        "humidity_fraction",
    ]
    assert len(sampled) == 4
    records = sampled.sort_values(["source_file", "marker_name"]).to_dict("records")
    assert records[0]["source_file"] == "0.5_processed.csv"
    assert records[0]["marker_name"] == "f_2pi3"
    assert records[0]["freq_target_ghz"] == 2.856
    assert records[0]["freq_ghz"] == 2.8561
    assert records[0]["freq_error_ghz"] == pytest.approx(0.0001)
    assert records[0]["s_db"] == -2.0
    assert records[0]["s_phase_deg"] == 20.0
    assert pd.isna(records[0]["port_side"])
    assert records[0]["marker_role"] == "exp"
    assert records[0]["frequency_scale_factor"] == 0.99957
    assert records[3]["source_file"] == "1.5_processed.csv"
    assert records[3]["marker_name"] == "f_mean"
    assert records[3]["freq_ghz"] == 2.8661
    assert records[3]["freq_error_ghz"] == pytest.approx(0.0001)


def test_sample_nearest_markers_groups_sim_without_tune_position_or_port_side() -> None:
    s_table = pd.DataFrame(
        [
            {
                "dataset_id": "sim_dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_001.s1p",
                "s_name": "S11",
                "freq_ghz": 2.8570,
                "s_db": -1.0,
                "s_phase_deg": 80.0,
                "source_format": "touchstone_ri",
                "reference_ohm": 0.0,
                "is_normalized": False,
                "run_id": 1,
                "scan_type": "grid_2d",
                "sim_r_c": 54.5,
                "sim_w_c": 18.5,
            },
            {
                "dataset_id": "sim_dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_001.s1p",
                "s_name": "S11",
                "freq_ghz": 2.8575,
                "s_db": -2.0,
                "s_phase_deg": 90.0,
                "source_format": "touchstone_ri",
                "reference_ohm": 0.0,
                "is_normalized": False,
                "run_id": 1,
                "scan_type": "grid_2d",
                "sim_r_c": 54.5,
                "sim_w_c": 18.5,
            },
        ]
    )
    markers = pd.DataFrame(
        [
            {
                "marker_name": "f_2pi3",
                "freq_ghz": 2.8574,
                "marker_role": "sim",
                "marker_source": "dispersion_mode_1",
                "uncorrected_freq_ghz": 2.8574,
                "frequency_scale_factor": 1.0,
            }
        ]
    )

    sampled = sample_nearest_markers(s_table, markers)

    assert len(sampled) == 1
    row = sampled.iloc[0]
    assert pd.isna(row["tune_position"])
    assert pd.isna(row["port_side"])
    assert row["freq_ghz"] == 2.8575
    assert row["freq_error_ghz"] == pytest.approx(0.0001)
    assert row["reference_ohm"] == 0.0
    assert not bool(row["is_normalized"])
    assert row["run_id"] == 1
    assert row["scan_type"] == "grid_2d"
    assert row["sim_r_c"] == 54.5
    assert row["sim_w_c"] == 18.5


def test_sample_nearest_markers_canonicalizes_known_simulation_metadata() -> None:
    s_table = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run.s1p",
                "s_name": "S11",
                "freq_ghz": 2.856,
                "s_db": -10.0,
                "s_phase_deg": 20.0,
                "source_format": "touchstone_ri",
                "sim_tuner_insertion_depth": 3.5,
                "sim_coupler_path_bot2_width": 8.0,
            }
        ]
    )
    markers = pd.DataFrame(
        [{"marker_name": "f_2pi3", "freq_ghz": 2.856, "marker_role": "sim", "marker_source": "test"}]
    )

    sampled = sample_nearest_markers(s_table, markers)

    assert sampled.loc[0, "sim_tuner_depth"] == 3.5
    assert sampled.loc[0, "sim_cpl_bot2_w"] == 8.0
    assert "sim_tuner_insertion_depth" not in sampled
    assert "sim_coupler_path_bot2_width" not in sampled


def test_sample_nearest_markers_rejects_source_alias_collisions() -> None:
    s_table = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run.s1p",
                "s_name": "S11",
                "freq_ghz": 2.856,
                "s_db": -10.0,
                "s_phase_deg": 20.0,
                "source_format": "touchstone_ri",
                "sim_tuner_insertion_depth": 3.5,
                "sim_tuner_depth": 4.0,
            }
        ]
    )
    markers = pd.DataFrame(
        [{"marker_name": "f_2pi3", "freq_ghz": 2.856, "marker_role": "sim", "marker_source": "test"}]
    )

    with pytest.raises(ValueError, match="source metadata alias collision"):
        sample_nearest_markers(s_table, markers)
