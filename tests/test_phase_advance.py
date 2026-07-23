import pandas as pd
import pytest

from deflector_tuning.analysis.phase_advance import compute_phase_advance


def test_compute_phase_advance_treats_negative_120_as_240_degree_advance() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.5_processed.csv",
                "tune_position": 1.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -2.0,
                "s_phase_deg": -110.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "2.5_processed.csv",
                "tune_position": 2.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 130.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )

    result = compute_phase_advance(marker_points)

    assert list(result.columns) == [
        "dataset_id",
        "data_kind",
        "data_layer",
        "marker_name",
        "marker_role",
        "port_side",
        "s_name",
        "position_family",
        "file_from",
        "file_to",
        "pos_from",
        "pos_to",
        "freq_target_ghz",
        "freq_ghz",
        "s_db_from",
        "s_db_to",
        "phase_from_deg",
        "phase_to_deg",
        "phase_step_deg",
        "phase_adv_deg",
        "phase_err_240_deg",
    ]
    assert len(result) == 2
    first = result.iloc[0]
    assert pd.isna(first["port_side"])
    assert first["pos_from"] == 0.5
    assert first["pos_to"] == 1.5
    assert first["phase_from_deg"] == 10.0
    assert first["phase_to_deg"] == -110.0
    assert first["phase_step_deg"] == pytest.approx(-120.0)
    assert first["phase_adv_deg"] == pytest.approx(240.0)
    assert first["phase_err_240_deg"] == pytest.approx(0.0)
    second = result.iloc[1]
    assert second["phase_step_deg"] == pytest.approx(-120.0)
    assert second["phase_adv_deg"] == pytest.approx(240.0)


def test_compute_phase_advance_keeps_markers_and_port_sides_separate() -> None:
    rows = []
    for marker_name, phase_offset in [("f_2pi3", 0.0), ("f_mean", 20.0)]:
        for port_side, db_offset in [("in", 0.0), ("out", -10.0)]:
            rows.extend(
                [
                    {
                        "dataset_id": "dataset",
                        "data_kind": "experiment",
                        "data_layer": "raw",
                        "source_file": f"{port_side}_0.5cell.csv",
                        "tune_position": 0.5,
                        "port_side": port_side,
                        "s_name": "S11",
                        "marker_name": marker_name,
                        "marker_role": "exp",
                        "freq_target_ghz": 2.856,
                        "freq_ghz": 2.8565,
                        "freq_error_ghz": 0.0005,
                        "s_db": -1.0 + db_offset,
                        "s_phase_deg": 10.0 + phase_offset,
                        "source_format": "raw_csv_ri",
                    },
                    {
                        "dataset_id": "dataset",
                        "data_kind": "experiment",
                        "data_layer": "raw",
                        "source_file": f"{port_side}_1.5cell.csv",
                        "tune_position": 1.5,
                        "port_side": port_side,
                        "s_name": "S11",
                        "marker_name": marker_name,
                        "marker_role": "exp",
                        "freq_target_ghz": 2.856,
                        "freq_ghz": 2.8565,
                        "freq_error_ghz": 0.0005,
                        "s_db": -2.0 + db_offset,
                        "s_phase_deg": -110.0 + phase_offset,
                        "source_format": "raw_csv_ri",
                    },
                ]
            )
    marker_points = pd.DataFrame(rows)

    result = compute_phase_advance(marker_points)

    assert len(result) == 4
    assert set(result["marker_name"]) == {"f_2pi3", "f_mean"}
    assert set(result["port_side"]) == {"in", "out"}
    for _, row in result.iterrows():
        assert row["pos_from"] == 0.5
        assert row["pos_to"] == 1.5
        assert row["phase_adv_deg"] == pytest.approx(240.0)
        assert row["phase_err_240_deg"] == pytest.approx(0.0)


def test_compute_phase_advance_skips_same_position_file_pairs() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "raw",
                "source_file": "1_noportE.S2P",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "raw",
                "source_file": "1_portE.S2P",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": 20.0,
            },
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "raw",
                "source_file": "2_portE.S2P",
                "tune_position": 2.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "s_db": -1.0,
                "s_phase_deg": -100.0,
            },
        ]
    )

    result = compute_phase_advance(marker_points)

    assert result[["file_from", "file_to", "pos_from", "pos_to"]].to_dict("records") == [
        {
            "file_from": "1_portE.S2P",
            "file_to": "2_portE.S2P",
            "pos_from": 1.0,
            "pos_to": 2.0,
        }
    ]


def test_compute_phase_advance_uses_periodic_position_families_not_adjacent_mixed_positions() -> None:
    rows = []
    for tune_position, phase in [(0.5, 10.0), (1.0, 80.0), (1.5, -110.0), (2.0, -40.0), (2.5, 130.0)]:
        rows.append(
            {
                "dataset_id": "dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": f"{tune_position}_processed.csv",
                "tune_position": tune_position,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": phase,
                "source_format": "processed_csv_db_phase",
            }
        )
    marker_points = pd.DataFrame(rows)

    result = compute_phase_advance(marker_points)

    transitions = set(zip(result["position_family"], result["pos_from"], result["pos_to"]))
    assert transitions == {
        ("cell", 0.5, 1.5),
        ("cell", 1.5, 2.5),
        ("iris", 1.0, 2.0),
    }
    assert (0.5, 1.0) not in set(zip(result["pos_from"], result["pos_to"]))
    assert (1.0, 1.5) not in set(zip(result["pos_from"], result["pos_to"]))


def test_compute_phase_advance_returns_empty_table_when_geometry_scan_has_no_tune_positions() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "sim_grid_260526_scan",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_001.s1p",
                "tune_position": pd.NA,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "freq_target_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "source_format": "touchstone_ri",
                "sim_r_c": 54.5,
                "sim_w_c": 18.5,
            }
        ]
    )

    result = compute_phase_advance(marker_points)

    assert result.empty
    assert "sim_r_c" in result.columns
    assert "sim_w_c" in result.columns


def test_compute_phase_advance_keeps_simulation_geometry_points_separate() -> None:
    rows = []
    for sim_r_c, sim_w_c, phase_offset in [(54.5, 18.5, 0.0), (54.75, 18.75, 15.0)]:
        for tune_position, phase in [(0.5, 10.0), (1.5, -110.0)]:
            rows.append(
                {
                    "dataset_id": "sim_grid_260526_scan",
                    "data_kind": "simulation",
                    "data_layer": "sim",
                    "source_file": f"cell_{tune_position}_{sim_r_c}_{sim_w_c}.s1p",
                    "tune_position": tune_position,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "sim",
                    "freq_target_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "freq_error_ghz": 0.0,
                    "s_db": -1.0,
                    "s_phase_deg": phase + phase_offset,
                    "source_format": "touchstone_ri",
                    "sim_r_c": sim_r_c,
                    "sim_w_c": sim_w_c,
                    "sim_NumTune": int(tune_position - 0.5),
                }
            )
    marker_points = pd.DataFrame(rows)

    result = compute_phase_advance(marker_points)

    assert len(result) == 2
    assert set(result["sim_r_c"]) == {54.5, 54.75}
    assert set(result["sim_w_c"]) == {18.5, 18.75}
    assert "sim_NumTune" not in result.columns
    for _, row in result.iterrows():
        assert row["pos_from"] == 0.5
        assert row["pos_to"] == 1.5
        assert row["phase_adv_deg"] == pytest.approx(240.0)


def test_compute_phase_advance_treats_num_depth_as_tune_axis_not_geometry() -> None:
    rows = []
    for depth_offset, phase_offset in [(-3.0, 0.0), (0.0, 10.0)]:
        for num_depth, tune_position, phase in [(1, 0.5, 10.0), (2, 1.5, -110.0)]:
            rows.append(
                {
                    "dataset_id": "sim_sweep_260526_iris_plunger_offset",
                    "data_kind": "simulation",
                    "data_layer": "sim",
                    "source_file": f"depth_{depth_offset}_{num_depth}.s1p",
                    "tune_position": tune_position,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "sim",
                    "freq_target_ghz": 2.856,
                    "freq_ghz": 2.856,
                    "freq_error_ghz": 0.0,
                    "s_db": -1.0,
                    "s_phase_deg": phase + phase_offset,
                    "source_format": "touchstone_ri",
                    "sim_DepthPlunger_offset": depth_offset,
                    "sim_NumDepth": num_depth,
                }
            )
    marker_points = pd.DataFrame(rows)

    result = compute_phase_advance(marker_points)

    assert len(result) == 2
    assert set(result["sim_DepthPlunger_offset"]) == {-3.0, 0.0}
    assert "sim_NumDepth" not in result.columns
    for _, row in result.iterrows():
        assert row["pos_from"] == 0.5
        assert row["pos_to"] == 1.5
        assert row["phase_adv_deg"] == pytest.approx(240.0)
