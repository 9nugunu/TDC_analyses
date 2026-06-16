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
                "target_freq_ghz": 2.856,
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
                "target_freq_ghz": 2.856,
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
                "target_freq_ghz": 2.856,
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
        "from_source_file",
        "to_source_file",
        "from_tune_position",
        "to_tune_position",
        "target_freq_ghz",
        "from_freq_ghz",
        "to_freq_ghz",
        "from_s_db",
        "to_s_db",
        "from_phase_deg",
        "to_phase_deg",
        "signed_phase_step_deg",
        "phase_advance_0to360_deg",
        "phase_error_from_240_deg",
    ]
    assert len(result) == 2
    first = result.iloc[0]
    assert pd.isna(first["port_side"])
    assert first["from_tune_position"] == 0.5
    assert first["to_tune_position"] == 1.5
    assert first["from_phase_deg"] == 10.0
    assert first["to_phase_deg"] == -110.0
    assert first["signed_phase_step_deg"] == pytest.approx(-120.0)
    assert first["phase_advance_0to360_deg"] == pytest.approx(240.0)
    assert first["phase_error_from_240_deg"] == pytest.approx(0.0)
    second = result.iloc[1]
    assert second["signed_phase_step_deg"] == pytest.approx(-120.0)
    assert second["phase_advance_0to360_deg"] == pytest.approx(240.0)


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
                        "target_freq_ghz": 2.856,
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
                        "target_freq_ghz": 2.856,
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
        assert row["from_tune_position"] == 0.5
        assert row["to_tune_position"] == 1.5
        assert row["phase_advance_0to360_deg"] == pytest.approx(240.0)
        assert row["phase_error_from_240_deg"] == pytest.approx(0.0)


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
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": phase,
                "source_format": "processed_csv_db_phase",
            }
        )
    marker_points = pd.DataFrame(rows)

    result = compute_phase_advance(marker_points)

    transitions = set(zip(result["position_family"], result["from_tune_position"], result["to_tune_position"]))
    assert transitions == {
        ("cell", 0.5, 1.5),
        ("cell", 1.5, 2.5),
        ("iris", 1.0, 2.0),
    }
    assert (0.5, 1.0) not in set(zip(result["from_tune_position"], result["to_tune_position"]))
    assert (1.0, 1.5) not in set(zip(result["from_tune_position"], result["to_tune_position"]))


def test_compute_phase_advance_returns_empty_table_when_geometry_scan_has_no_tune_positions() -> None:
    marker_points = pd.DataFrame(
        [
            {
                "dataset_id": "grid_scan",
                "data_kind": "simulation",
                "data_layer": "sim",
                "source_file": "run_001.s1p",
                "tune_position": pd.NA,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "sim",
                "target_freq_ghz": 2.856,
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
                    "dataset_id": "grid_scan",
                    "data_kind": "simulation",
                    "data_layer": "sim",
                    "source_file": f"cell_{tune_position}_{sim_r_c}_{sim_w_c}.s1p",
                    "tune_position": tune_position,
                    "port_side": None,
                    "s_name": "S11",
                    "marker_name": "f_2pi3",
                    "marker_role": "sim",
                    "target_freq_ghz": 2.856,
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
        assert row["from_tune_position"] == 0.5
        assert row["to_tune_position"] == 1.5
        assert row["phase_advance_0to360_deg"] == pytest.approx(240.0)
