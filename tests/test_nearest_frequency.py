from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.central_loader import DataLoader


def test_extract_nearest_picks_one_row_per_raw_file_port_and_s_name(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "250609_beforebrazing"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2854000000.0;1.0;0.0;\n"
        "2857000000.0;0.0;-1.0;\n"
        "2860000000.0;-1.0;0.0;\n",
        encoding="utf-8",
    )
    (folder / "out_1.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2853000000.0;0.0;1.0;\n"
        "2855800000.0;1.0;0.0;\n",
        encoding="utf-8",
    )

    nearest = DataLoader().extract_nearest(folder, target_freq_ghz=2.856)

    assert list(nearest.columns) == [
        "dataset_id",
        "data_kind",
        "data_layer",
        "source_file",
        "tune_position",
        "port_side",
        "s_name",
        "target_freq_ghz",
        "freq_ghz",
        "freq_error_ghz",
        "s_db",
        "s_phase_deg",
        "source_format",
    ]
    records = nearest.sort_values(["port_side", "tune_position"]).to_dict("records")
    assert records == [
        {
            "dataset_id": "250609_beforebrazing",
            "data_kind": "experiment",
            "data_layer": "raw",
            "source_file": "in_0.5cell.csv",
            "tune_position": 0.5,
            "port_side": "in",
            "s_name": "S11",
            "target_freq_ghz": 2.856,
            "freq_ghz": 2.857,
            "freq_error_ghz": pytest.approx(0.001),
            "s_db": 0.0,
            "s_phase_deg": -90.0,
            "source_format": "raw_csv_ri",
        },
        {
            "dataset_id": "250609_beforebrazing",
            "data_kind": "experiment",
            "data_layer": "raw",
            "source_file": "out_1.5cell.csv",
            "tune_position": 1.5,
            "port_side": "out",
            "s_name": "S11",
            "target_freq_ghz": 2.856,
            "freq_ghz": 2.8558,
            "freq_error_ghz": pytest.approx(-0.0002),
            "s_db": 0.0,
            "s_phase_deg": 0.0,
            "source_format": "raw_csv_ri",
        },
    ]


def test_extract_nearest_keeps_missing_port_side_as_na_for_prepro(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "260415_fullbrazing"
    folder.mkdir(parents=True)
    (folder / "8_processed.csv").write_text(
        "freq[Hz],Magnitude,Phase_deg\n"
        "2854000000.0,-2.0,10\n"
        "2856100000.0,-3.0,20\n",
        encoding="utf-8",
    )

    nearest = DataLoader().extract_nearest(folder, target_freq_ghz=2.856)

    assert len(nearest) == 1
    row = nearest.iloc[0]
    assert row["source_file"] == "8_processed.csv"
    assert row["tune_position"] == 8.0
    assert pd.isna(row["port_side"])
    assert row["freq_ghz"] == 2.8561
    assert row["freq_error_ghz"] == pytest.approx(0.0001)
    assert row["s_db"] == -3.0
    assert row["s_phase_deg"] == 20.0


def test_extract_nearest_groups_sim_without_tune_position_or_port_side(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "260527_iris_line_sweep"
    folder.mkdir(parents=True)
    (folder / "run_001.s1p").write_text(
        "# GHz S RI R 0\n"
        "2.855 1.0 0.0\n"
        "2.8562 0.0 1.0\n",
        encoding="utf-8",
    )

    nearest = DataLoader().extract_nearest(folder, target_freq_ghz=2.856)

    assert len(nearest) == 1
    row = nearest.iloc[0]
    assert row["source_file"] == "run_001.s1p"
    assert pd.isna(row["tune_position"])
    assert pd.isna(row["port_side"])
    assert row["s_name"] == "S11"
    assert row["freq_ghz"] == 2.8562
    assert row["freq_error_ghz"] == pytest.approx(0.0002)
    assert row["s_phase_deg"] == 90.0
    assert row["source_format"] == "touchstone_ri"
