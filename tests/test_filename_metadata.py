from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_halfbrazing_in_out_file_gets_port_side_and_tune_position(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "250609_beforebrazing"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2600000000.0;1.0;0.0;\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["tune_position"] == 0.5
    assert row["port_side"] == "in"


def test_fullbrazing_ignores_in_out_even_if_filename_contains_it(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "260415_fullbrazing"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell_processed.csv").write_text(
        "freq[Hz],Magnitude,Phase_deg\n2756000000.0,-1.2,90\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["tune_position"] == 0.5
    assert row["port_side"] is None


def test_simple_numeric_file_gets_tune_position_only(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "260415_fullbrazing"
    folder.mkdir(parents=True)
    (folder / "8_processed.csv").write_text(
        "freq[Hz],Magnitude,Phase_deg\n2756000000.0,-1.2,90\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["tune_position"] == 8.0
    assert row["port_side"] is None


def test_touchstone_cell_file_gets_cell_tune_position_not_date(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "260504_tds_body_plunger"
    folder.mkdir(parents=True)
    (folder / "260504_Cell_0.5_RI.s2p").write_text(
        "# GHz S RI R 50\n2.6 1 0 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["tune_position"] == 0.5
    assert row["port_side"] is None


def test_sim_touchstone_run_id_is_not_treated_as_tune_position(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "260527_iris_line_sweep"
    folder.mkdir(parents=True)
    (folder / "run_001.s1p").write_text(
        "# GHz S RI R 0\n2.6 1 0\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["tune_position"] is None
    assert row["port_side"] is None
