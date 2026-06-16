from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_raw_loader_processes_ri_csv_export(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "case_ri"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc1_S11;im:Trc1_S11;\n"
        "2600000000.0;1.0;0.0;\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["dataset_id"] == "case_ri"
    assert row["data_kind"] == "experiment"
    assert row["data_layer"] == "raw"
    assert row["s_name"] == "S11"
    assert row["s_real"] == 1.0
    assert row["s_imag"] == 0.0
    assert row["s_db"] == 0.0
    assert row["s_phase_deg"] == 0.0
    assert row["source_format"] == "raw_csv_ri"


def test_raw_loader_processes_ri_csv_export_with_non_trc1_header(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "case_ri_trc3"
    folder.mkdir(parents=True)
    (folder / "in_1.0iris.csv").write_text(
        "# Version 1.00\n#\nfreq[Hz];re:Trc3_S11;im:Trc3_S11;re:Trc4_S11;im:Trc4_S11;\n"
        "2600000000.0;1.0;0.0;2.0;0.0;\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["dataset_id"] == "case_ri_trc3"
    assert row["source_file"] == "in_1.0iris.csv"
    assert row["tune_position"] == 1.0
    assert row["port_side"] == "in"
    assert row["s_real"] == 1.0
    assert row["s_imag"] == 0.0
    assert row["s_db"] == 0.0
    assert row["s_phase_deg"] == 0.0
    assert row["source_format"] == "raw_csv_ri"


def test_raw_loader_processes_formatted_mag_phase_csv_export(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "case_mag_phase"
    folder.mkdir(parents=True)
    (folder / "0.5.csv").write_text(
        '"# Channel 1"\n"# Trace 1"\nFrequency, Formatted Data, Formatted Data\n'
        "+2.80600000000E+009, +1.88210142879E-001, -9.23982503755E+001\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["freq_ghz"] == 2.806
    assert row["s_db"] == 0.188210142879
    assert row["s_phase_deg"] == -92.3982503755
    assert row["source_format"] == "raw_csv_db_phase"
