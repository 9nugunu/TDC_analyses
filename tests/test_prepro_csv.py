from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_prepro_loader_reads_processed_csv_with_phase(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "prepro_sweep_260415_case_a"
    folder.mkdir(parents=True)
    (folder / "8_processed.csv").write_text(
        "freq[Hz],Magnitude,Phase_deg\n2756000000.0,-0.089,136.27\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    row = table.iloc[0]
    assert row["dataset_id"] == "prepro_sweep_260415_case_a"
    assert row["data_kind"] == "experiment"
    assert row["data_layer"] == "prepro"
    assert row["source_file"] == "8_processed.csv"
    assert row["freq_hz"] == 2756000000.0
    assert row["freq_ghz"] == 2.756
    assert row["s_name"] == "S11"
    assert row["s_db"] == -0.089
    assert row["s_phase_deg"] == 136.27
    assert row["source_format"] == "processed_csv_db_phase"


def test_prepro_loader_reads_processed_csv_without_phase(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "prepro_sweep_260415_case_b"
    folder.mkdir(parents=True)
    (folder / "in_0.5cell_processed.csv").write_text(
        "freq[Hz],Magnitude\n2756000000.0,-0.12\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)

    assert table.iloc[0]["s_db"] == -0.12
    assert "s_phase_deg" in table.columns
