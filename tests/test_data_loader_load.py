from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_data_loader_load_returns_touchstone_dataframe_for_sim_folder(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_sweep_260605_case_a"
    folder.mkdir(parents=True)
    (folder / "trace_1.s1p").write_text("# GHz S RI R 0\n2.6 1 0\n", encoding="utf-8")

    table = DataLoader().load(folder)

    assert table.iloc[0]["dataset_id"] == "sim_sweep_260605_case_a"
    assert table.iloc[0]["data_kind"] == "sim"
    assert table.iloc[0]["s_name"] == "S11"


def test_data_loader_loads_requested_nested_touchstone_export_instead_of_dataset_root(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data" / "sim" / "sim_sweep_260605_case_nested"
    nested_export = dataset / "no_norm"
    nested_export.mkdir(parents=True)
    (dataset / "trace_1.s1p").write_text(
        "# GHz S RI R 50\n2.6 1 0\n",
        encoding="utf-8",
    )
    (nested_export / "trace_1.s1p").write_text(
        "# GHz S RI R 0\n2.6 0 1\n",
        encoding="utf-8",
    )

    table = DataLoader().load(nested_export)

    assert table.iloc[0]["dataset_id"] == "sim_sweep_260605_case_nested"
    assert table.iloc[0]["reference_ohm"] == 0.0
    assert table.iloc[0]["is_normalized"] == False
    assert table.iloc[0]["s_phase_deg"] == 90.0


def test_data_loader_load_returns_touchstone_dataframe_for_raw_folder(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "raw_sweep_260604_case_b"
    folder.mkdir(parents=True)
    (folder / "trace_1.s1p").write_text("# GHz S RI R 50\n2.6 1 0\n", encoding="utf-8")

    table = DataLoader().load(folder)

    assert table.iloc[0]["data_kind"] == "experiment"
    assert table.iloc[0]["is_normalized"] == True


def test_data_loader_load_returns_prepro_csv_dataframe(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "prepro_sweep_260415_case_c"
    folder.mkdir(parents=True)
    (folder / "trace_processed.csv").write_text("freq[Hz],Magnitude,Phase_deg\n1,0,0\n", encoding="utf-8")

    table = DataLoader().load(folder)

    assert table.iloc[0]["data_layer"] == "prepro"
    assert table.iloc[0]["data_kind"] == "experiment"
