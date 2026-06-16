from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader


def test_sim_loader_merges_result_navigator_by_run_id(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "case_a"
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"NumDepth"\t"r_c"\t"w_c"\n'
        '"7"\t"2"\t"54.59"\t"18.3224"\n',
        encoding="utf-8",
    )
    (folder / "run_7.s1p").write_text(
        "# GHz S RI R 0\n2.6 1 0\n",
        encoding="utf-8",
    )

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    row = table.iloc[0].to_dict()
    assert row["run_id"] == 7
    assert row["sim_NumDepth"] == 2
    assert row["sim_r_c"] == 54.59
    assert row["sim_w_c"] == 18.3224
    assert row["scan_type"] == "single_point"


def test_sim_loader_merges_result_navigator_by_num_tune_for_cell_files(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "tds_body_plunger"
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"NumTune"\n'
        '"6"\t"0"\n'
        '"5"\t"1"\n',
        encoding="utf-8",
    )
    (folder / "260504_Cell_0.5_RI.s1p").write_text(
        "# GHz S RI R 50\n2.6 1 0\n",
        encoding="utf-8",
    )
    (folder / "260504_Cell_1.5_RI.s1p").write_text(
        "# GHz S RI R 50\n2.6 1 0\n",
        encoding="utf-8",
    )

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    rows = table[["source_file", "tune_position", "sim_NumTune", "run_id"]].drop_duplicates()
    by_file = rows.set_index("source_file").to_dict("index")
    assert by_file["260504_Cell_0.5_RI.s1p"]["tune_position"] == 0.5
    assert by_file["260504_Cell_0.5_RI.s1p"]["sim_NumTune"] == 0
    assert by_file["260504_Cell_0.5_RI.s1p"]["run_id"] == 6
    assert by_file["260504_Cell_1.5_RI.s1p"]["tune_position"] == 1.5
    assert by_file["260504_Cell_1.5_RI.s1p"]["sim_NumTune"] == 1
    assert by_file["260504_Cell_1.5_RI.s1p"]["run_id"] == 5
    assert table["scan_type"].unique().tolist() == ["tune_position"]


def test_sim_loader_marks_2d_grid_scan_from_result_navigator_geometry(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "grid_scan"
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\t"w_c"\n'
        '"1"\t"54.59"\t"18.3224"\n'
        '"2"\t"55.59"\t"19.3224"\n',
        encoding="utf-8",
    )
    (folder / "run_1.s1p").write_text("# GHz S RI R 50\n2.6 1 0\n", encoding="utf-8")
    (folder / "run_2.s1p").write_text("# GHz S RI R 50\n2.6 1 0\n", encoding="utf-8")

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    assert table["scan_type"].unique().tolist() == ["grid_2d"]


def test_raw_loader_does_not_expect_result_navigator(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "case_b"
    folder.mkdir(parents=True)
    (folder / "run_7.s1p").write_text(
        "# GHz S RI R 50\n2.6 1 0\n",
        encoding="utf-8",
    )

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    assert "sim_r_c" not in table.columns
    assert table.iloc[0]["data_kind"] == "experiment"
