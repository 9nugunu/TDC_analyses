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
