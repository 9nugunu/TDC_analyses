from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.records import DataFiles


def test_loader_lists_touchstone_csv_and_other_files(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "case_a"
    folder.mkdir(parents=True)
    (folder / "a.s1p").write_text("", encoding="utf-8")
    (folder / "b.S2P").write_text("", encoding="utf-8")
    (folder / "table.csv").write_text("", encoding="utf-8")
    (folder / "note.txt").write_text("", encoding="utf-8")

    files = DataLoader().select_loader(folder).list_files(folder)

    assert files == DataFiles(
        touchstone_files=[folder / "a.s1p", folder / "b.S2P"],
        csv_files=[folder / "table.csv"],
        other_files=[folder / "note.txt"],
    )


def test_loader_ignores_subfolders_for_now(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "case_b"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "root.s2p").write_text("", encoding="utf-8")
    (nested / "nested.s2p").write_text("", encoding="utf-8")

    files = DataLoader().select_loader(folder).list_files(folder)

    assert files.touchstone_files == [folder / "root.s2p"]


def test_file_lists_are_sorted_by_name(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "prepro" / "case_c"
    folder.mkdir(parents=True)
    (folder / "z.csv").write_text("", encoding="utf-8")
    (folder / "a.csv").write_text("", encoding="utf-8")

    files = DataLoader().select_loader(folder).list_files(folder)

    assert files.csv_files == [folder / "a.csv", folder / "z.csv"]
