from pathlib import Path

import pytest

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.records import DataKind


def test_sim_loader_reads_touchstone_files_as_dataframe(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_sweep_260605_case_a"
    folder.mkdir(parents=True)
    (folder / "trace.s1p").write_text(
        "# GHz S RI R 0\n2.6 1.0 0.0\n2.7 0.0 -1.0\n",
        encoding="utf-8",
    )

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    assert list(table.columns) == [
        "dataset_id",
        "data_kind",
        "data_layer",
        "source_file",
        "freq_ghz",
        "s_name",
        "s_real",
        "s_imag",
        "s_db",
        "s_phase_deg",
        "source_format",
        "reference_ohm",
        "is_normalized",
        "tune_position",
        "port_side",
        "scan_type",
    ]
    assert table.to_dict("records") == [
        {
            "dataset_id": "sim_sweep_260605_case_a",
            "data_kind": "sim",
            "data_layer": "sim",
            "source_file": "trace.s1p",
            "freq_ghz": 2.6,
            "s_name": "S11",
            "s_real": 1.0,
            "s_imag": 0.0,
            "s_db": 0.0,
            "s_phase_deg": 0.0,
            "source_format": "touchstone_ri",
            "reference_ohm": 0.0,
            "is_normalized": False,
            "tune_position": None,
            "port_side": None,
            "scan_type": "single_point",
        },
        {
            "dataset_id": "sim_sweep_260605_case_a",
            "data_kind": "sim",
            "data_layer": "sim",
            "source_file": "trace.s1p",
            "freq_ghz": 2.7,
            "s_name": "S11",
            "s_real": 0.0,
            "s_imag": -1.0,
            "s_db": 0.0,
            "s_phase_deg": -90.0,
            "source_format": "touchstone_ri",
            "reference_ohm": 0.0,
            "is_normalized": False,
            "tune_position": None,
            "port_side": None,
            "scan_type": "single_point",
        },
    ]


def test_raw_loader_reads_touchstone_but_keeps_experiment_kind(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "raw_sweep_260604_case_b"
    folder.mkdir(parents=True)
    (folder / "trace.s2p").write_text(
        "# GHz S RI R 50\n2.6 1 0 2 0 3 0 4 0\n",
        encoding="utf-8",
    )

    table = DataLoader().select_loader(folder).load_touchstone(folder)

    assert table["data_kind"].unique().tolist() == [DataKind.EXP.value]
    assert table["s_name"].tolist() == ["S11", "S21", "S12", "S22"]


def test_raw_loader_reads_db_touchstone_into_common_dataframe(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "raw_sweep_260604_case_db"
    folder.mkdir(parents=True)
    (folder / "trace.s1p").write_text("# Hz S DB R 50\n2756000000 -6 90\n", encoding="utf-8")

    table = DataLoader().load(folder)
    row = table.iloc[0]

    assert row["freq_ghz"] == pytest.approx(2.756)
    assert row["s_name"] == "S11"
    assert row["s_db"] == pytest.approx(-6.0)
    assert row["s_phase_deg"] == pytest.approx(90.0)
    assert row["source_format"] == "touchstone_db"


def test_data_loader_parallel_touchstone_loading_matches_sequential_order(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "raw" / "raw_sweep_260604_parallel"
    folder.mkdir(parents=True)
    for index, phase in enumerate([0.0, 45.0, 90.0], start=1):
        (folder / f"trace_{index}.s1p").write_text(
            f"# GHz S DB R 50\n2.6 -{index} {phase}\n",
            encoding="utf-8",
        )

    sequential = DataLoader(file_workers=1).load(folder)
    parallel = DataLoader(file_workers=2).load(folder)

    assert parallel.to_dict("records") == sequential.to_dict("records")


def test_sim_loader_reads_cst_txt_magnitude_export(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "sim" / "sim_grid_undated_coupler_s11_radius_sweep"
    folder.mkdir(parents=True)
    (folder / "s11_rc5209.txt").write_text(
        "#Parameters = {NumTune=0; r_c=52.09; w_c=19.3224}\n"
        '#"Frequency / GHz"\t"S1,1 (20) [Magnitude]"\n'
        "#-----------------------------------------\n"
        "2.6000000000000\t-0.0089373164991483\n",
        encoding="utf-8",
    )

    table = DataLoader().load(folder)
    row = table.iloc[0]

    assert row["dataset_id"] == "sim_grid_undated_coupler_s11_radius_sweep"
    assert row["source_file"] == "s11_rc5209.txt"
    assert row["freq_ghz"] == pytest.approx(2.6)
    assert row["s_name"] == "S11"
    assert row["s_db"] == pytest.approx(-0.0089373164991483)
    assert row["s_phase_deg"] == pytest.approx(0.0)
    assert row["source_format"] == "cst_txt_magnitude"
    assert row["sim_NumTune"] == 0
    assert row["sim_r_c"] == pytest.approx(52.09)
    assert row["sim_w_c"] == pytest.approx(19.3224)
