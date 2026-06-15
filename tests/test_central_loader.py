from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.records import DataFolder, DataKind
from deflector_tuning.data_loading.source_layer import DataLayer


def test_data_loader_returns_folder_for_sim_path() -> None:
    dataset = DataLoader().load_folder(Path("data/sim/260605_case"))

    assert dataset == DataFolder(
        dataset_id="260605_case",
        path=Path("data/sim/260605_case"),
        data_layer=DataLayer.SIM,
        data_kind=DataKind.SIM,
    )


def test_data_loader_keeps_raw_data_as_experiment() -> None:
    dataset = DataLoader().load_folder(Path("data/raw/260604_case"))

    assert dataset.data_layer is DataLayer.RAW
    assert dataset.data_kind is DataKind.EXP


def test_raw_touchstone_can_share_parser_later_without_losing_experiment_identity() -> None:
    dataset = DataLoader().load_folder(Path("data/raw/260604_case/trace.s2p"))

    assert dataset.dataset_id == "260604_case"
    assert dataset.data_kind is DataKind.EXP
    assert dataset.data_layer is DataLayer.RAW


def test_prepro_data_is_corrected_experiment_data_by_default() -> None:
    dataset = DataLoader().load_folder(Path("data/prepro/260415_case/cleaned.csv"))

    assert dataset.dataset_id == "260415_case"
    assert dataset.data_layer is DataLayer.PREPRO
    assert dataset.data_kind is DataKind.EXP
