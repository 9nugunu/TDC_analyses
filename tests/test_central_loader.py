from pathlib import Path

import pytest

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.records import DataFolder, DataKind
from deflector_tuning.data_loading.source_layer import DataLayer


def test_data_loader_returns_folder_for_sim_path() -> None:
    dataset = DataLoader().load_folder(Path("data/sim/sim_260605_sweep_case"))

    assert dataset == DataFolder(
        dataset_id="sim_260605_sweep_case",
        path=Path("data/sim/sim_260605_sweep_case"),
        data_layer=DataLayer.SIM,
        data_kind=DataKind.SIM,
    )


def test_data_loader_keeps_raw_data_as_experiment() -> None:
    dataset = DataLoader().load_folder(Path("data/raw/raw_260604_sweep_case"))

    assert dataset.data_layer is DataLayer.RAW
    assert dataset.data_kind is DataKind.EXP


def test_raw_touchstone_can_share_parser_later_without_losing_experiment_identity() -> None:
    dataset = DataLoader().load_folder(Path("data/raw/raw_260604_sweep_case/trace.s2p"))

    assert dataset.dataset_id == "raw_260604_sweep_case"
    assert dataset.data_kind is DataKind.EXP
    assert dataset.data_layer is DataLayer.RAW


def test_prepro_data_is_corrected_experiment_data_by_default() -> None:
    dataset = DataLoader().load_folder(Path("data/prepro/prepro_260415_sweep_case/cleaned.csv"))

    assert dataset.dataset_id == "prepro_260415_sweep_case"
    assert dataset.data_layer is DataLayer.PREPRO
    assert dataset.data_kind is DataKind.EXP


def test_loader_stops_for_dataset_id_outside_naming_rule() -> None:
    with pytest.raises(ValueError, match="Dataset id does not follow the naming rule"):
        DataLoader().load_folder(Path("data/sim/260605_case"))


def test_loader_stops_when_dataset_prefix_does_not_match_parent_layer() -> None:
    with pytest.raises(ValueError, match="does not match parent data/raw"):
        DataLoader().load_folder(Path("data/raw/sim_260605_sweep_case"))
