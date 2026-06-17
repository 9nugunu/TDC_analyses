from pathlib import Path

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.loaders.prepro_loader import PreproLoader
from deflector_tuning.data_loading.loaders.raw_loader import RawLoader
from deflector_tuning.data_loading.loaders.sim_loader import SimLoader
from deflector_tuning.data_loading.records import DataKind


def test_data_loader_selects_sim_loader_for_sim_path() -> None:
    loader = DataLoader().select_loader(Path("data/sim/sim_sweep_260605_case"))

    assert isinstance(loader, SimLoader)


def test_data_loader_selects_raw_loader_for_raw_path() -> None:
    loader = DataLoader().select_loader(Path("data/raw/raw_sweep_260604_case"))

    assert isinstance(loader, RawLoader)


def test_data_loader_selects_prepro_loader_for_prepro_path() -> None:
    loader = DataLoader().select_loader(Path("data/prepro/prepro_sweep_260415_case"))

    assert isinstance(loader, PreproLoader)


def test_selected_loader_returns_same_simple_data_folder_record() -> None:
    dataset = DataLoader().load_folder(Path("data/raw/raw_sweep_260604_case/trace.s2p"))

    assert dataset.dataset_id == "raw_sweep_260604_case"
    assert dataset.data_kind is DataKind.EXP
