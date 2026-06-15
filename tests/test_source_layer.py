from pathlib import Path

import pytest

from deflector_tuning.data_loading.source_layer import DataLayer, detect_data_layer


def test_sim_folder_maps_to_sim_layer() -> None:
    assert detect_data_layer(Path("data/sim/260605_case")) is DataLayer.SIM


def test_raw_folder_maps_to_raw_layer() -> None:
    assert detect_data_layer(Path("data/raw/260604_case")) is DataLayer.RAW


def test_prepro_folder_maps_to_prepro_layer() -> None:
    assert detect_data_layer(Path("data/prepro/260415_case")) is DataLayer.PREPRO


def test_prepro_policy_is_csv_or_corrected_experiment_data() -> None:
    assert DataLayer.PREPRO.note == "corrected or csv-style experiment data"


def test_unknown_folder_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Expected path under data/sim, data/raw, or data/prepro"):
        detect_data_layer(Path("notes/random"))
