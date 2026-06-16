from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.visualization.phase_advance_plots import plot_phase_advance
from deflector_tuning.visualization.plot_config import PlotConfig


def _phase_advance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_source_file": "0.5_processed.csv",
                "to_source_file": "1.0_processed.csv",
                "from_tune_position": 0.5,
                "to_tune_position": 1.0,
                "phase_advance_0to360_deg": 240.0,
                "phase_error_from_240_deg": 0.0,
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_source_file": "1.0_processed.csv",
                "to_source_file": "1.5_processed.csv",
                "from_tune_position": 1.0,
                "to_tune_position": 1.5,
                "phase_advance_0to360_deg": 260.0,
                "phase_error_from_240_deg": 20.0,
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_source_file": "0.5_processed.csv",
                "to_source_file": "1.0_processed.csv",
                "from_tune_position": 0.5,
                "to_tune_position": 1.0,
                "phase_advance_0to360_deg": 230.0,
                "phase_error_from_240_deg": -10.0,
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "port_side": None,
                "s_name": "S11",
                "from_source_file": "1.0_processed.csv",
                "to_source_file": "1.5_processed.csv",
                "from_tune_position": 1.0,
                "to_tune_position": 1.5,
                "phase_advance_0to360_deg": 210.0,
                "phase_error_from_240_deg": -30.0,
            },
        ]
    )


def test_plot_phase_advance_writes_advance_and_error_pngs(tmp_path: Path) -> None:
    paths = plot_phase_advance(_phase_advance_table(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["phase_advance", "phase_error"]
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    assert paths["phase_advance"].name == "phase_advance_by_marker.png"
    assert paths["phase_error"].name == "phase_error_by_marker.png"


def test_plot_phase_advance_can_write_individual_position_family_pngs(tmp_path: Path) -> None:
    table = _phase_advance_table().copy()
    table["position_family"] = ["cell", "cell", "iris", "iris"]

    paths = plot_phase_advance(table, tmp_path, split_by_family=True, config=PlotConfig(dpi=120))

    assert "phase_advance_cell" in paths
    assert "phase_error_cell" in paths
    assert "phase_advance_iris" in paths
    assert "phase_error_iris" in paths
    assert paths["phase_advance_cell"].name == "phase_advance_cell.png"
    assert paths["phase_error_iris"].name == "phase_error_iris.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_phase_advance_can_exclude_edge_transitions(tmp_path: Path) -> None:
    phase_table = pd.concat(
        [
            _phase_advance_table(),
            pd.DataFrame(
                [
                    {
                        "dataset_id": "sample_dataset",
                        "data_kind": "experiment",
                        "data_layer": "prepro",
                        "marker_name": "f_2pi3",
                        "marker_role": "exp",
                        "port_side": None,
                        "s_name": "S11",
                        "from_source_file": "1.5_processed.csv",
                        "to_source_file": "2.0_processed.csv",
                        "from_tune_position": 1.5,
                        "to_tune_position": 2.0,
                        "phase_advance_0to360_deg": 180.0,
                        "phase_error_from_240_deg": -60.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    paths = plot_phase_advance(phase_table, tmp_path, transition_scope="internal", config=PlotConfig(dpi=120))

    assert paths["phase_advance"].exists()
    assert paths["phase_error"].exists()


def test_plot_phase_advance_skips_family_split_when_positions_are_missing(tmp_path: Path) -> None:
    table = _phase_advance_table().assign(
        from_tune_position=pd.NA,
        to_tune_position=pd.NA,
        position_family="offset_nan",
    )

    paths = plot_phase_advance(table, tmp_path, split_by_family=True, config=PlotConfig(dpi=120))

    assert list(paths) == ["phase_advance", "phase_error"]
    assert paths["phase_advance"].exists()
    assert paths["phase_error"].exists()


def test_plot_phase_advance_rejects_empty_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_advance is empty"):
        plot_phase_advance(pd.DataFrame(), tmp_path)
