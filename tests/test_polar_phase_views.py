from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from deflector_tuning.visualization.polar_phase_views import plot_marker_phase_polar_views
from deflector_tuning.visualization.plot_config import PlotConfig


def _marker_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "target_freq_ghz": 2.866,
                "freq_ghz": 2.866,
                "freq_error_ghz": 0.0,
                "s_db": -2.0,
                "s_phase_deg": -110.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_pi2",
                "marker_role": "exp",
                "target_freq_ghz": 2.876,
                "freq_ghz": 2.876,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 130.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -1.5,
                "s_phase_deg": 20.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_mean",
                "marker_role": "exp",
                "target_freq_ghz": 2.866,
                "freq_ghz": 2.866,
                "freq_error_ghz": 0.0,
                "s_db": -2.5,
                "s_phase_deg": -100.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "sample_dataset",
                "data_kind": "experiment",
                "data_layer": "prepro",
                "source_file": "1.0_processed.csv",
                "tune_position": 1.0,
                "port_side": None,
                "s_name": "S11",
                "marker_name": "f_pi2",
                "marker_role": "exp",
                "target_freq_ghz": 2.876,
                "freq_ghz": 2.876,
                "freq_error_ghz": 0.0,
                "s_db": -3.5,
                "s_phase_deg": 140.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )


def test_plot_marker_phase_polar_views_writes_per_position_and_overview_pngs(tmp_path: Path) -> None:
    paths = plot_marker_phase_polar_views(_marker_points(), tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["0.5", "1.0", "overview"]
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    assert paths["0.5"].name == "position_0p5.png"
    assert paths["overview"].name == "all_positions.png"
    assert plt.rcParams["font.sans-serif"][:4] == ["Pretendard", "Noto Sans", "Malgun Gothic", "DejaVu Sans"]


def test_plot_marker_phase_polar_views_rejects_empty_marker_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="marker_points is empty"):
        plot_marker_phase_polar_views(pd.DataFrame(), tmp_path)
