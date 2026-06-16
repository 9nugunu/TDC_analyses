from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.visualization.plot_config import PlotConfig
from deflector_tuning.visualization.s11_frequency_plots import plot_s11_with_markers


def _sparameter_table() -> pd.DataFrame:
    rows = []
    for source_file, tune_position, phase_offset in [
        ("0.5_processed.csv", 0.5, 0.0),
        ("1.5_processed.csv", 1.5, -20.0),
    ]:
        for freq_ghz, s_db, phase in [
            (2.84, -1.0, 10.0 + phase_offset),
            (2.856, -3.0, 20.0 + phase_offset),
            (2.872, -2.0, 30.0 + phase_offset),
        ]:
            rows.append(
                {
                    "dataset_id": "dataset",
                    "data_layer": "prepro",
                    "source_file": source_file,
                    "tune_position": tune_position,
                    "s_name": "S11",
                    "freq_ghz": freq_ghz,
                    "s_db": s_db,
                    "s_phase_deg": phase,
                    "source_format": "processed_csv_db_phase",
                }
            )
    return pd.DataFrame(rows)


def _marker_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "data_layer": "prepro",
                "source_file": "0.5_processed.csv",
                "tune_position": 0.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 20.0,
                "source_format": "processed_csv_db_phase",
            },
            {
                "dataset_id": "dataset",
                "data_layer": "prepro",
                "source_file": "1.5_processed.csv",
                "tune_position": 1.5,
                "s_name": "S11",
                "marker_name": "f_2pi3",
                "marker_role": "exp",
                "target_freq_ghz": 2.856,
                "freq_ghz": 2.856,
                "freq_error_ghz": 0.0,
                "s_db": -3.0,
                "s_phase_deg": 0.0,
                "source_format": "processed_csv_db_phase",
            },
        ]
    )


def test_plot_s11_with_markers_writes_overview_and_individual_pngs(tmp_path: Path) -> None:
    paths = plot_s11_with_markers(
        _sparameter_table(),
        _marker_points(),
        tmp_path,
        split_by_position=True,
        config=PlotConfig(dpi=120),
    )

    assert "overview" in paths
    assert "position_0p5" in paths
    assert "position_1p5" in paths
    assert paths["overview"].name == "plot_s11_with_markers.png"
    assert paths["position_0p5"].name == "plot_s11_position_0p5.png"
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_plot_s11_with_markers_accepts_grid_scan_without_tune_position(tmp_path: Path) -> None:
    sparameter_table = pd.DataFrame(
        [
            {
                "source_file": "run1.s1p",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            },
            {
                "source_file": "run1.s1p",
                "freq_ghz": 2.86,
                "s_db": -2.0,
                "s_phase_deg": 20.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            },
        ]
    )
    marker_points = pd.DataFrame(
        [
            {
                "source_file": "run1.s1p",
                "marker_name": "f_2pi3",
                "freq_ghz": 2.85,
                "s_db": -1.0,
                "s_phase_deg": 10.0,
                "sim_r_c": 56.59,
                "sim_w_c": 19.32,
            }
        ]
    )

    paths = plot_s11_with_markers(sparameter_table, marker_points, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == ["grid_r_c_56p59_w_c_19p32"]
    assert paths["grid_r_c_56p59_w_c_19p32"].name == "plot_s11_r_c_56p59_w_c_19p32.png"
    assert paths["grid_r_c_56p59_w_c_19p32"].exists()


def test_plot_s11_with_markers_rejects_empty_sparameter_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sparameter_table is empty"):
        plot_s11_with_markers(pd.DataFrame(), _marker_points(), tmp_path)
