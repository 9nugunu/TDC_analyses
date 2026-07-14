from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.analysis.geometry_phase_response import compute_geometry_phase_response
from deflector_tuning.visualization.geometry_phase_response_plots import plot_geometry_phase_response
from deflector_tuning.visualization.plot_config import PlotConfig


def _marker_points() -> pd.DataFrame:
    rows = []
    phases = {
        (-1.0, 4.5): {"f_2pi3": 10.0, "f_mean": 40.0},
        (-1.0, 5.0): {"f_2pi3": 110.0, "f_mean": 130.0},
        (0.0, 4.5): {"f_2pi3": 20.0, "f_mean": 45.0},
        (0.0, 5.0): {"f_2pi3": 140.0, "f_mean": 135.0},
        (1.0, 4.5): {"f_2pi3": 30.0, "f_mean": 50.0},
        (1.0, 5.0): {"f_2pi3": 170.0, "f_mean": 150.0},
    }
    for (offset, tune_position), marker_phases in phases.items():
        for marker_name, phase in marker_phases.items():
            rows.append(
                {
                    "dataset_id": "sim_sweep",
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": f"run_{offset}_{tune_position}.s2p",
                    "marker_name": marker_name,
                    "marker_role": "sim",
                    "s_name": "S11",
                    "tune_position": tune_position,
                    "sim_NumDepth": tune_position,
                    "sim_offset_cell_03": offset,
                    "s_phase_deg": phase,
                }
            )
    return pd.DataFrame(rows)


def test_compute_geometry_phase_response_pairs_cell_and_iris_by_sweep_axis() -> None:
    response = compute_geometry_phase_response(_marker_points())

    f_2pi3 = response[response["marker_name"] == "f_2pi3"].sort_values("sweep_value")

    assert f_2pi3["sweep_axis"].unique().tolist() == ["sim_offset_cell_03"]
    assert f_2pi3["sweep_value"].tolist() == [-1.0, 0.0, 1.0]
    assert f_2pi3["sweep_base"].unique().tolist() == [0.0]
    assert f_2pi3["cell_iris_phase_delta_deg"].tolist() == pytest.approx([100.0, 120.0, 140.0])
    assert f_2pi3["cell_iris_delta_shift_deg"].tolist() == pytest.approx([-20.0, 0.0, 20.0])
    assert f_2pi3["cell_phase_shift_deg"].tolist() == pytest.approx([-10.0, 0.0, 10.0])
    assert f_2pi3["iris_phase_shift_deg"].tolist() == pytest.approx([-30.0, 0.0, 30.0])


def test_compute_geometry_phase_response_returns_empty_when_sweep_axis_is_ambiguous() -> None:
    marker_points = _marker_points()
    marker_points["sim_other_offset"] = marker_points["sim_offset_cell_03"] * 2.0

    response = compute_geometry_phase_response(marker_points)

    assert response.empty


def test_plot_geometry_phase_response_writes_absolute_and_family_pickup_figures(tmp_path: Path) -> None:
    response = compute_geometry_phase_response(_marker_points())

    paths = plot_geometry_phase_response(response, tmp_path, config=PlotConfig(dpi=120))

    assert list(paths) == [
        "cell_iris_absolute_phase",
        "cell_iris_phase_pickup",
        "f_2pi3_absolute_phase",
        "f_2pi3_phase_pickup",
        "f_mean_absolute_phase",
        "f_mean_phase_pickup",
    ]
    assert paths["cell_iris_absolute_phase"].name == "absolute_phase.png"
    assert paths["cell_iris_phase_pickup"].name == "phase_pickup.png"
    assert paths["f_2pi3_absolute_phase"].name == "f_2pi3_absolute_phase.png"
    assert paths["f_2pi3_phase_pickup"].name == "f_2pi3_phase_pickup.png"
    assert paths["cell_iris_absolute_phase"].exists()
    assert paths["cell_iris_phase_pickup"].exists()
    assert paths["f_2pi3_absolute_phase"].exists()
    assert paths["f_2pi3_phase_pickup"].exists()
