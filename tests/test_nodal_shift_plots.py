from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.nodal_shift_plots import plot_nodal_shift
from deflector_tuning.visualization.plot_config import PlotConfig


def _nodal_shift_table() -> pd.DataFrame:
    rows = []
    for r_c, w_c, f2_error, fpi_error in [
        (54.5, 18.5, 0.0, 4.0),
        (54.5, 18.75, 3.0, 5.0),
        (54.75, 18.5, 7.0, 2.0),
        (54.75, 18.75, 2.0, 1.0),
    ]:
        for marker_name, target, error in [
            ("f_2pi3", 240.0, f2_error),
            ("f_pi2", 180.0, -fpi_error),
        ]:
            rows.append(
                {
                    "dataset_id": "scan",
                    "marker_name": marker_name,
                    "position_family": "cell",
                    "from_tune_position": 0.5,
                    "to_tune_position": 1.5,
                    "phase_advance_0to360_deg": target + error,
                    "target_phase_advance_deg": target,
                    "phase_error_from_target_deg": error,
                    "abs_phase_error_from_target_deg": abs(error),
                    "sim_r_c": r_c,
                    "sim_w_c": w_c,
                }
            )
    return pd.DataFrame(rows)


def test_plot_nodal_shift_writes_bar_and_grid_objective_pngs(tmp_path: Path) -> None:
    paths = plot_nodal_shift(_nodal_shift_table(), tmp_path, config=PlotConfig(dpi=120))

    assert "nodal_shift_cell_bar" in paths
    assert "f_2pi3_abs_error" in paths
    assert "f_pi2_abs_error" in paths
    assert "combined_abs_error" in paths
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (tmp_path / "nodal_shift_grid_objective.csv").exists()
