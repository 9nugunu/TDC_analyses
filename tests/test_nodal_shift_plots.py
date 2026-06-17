from pathlib import Path

import pandas as pd

from deflector_tuning.visualization.nodal_shift_plots import _cumulative_phase_table, plot_nodal_shift
from deflector_tuning.visualization.plot_config import PlotConfig


def _nodal_shift_table() -> pd.DataFrame:
    rows = []
    for r_c, w_c, from_position, to_position, f2_error, fpi_error in [
        (54.5, 18.5, 0.5, 1.5, 80.0, 70.0),
        (54.5, 18.5, 1.5, 2.5, 0.0, 4.0),
        (54.5, 18.5, 2.5, 3.5, 75.0, 65.0),
        (54.5, 18.75, 0.5, 1.5, 81.0, 71.0),
        (54.5, 18.75, 1.5, 2.5, 3.0, 5.0),
        (54.5, 18.75, 2.5, 3.5, 76.0, 66.0),
        (54.75, 18.5, 0.5, 1.5, 82.0, 72.0),
        (54.75, 18.5, 1.5, 2.5, 7.0, 2.0),
        (54.75, 18.5, 2.5, 3.5, 77.0, 67.0),
        (54.75, 18.75, 0.5, 1.5, 83.0, 73.0),
        (54.75, 18.75, 1.5, 2.5, 2.0, 1.0),
        (54.75, 18.75, 2.5, 3.5, 78.0, 68.0),
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
                    "from_tune_position": from_position,
                    "to_tune_position": to_position,
                    "phase_advance_0to360_deg": target + error,
                    "target_phase_advance_deg": target,
                    "phase_error_from_target_deg": error,
                    "abs_phase_error_from_target_deg": abs(error),
                    "sim_r_c": r_c,
                    "sim_w_c": w_c,
                }
            )
    return pd.DataFrame(rows)


def _iris_nodal_shift_table() -> pd.DataFrame:
    rows = []
    for from_position, to_position, f2_error, fpi_error in [
        (0.0, 1.0, 80.0, 70.0),
        (1.0, 2.0, 0.0, 4.0),
        (2.0, 3.0, 75.0, 65.0),
    ]:
        for marker_name, target, error in [
            ("f_2pi3", 240.0, f2_error),
            ("f_pi2", 180.0, -fpi_error),
        ]:
            rows.append(
                {
                    "dataset_id": "raw_sweep_260604_iris_portE",
                    "marker_name": marker_name,
                    "position_family": "iris",
                    "from_tune_position": from_position,
                    "to_tune_position": to_position,
                    "phase_advance_0to360_deg": target + error,
                    "target_phase_advance_deg": target,
                    "phase_error_from_target_deg": error,
                    "abs_phase_error_from_target_deg": abs(error),
                }
            )
    return pd.DataFrame(rows)


def test_plot_nodal_shift_writes_bar_and_grid_objective_pngs(tmp_path: Path) -> None:
    paths = plot_nodal_shift(_nodal_shift_table(), tmp_path, config=PlotConfig(dpi=120))

    assert "nodal_shift_cell_bar" in paths
    assert "f_2pi3_abs_error" in paths
    assert "f_pi2_abs_error" in paths
    assert "combined_abs_error" in paths
    assert "regular_cell_f_2pi3_signed_error" in paths
    assert "regular_cell_f_2pi3_phase_movement" in paths
    assert "regular_cell_f_2pi3_cumulative_phase" in paths
    assert "regular_cell_f_2pi3_cumulative_error" in paths
    assert "regular_cell_f_pi2_signed_error" in paths
    assert "regular_cell_f_pi2_phase_movement" in paths
    assert "regular_cell_f_pi2_cumulative_phase" in paths
    assert "regular_cell_f_pi2_cumulative_error" in paths
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (tmp_path / "nodal_shift_grid_objective.csv").exists()
    regular = pd.read_csv(tmp_path / "regular_cell_nodal_shift.csv")
    assert set(regular["from_tune_position"]) == {1.5}
    assert set(regular["to_tune_position"]) == {2.5}


def test_plot_nodal_shift_writes_regular_iris_outputs(tmp_path: Path) -> None:
    paths = plot_nodal_shift(_iris_nodal_shift_table(), tmp_path, config=PlotConfig(dpi=120))

    assert "nodal_shift_iris_bar" in paths
    assert "regular_iris_f_2pi3_signed_error" in paths
    assert "regular_iris_f_2pi3_phase_movement" in paths
    assert "regular_iris_f_2pi3_cumulative_phase" in paths
    assert "regular_iris_f_2pi3_cumulative_error" in paths
    assert "regular_iris_f_pi2_signed_error" in paths
    assert "regular_iris_f_pi2_phase_movement" in paths
    assert "regular_iris_f_pi2_cumulative_phase" in paths
    assert "regular_iris_f_pi2_cumulative_error" in paths
    for key in (
        "regular_iris_f_2pi3_signed_error",
        "regular_iris_f_2pi3_phase_movement",
        "regular_iris_f_2pi3_cumulative_phase",
        "regular_iris_f_2pi3_cumulative_error",
        "regular_iris_f_pi2_signed_error",
        "regular_iris_f_pi2_phase_movement",
        "regular_iris_f_pi2_cumulative_phase",
        "regular_iris_f_pi2_cumulative_error",
    ):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0
    regular = pd.read_csv(tmp_path / "regular_iris_nodal_shift.csv")
    assert set(regular["from_tune_position"]) == {1.0}
    assert set(regular["to_tune_position"]) == {2.0}


def test_regular_phase_movement_plots_phase_advance_with_target_lines(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, tuple[str, str, float | None]] = {}

    def _capture_marker_bar(
        table: pd.DataFrame,
        output_path: Path,
        *,
        marker: str,
        family: str,
        value_column: str,
        ylabel: str,
        title: str,
        reference_line: float | None,
        config: PlotConfig,
    ) -> Path:
        del table, marker, family, title, config
        output_path.write_text("plot", encoding="utf-8")
        calls[output_path.name] = (value_column, ylabel, reference_line)
        return output_path

    monkeypatch.setattr(
        "deflector_tuning.visualization.nodal_shift_plots._plot_marker_bar",
        _capture_marker_bar,
    )

    plot_nodal_shift(_iris_nodal_shift_table(), tmp_path, config=PlotConfig(dpi=120))

    assert calls["regular_iris_f_2pi3_signed_error.png"] == (
        "phase_error_from_target_deg",
        "Phase error from target [deg]",
        0.0,
    )
    assert calls["regular_iris_f_2pi3_phase_movement.png"] == (
        "phase_advance_0to360_deg",
        "Phase movement [deg]",
        240.0,
    )
    assert calls["regular_iris_f_pi2_phase_movement.png"] == (
        "phase_advance_0to360_deg",
        "Phase movement [deg]",
        180.0,
    )


def test_cumulative_phase_table_accumulates_measured_and_target_phase() -> None:
    table = pd.DataFrame(
        {
            "transition_label": ["1.0->2.0", "2.0->3.0", "3.0->4.0"],
            "_from_sort": [1.0, 2.0, 3.0],
            "_to_sort": [2.0, 3.0, 4.0],
            "phase_advance_0to360_deg": [250.0, 230.0, 245.0],
            "target_phase_advance_deg": [240.0, 240.0, 240.0],
        }
    )

    cumulative = _cumulative_phase_table(table)

    assert cumulative["measured_cumulative_phase_deg"].tolist() == [250.0, 480.0, 725.0]
    assert cumulative["target_cumulative_phase_deg"].tolist() == [240.0, 480.0, 720.0]
    assert cumulative["target_minus_measured_cumulative_phase_deg"].tolist() == [-10.0, 0.0, -5.0]
