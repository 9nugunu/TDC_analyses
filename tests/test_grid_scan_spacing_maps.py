from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.analysis.grid_scan_spacing import (
    summarize_marker_phase_sensitivity_for_grid_scan,
    summarize_marker_spacing_for_grid_scan,
)
from deflector_tuning.visualization.plot_config import (
    BEST_MARKER_COLOR,
    DEFAULT_DESIGN_POINT_BY_AXIS,
    DESIGN_REFERENCE_LINEWIDTH,
    REFERENCE_GUIDE_COLOR,
    REFERENCE_GUIDE_LINESTYLE,
)
from deflector_tuning.visualization.grid_scan_spacing_maps import (
    _default_design_point,
    plot_grid_scan_phase_sensitivity_maps,
    plot_grid_scan_spacing_error_maps,
)


def _marker_points() -> pd.DataFrame:
    rows = []
    phases_by_run = {
        1: (0.0, 60.0, 120.0),
        2: (0.0, 55.0, 125.0),
        3: (0.0, 50.0, 118.0),
        4: (0.0, 62.0, 119.0),
    }
    coords_by_run = {
        1: (54.5, 18.5),
        2: (54.5, 18.75),
        3: (54.75, 18.5),
        4: (54.75, 18.75),
    }
    for run_id, phases in phases_by_run.items():
        r_c, w_c = coords_by_run[run_id]
        for marker_name, phase in zip(["f_2pi3", "f_mean", "f_pi2"], phases, strict=True):
            rows.append(
                {
                    "dataset_id": "scan",
                    "data_kind": "simulation",
                    "data_layer": "sim",
                    "source_file": f"run_{run_id}.s1p",
                    "run_id": run_id,
                    "sim_r_c": r_c,
                    "sim_w_c": w_c,
                    "marker_name": marker_name,
                    "s_phase_deg": phase,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_marker_spacing_for_grid_scan_computes_two_requested_errors() -> None:
    summary = summarize_marker_spacing_for_grid_scan(_marker_points())

    row = summary.loc[summary["run_id"] == 2].iloc[0]
    assert row["spacing_fmean_minus_f2pi3_deg"] == 55.0
    assert row["spacing_fpi2_minus_fmean_deg"] == 70.0
    assert row["spacing_equality_error_deg"] == 15.0
    assert row["spacing_60deg_target_error_deg"] == 15.0
    assert set(summary["sim_r_c"]) == {54.5, 54.75}
    assert set(summary["sim_w_c"]) == {18.5, 18.75}


def test_summarize_marker_phase_sensitivity_for_grid_scan_computes_wrapped_finite_differences() -> None:
    marker_points = _marker_points()
    marker_points.loc[
        (marker_points["run_id"] == 1) & (marker_points["marker_name"] == "f_2pi3"),
        "s_phase_deg",
    ] = 170.0
    marker_points.loc[
        (marker_points["run_id"] == 3) & (marker_points["marker_name"] == "f_2pi3"),
        "s_phase_deg",
    ] = -170.0

    summary = summarize_marker_phase_sensitivity_for_grid_scan(marker_points)

    row = summary[(summary["run_id"] == 1) & (summary["marker_name"] == "f_2pi3")].iloc[0]
    assert row["phase_deg"] == 170.0
    assert row["dphase_d_sim_r_c_deg_per_mm"] == pytest.approx(80.0)
    assert row["dphase_d_sim_w_c_deg_per_mm"] == pytest.approx(-680.0)
    assert row["gradient_magnitude_deg_per_mm"] == pytest.approx((80.0**2 + 680.0**2) ** 0.5)


def test_plot_grid_scan_spacing_error_maps_writes_only_requested_2d_error_pngs(tmp_path: Path) -> None:
    summary = summarize_marker_spacing_for_grid_scan(_marker_points())

    paths = plot_grid_scan_spacing_error_maps(summary, tmp_path)

    assert list(paths) == ["spacing_60deg_target_error", "spacing_equality_error"]
    assert paths["spacing_60deg_target_error"].name == "spacing_60deg_target_error.png"
    assert paths["spacing_equality_error"].name == "spacing_equality_error.png"
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (tmp_path / "grid_scan_spacing_summary.csv").exists()


def test_plot_grid_scan_phase_sensitivity_maps_writes_marker_derivative_pngs(tmp_path: Path) -> None:
    summary = summarize_marker_phase_sensitivity_for_grid_scan(_marker_points())

    paths = plot_grid_scan_phase_sensitivity_maps(summary, tmp_path)

    assert "f_2pi3_dphase_d_sim_r_c" in paths
    assert "f_mean_dphase_d_sim_w_c" in paths
    assert "f_pi2_gradient_magnitude" in paths
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (tmp_path / "grid_scan_phase_sensitivity_summary.csv").exists()


def test_plot_grid_scan_spacing_error_maps_reports_non_finite_summary_values(tmp_path: Path) -> None:
    summary = summarize_marker_spacing_for_grid_scan(_marker_points())
    summary.loc[0, "spacing_equality_error_deg"] = float("inf")

    with pytest.raises(ValueError, match=r"Non-finite plotting values in grid_scan spacing_summary"):
        plot_grid_scan_spacing_error_maps(summary, tmp_path)


def test_grid_scan_spacing_error_maps_use_requested_visual_reference_points() -> None:
    assert BEST_MARKER_COLOR == "#c51b7d"
    assert REFERENCE_GUIDE_COLOR == "0.45"
    assert REFERENCE_GUIDE_LINESTYLE == "--"
    assert DESIGN_REFERENCE_LINEWIDTH == 0.9
    assert DEFAULT_DESIGN_POINT_BY_AXIS == {"sim_r_c": 56.59, "sim_w_c": 19.3224}
    assert _default_design_point("sim_r_c", "sim_w_c") == (56.59, 19.3224)
    assert _default_design_point("custom_x", "sim_w_c") is None
