"""Unavailable phase must not enable phase analysis or plotting."""

import numpy as np
import pandas as pd

from deflector_tuning.workflows.mode_detection import detect_analysis_modes, detection_report


def _markers(phases):
    return pd.DataFrame([
        {"data_kind": "simulation", "sim_r_c": radius, "sim_w_c": 19.3224,
         "marker_name": marker, "s_phase_deg": phase, "s_db": -6.0}
        for radius in (56.5, 56.6)
        for marker, phase in zip(("f_2pi3", "f_mean", "f_pi2"), phases)
    ])


def test_magnitude_only_data_disables_phase_modes_and_explains_why():
    tables = {"marker_pts": _markers([np.nan] * 3)}
    assert detect_analysis_modes(tables, dataset_category="grid") == ("marker_analysis", "s11")
    report = detection_report(tables, dataset_category="grid")
    assert not report["phase_measurements"]["enabled"]
    assert "finite" in report["phase_measurements"]["reason"]
    assert not report["grid_scan_spacing"]["enabled"]


def test_finite_phase_still_enables_supported_modes():
    modes = detect_analysis_modes({"marker_pts": _markers([0, 60, 120])}, dataset_category="grid")
    assert {"polar", "phase_advance", "nodal_shift", "grid_scan_spacing"}.issubset(modes)


def test_grid_requires_complete_phase_capable_markers():
    tables = {"marker_pts": _markers([0, np.nan, 120])}
    assert "polar" in detect_analysis_modes(tables, dataset_category="grid")
    assert "grid_scan_spacing" not in detect_analysis_modes(tables, dataset_category="grid")


def test_grid_marker_union_cannot_substitute_for_complete_independent_runs():
    points = _markers([0, 60, 120])
    points["source_file"] = ["a.s2p"] * 3 + ["b.s2p"] * 3
    points.loc[[2, 3], "s_phase_deg"] = np.nan
    assert "grid_scan_spacing" not in detect_analysis_modes(
        {"marker_pts": points}, dataset_category="grid",
    )
