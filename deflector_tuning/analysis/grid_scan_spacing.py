"""Marker phase-spacing metrics for simulation 2D geometry scans."""

from __future__ import annotations

import pandas as pd

MARKER_ORDER: tuple[str, str, str] = ("f_2pi3", "f_mean", "f_pi2")
REQUIRED_COLUMNS: tuple[str, ...] = ("data_kind", "source_file", "marker_name", "s_phase_deg", "sim_r_c", "sim_w_c")
OPTIONAL_GROUP_COLUMNS: tuple[str, ...] = ("run_id", "sim_NumDepth")


def summarize_marker_spacing_for_grid_scan(marker_points: pd.DataFrame) -> pd.DataFrame:
    """Summarize three-marker phase spacing for simulation r_c x w_c scans.

    The output is one row per simulated run/geometry point and includes only the
    two requested objective/error metrics:
    - spacing_60deg_target_error_deg: |d21|-60 plus |d32|-60 absolute errors
    - spacing_equality_error_deg: absolute difference between the two spacings
    """

    if marker_points.empty:
        raise ValueError("marker_points is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    table = marker_points.copy()
    sim_mask = table["data_kind"].astype(str).str.lower().isin({"simulation", "sim"})
    if not sim_mask.all():
        raise ValueError("grid-scan spacing maps are only defined for simulation marker points")

    group_columns = [column for column in ["dataset_id", "source_file", *OPTIONAL_GROUP_COLUMNS, "sim_r_c", "sim_w_c"] if column in table]
    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(group_columns, dropna=False, sort=False):
        phases = {row["marker_name"]: float(row["s_phase_deg"]) for _, row in group.iterrows()}
        if not all(marker in phases for marker in MARKER_ORDER):
            continue
        metadata = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else (group_values,), strict=True))
        spacing_21 = _wrap180(phases["f_mean"] - phases["f_2pi3"])
        spacing_32 = _wrap180(phases["f_pi2"] - phases["f_mean"])
        abs_21 = abs(spacing_21)
        abs_32 = abs(spacing_32)
        rows.append(
            {
                **metadata,
                "spacing_fmean_minus_f2pi3_deg": spacing_21,
                "spacing_fpi2_minus_fmean_deg": spacing_32,
                "spacing_equality_error_deg": abs(abs_21 - abs_32),
                "spacing_60deg_target_error_deg": abs(abs_21 - 60.0) + abs(abs_32 - 60.0),
            }
        )
    if not rows:
        raise ValueError("no complete f_2pi3/f_mean/f_pi2 marker groups found for grid-scan spacing")
    return pd.DataFrame(rows).sort_values(["sim_r_c", "sim_w_c"], kind="mergesort").reset_index(drop=True)


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0
