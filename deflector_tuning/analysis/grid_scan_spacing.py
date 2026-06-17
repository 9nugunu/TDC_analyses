"""Marker phase metrics for simulation 2D geometry scans."""

from __future__ import annotations

import math

import pandas as pd

MARKER_ORDER: tuple[str, str, str] = ("f_2pi3", "f_mean", "f_pi2")
REQUIRED_COLUMNS: tuple[str, ...] = ("data_kind", "source_file", "marker_name", "s_phase_deg", "sim_r_c", "sim_w_c")
OPTIONAL_GROUP_COLUMNS: tuple[str, ...] = ("run_id", "sim_NumDepth")
SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "source_file",
    "run_id",
    "sim_NumDepth",
    "sim_r_c",
    "sim_w_c",
    "marker_name",
    "phase_deg",
    "dphase_d_sim_r_c_deg_per_mm",
    "dphase_d_sim_w_c_deg_per_mm",
    "gradient_magnitude_deg_per_mm",
)


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


def summarize_marker_phase_sensitivity_for_grid_scan(marker_points: pd.DataFrame) -> pd.DataFrame:
    """Estimate local phase sensitivity to grid geometry parameters.

    Derivatives are finite differences of wrapped phase changes, reported in
    deg/mm for each marker at each available grid point. Interior points use a
    central difference; edge points use the nearest one-sided difference.
    """

    _validate_grid_marker_points(marker_points)
    table = marker_points.copy()
    rows: list[dict[str, object]] = []
    group_columns = [column for column in ("dataset_id", "sim_NumDepth") if column in table]
    if not group_columns:
        group_columns = ["_all"]
        table["_all"] = "all"
    for group_values, group in table.groupby(group_columns, dropna=False, sort=False):
        metadata = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else (group_values,), strict=True))
        metadata.pop("_all", None)
        rows.extend(_phase_sensitivity_rows(group, metadata))
    if not rows:
        raise ValueError("no complete marker phase groups found for grid-scan sensitivity")
    output = pd.DataFrame(rows)
    columns = [column for column in SENSITIVITY_COLUMNS if column in output]
    return output[columns].sort_values(["marker_name", "sim_r_c", "sim_w_c"], kind="mergesort").reset_index(drop=True)


def _validate_grid_marker_points(marker_points: pd.DataFrame) -> None:
    if marker_points.empty:
        raise ValueError("marker_points is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")
    sim_mask = marker_points["data_kind"].astype(str).str.lower().isin({"simulation", "sim"})
    if not sim_mask.all():
        raise ValueError("grid-scan maps are only defined for simulation marker points")


def _phase_sensitivity_rows(group: pd.DataFrame, metadata: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for marker_name, marker_group in group.groupby("marker_name", dropna=False, sort=False):
        points = _unique_marker_grid_points(marker_group)
        for point in points:
            dphase_dr = _finite_difference(points, point, varying_column="sim_r_c", fixed_column="sim_w_c")
            dphase_dw = _finite_difference(points, point, varying_column="sim_w_c", fixed_column="sim_r_c")
            rows.append(
                {
                    **metadata,
                    **_point_metadata(point),
                    "marker_name": marker_name,
                    "phase_deg": point["s_phase_deg"],
                    "dphase_d_sim_r_c_deg_per_mm": dphase_dr,
                    "dphase_d_sim_w_c_deg_per_mm": dphase_dw,
                    "gradient_magnitude_deg_per_mm": _gradient_magnitude(dphase_dr, dphase_dw),
                }
            )
    return rows


def _unique_marker_grid_points(marker_group: pd.DataFrame) -> list[dict[str, object]]:
    sort_columns = [column for column in ("sim_r_c", "sim_w_c", "source_file") if column in marker_group]
    sorted_group = marker_group.sort_values(sort_columns, kind="mergesort")
    deduped = sorted_group.drop_duplicates(["sim_r_c", "sim_w_c"], keep="first")
    return deduped.to_dict("records")


def _point_metadata(point: dict[str, object]) -> dict[str, object]:
    return {
        column: point[column]
        for column in ("source_file", "run_id", "sim_NumDepth", "sim_r_c", "sim_w_c")
        if column in point and not pd.isna(point[column])
    }


def _finite_difference(
    points: list[dict[str, object]],
    point: dict[str, object],
    *,
    varying_column: str,
    fixed_column: str,
) -> float:
    same_line = [
        candidate
        for candidate in points
        if _same_coordinate(candidate[fixed_column], point[fixed_column])
        and not _same_coordinate(candidate[varying_column], point[varying_column])
    ]
    if not same_line:
        return math.nan
    current_value = float(point[varying_column])
    lower = [candidate for candidate in same_line if float(candidate[varying_column]) < current_value]
    upper = [candidate for candidate in same_line if float(candidate[varying_column]) > current_value]
    lower_point = max(lower, key=lambda item: float(item[varying_column])) if lower else None
    upper_point = min(upper, key=lambda item: float(item[varying_column])) if upper else None
    if lower_point is not None and upper_point is not None:
        return _phase_slope(lower_point, upper_point, varying_column=varying_column)
    neighbor = lower_point if lower_point is not None else upper_point
    if neighbor is None:
        return math.nan
    return _phase_slope(neighbor, point, varying_column=varying_column)


def _phase_slope(start: dict[str, object], end: dict[str, object], *, varying_column: str) -> float:
    delta_u = float(end[varying_column]) - float(start[varying_column])
    if delta_u == 0.0:
        return math.nan
    return _wrap180(float(end["s_phase_deg"]) - float(start["s_phase_deg"])) / delta_u


def _same_coordinate(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _gradient_magnitude(dphase_dr: float, dphase_dw: float) -> float:
    if math.isnan(dphase_dr) or math.isnan(dphase_dw):
        return math.nan
    return math.hypot(dphase_dr, dphase_dw)


def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0
