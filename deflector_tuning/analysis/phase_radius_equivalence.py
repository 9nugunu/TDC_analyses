"""Project measured S11 phase changes onto a simulated ``r_c`` line scan."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PhaseRadiusMappingError(ValueError):
    """Raised when a phase trace cannot define one equivalent ``r_c`` value."""


def fit_rc_states(
    simulation_phase_line: pd.DataFrame,
    raw_phase_observation: pd.DataFrame,
    *,
    design_r_c_mm: float,
    anchor_marker: str = "f_mean",
    step_um: float = 0.1,
) -> pd.DataFrame:
    """Fit baseline/current equivalent ``r_c`` coordinates on a 1D phase scan.

    The current coordinate is anchored first by adding the measured current
    ``f_mean`` phase to the simulated design-point phase.  With that coordinate
    fixed, the baseline coordinate is fitted to the phase changes of every
    measured marker.  These are simulation-equivalent coordinates, not a
    mechanical displacement calibration.
    """

    _require_columns(
        simulation_phase_line,
        ("sim_r_c", "marker_name", "s_phase_deg"),
        "simulation_phase_line",
    )
    _require_columns(
        raw_phase_observation,
        ("marker_name", "before_phase_deg", "after_phase_deg"),
        "raw_phase_observation",
    )
    if step_um <= 0.0:
        raise ValueError("step_um must be positive")

    observed = raw_phase_observation.loc[
        :, ["marker_name", "before_phase_deg", "after_phase_deg"]
    ].copy()
    for column in ("before_phase_deg", "after_phase_deg"):
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
    observed = observed.dropna()
    if observed["marker_name"].duplicated().any():
        raise ValueError("raw_phase_observation contains duplicate marker names")
    if observed.empty:
        raise ValueError("raw_phase_observation has no finite phase rows")

    anchor_rows = observed[observed["marker_name"] == anchor_marker]
    if len(anchor_rows) != 1:
        raise ValueError(f"Expected one observed {anchor_marker!r} row; found {len(anchor_rows)}")
    anchor_radii, anchor_phase = _marker_curve(simulation_phase_line, anchor_marker)
    if not anchor_radii.min() <= design_r_c_mm <= anchor_radii.max():
        raise PhaseRadiusMappingError(
            f"design_r_c_mm={design_r_c_mm:g} is outside the simulated anchor range"
        )
    design_phase = float(np.interp(design_r_c_mm, anchor_radii, anchor_phase))
    current_target_phase = design_phase + float(anchor_rows["after_phase_deg"].iloc[0])
    current_r_c_mm = _invert_monotonic_curve(
        anchor_radii,
        anchor_phase,
        current_target_phase,
        label=anchor_marker,
    )

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    common_min = -np.inf
    for marker_name in observed["marker_name"].astype(str):
        radii, phase = _marker_curve(simulation_phase_line, marker_name)
        curves[marker_name] = (radii, phase)
        common_min = max(common_min, float(radii.min()))
        if current_r_c_mm > radii.max() + 1e-12:
            raise PhaseRadiusMappingError(
                f"current equivalent r_c is outside the simulated {marker_name} range"
            )
    if common_min > current_r_c_mm:
        raise PhaseRadiusMappingError(
            "No simulated baseline range exists below the current equivalent r_c"
        )

    baseline_grid = _radius_grid(common_min, current_r_c_mm, step_um * 1e-3)
    observed_delta = _wrap180(
        observed["after_phase_deg"].to_numpy(dtype=float)
        - observed["before_phase_deg"].to_numpy(dtype=float)
    )
    predictions: list[np.ndarray] = []
    for marker_name in observed["marker_name"].astype(str):
        radii, phase = curves[marker_name]
        current_phase = float(np.interp(current_r_c_mm, radii, phase))
        predictions.append(_wrap180(current_phase - np.interp(baseline_grid, radii, phase)))
    predicted = np.column_stack(predictions)
    residual = _wrap180(predicted - observed_delta)
    objective = np.mean(np.square(residual), axis=1)
    best_index = int(np.argmin(objective))
    baseline_r_c_mm = float(baseline_grid[best_index])
    rms_residual_deg = float(np.sqrt(objective[best_index]))

    return pd.DataFrame(
        [
            {
                "state": "baseline",
                "r_c_mm": baseline_r_c_mm,
                "method": "three_marker_phase_change_fit",
                "rms_residual_deg": rms_residual_deg,
            },
            {
                "state": "current",
                "r_c_mm": current_r_c_mm,
                "method": f"{anchor_marker}_design_phase_anchor",
                "rms_residual_deg": np.nan,
            },
            {
                "state": "design",
                "r_c_mm": float(design_r_c_mm),
                "method": "campaign_design",
                "rms_residual_deg": np.nan,
            },
        ]
    )


def estimate_phase_radius_equivalence(
    simulation_phase_line: pd.DataFrame,
    observed_phase_change: pd.DataFrame,
    *,
    baseline_r_c_mm: float,
    min_delta_r_c_um: float = 0.0,
    max_delta_r_c_um: float = 250.0,
    step_um: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate one ``r_c`` shift from measured phase changes at shared markers.

    Absolute raw and simulated S11 phases may have different reference-plane
    offsets.  The fit therefore uses only ``after - before`` phase change at
    identical marker frequencies, and compares it with the simulated phase
    change relative to ``baseline_r_c_mm``.
    """

    _require_columns(simulation_phase_line, ("sim_r_c", "marker_name", "s_phase_deg"), "simulation_phase_line")
    _require_columns(observed_phase_change, ("marker_name", "phase_delta_deg"), "observed_phase_change")
    if max_delta_r_c_um < min_delta_r_c_um:
        raise ValueError("max_delta_r_c_um must be greater than or equal to min_delta_r_c_um")
    if step_um <= 0.0:
        raise ValueError("step_um must be positive")

    observed = observed_phase_change.loc[:, ["marker_name", "phase_delta_deg"]].copy()
    observed["phase_delta_deg"] = pd.to_numeric(observed["phase_delta_deg"], errors="coerce")
    observed = observed[np.isfinite(observed["phase_delta_deg"])].copy()
    if observed.empty:
        raise ValueError("observed_phase_change has no finite phase_delta_deg values")

    delta_grid_um = _delta_grid(min_delta_r_c_um, max_delta_r_c_um, step_um)
    curves = _simulated_phase_change_curves(
        simulation_phase_line,
        observed["marker_name"].tolist(),
        baseline_r_c_mm=baseline_r_c_mm,
        delta_grid_um=delta_grid_um,
    )
    prediction = curves.pivot(index="delta_r_c_um", columns="marker_name", values="sim_phase_delta_deg")
    marker_names = observed["marker_name"].tolist()
    missing = [marker for marker in marker_names if marker not in prediction]
    if missing:
        raise ValueError(f"simulation_phase_line has no usable phase curve for markers: {missing}")

    observed_by_marker = observed.set_index("marker_name")["phase_delta_deg"]
    residual = _wrap180(prediction[marker_names].to_numpy() - observed_by_marker[marker_names].to_numpy())
    objective = np.square(residual).sum(axis=1)
    shared_index = int(np.argmin(objective))
    shared_delta_um = float(prediction.index[shared_index])

    rows: list[dict[str, object]] = []
    for marker_name in marker_names:
        marker_curve = prediction[marker_name].to_numpy(dtype=float)
        marker_residual = _wrap180(marker_curve - float(observed_by_marker[marker_name]))
        marker_index = int(np.argmin(np.square(marker_residual)))
        rows.append(
            {
                "marker_name": marker_name,
                "phase_delta_deg": float(observed_by_marker[marker_name]),
                "marker_equivalent_delta_r_c_um": float(prediction.index[marker_index]),
                "equivalent_delta_r_c_um": shared_delta_um,
                "predicted_phase_delta_at_shared_fit_deg": float(marker_curve[shared_index]),
                "phase_residual_deg": float(marker_residual[shared_index]),
            }
        )
    return pd.DataFrame(rows), curves


def build_raw_anchored_phase_position(
    simulation_phase_line: pd.DataFrame,
    raw_phase_observation: pd.DataFrame,
    *,
    baseline_r_c_mm: float,
    target_r_c_mm: float,
    projection_max_r_c_mm: float | None = None,
    step_um: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Place a raw before/after measurement on a simulated ``r_c`` curve.

    The raw phase at the pre-tuning state is used as the vertical reference at
    ``baseline_r_c_mm``.  The simulation contributes only its phase *change*
    from that point, so a raw/simulation reference-plane phase offset is not
    interpreted as a geometry displacement.
    """

    _require_columns(simulation_phase_line, ("sim_r_c", "marker_name", "s_phase_deg"), "simulation_phase_line")
    _require_columns(
        raw_phase_observation,
        ("marker_name", "before_phase_deg", "after_phase_deg"),
        "raw_phase_observation",
    )
    if target_r_c_mm < baseline_r_c_mm:
        raise ValueError("target_r_c_mm must be greater than or equal to baseline_r_c_mm")

    observed = raw_phase_observation.loc[:, ["marker_name", "before_phase_deg", "after_phase_deg"]].copy()
    for column in ("before_phase_deg", "after_phase_deg"):
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
    observed = observed.dropna()
    if observed.empty:
        raise ValueError("raw_phase_observation has no finite before/after phase rows")
    observed["phase_delta_deg"] = _wrap180(observed["after_phase_deg"] - observed["before_phase_deg"])

    if projection_max_r_c_mm is None:
        upper_bounds = []
        for marker_name in observed["marker_name"]:
            marker_radii = pd.to_numeric(
                simulation_phase_line.loc[simulation_phase_line["marker_name"] == marker_name, "sim_r_c"],
                errors="coerce",
            )
            if marker_radii.notna().any():
                upper_bounds.append(float(marker_radii.max()))
        if not upper_bounds:
            raise ValueError("simulation_phase_line has no finite r_c values for the observed markers")
        projection_max_r_c_mm = min(upper_bounds)
    if projection_max_r_c_mm < baseline_r_c_mm:
        raise ValueError("projection_max_r_c_mm must be greater than or equal to baseline_r_c_mm")
    max_delta_um = (projection_max_r_c_mm - baseline_r_c_mm) * 1e3
    estimates, _ = estimate_phase_radius_equivalence(
        simulation_phase_line,
        observed.loc[:, ["marker_name", "phase_delta_deg"]],
        baseline_r_c_mm=baseline_r_c_mm,
        min_delta_r_c_um=0.0,
        max_delta_r_c_um=max_delta_um,
        step_um=step_um,
    )
    position = observed.merge(estimates, on=["marker_name", "phase_delta_deg"], validate="one_to_one")
    position["marker_equivalent_r_c_mm"] = baseline_r_c_mm + position["marker_equivalent_delta_r_c_um"] * 1e-3
    position["shared_equivalent_r_c_mm"] = baseline_r_c_mm + position["equivalent_delta_r_c_um"] * 1e-3
    position["remaining_to_target_um"] = (target_r_c_mm - position["shared_equivalent_r_c_mm"]) * 1e3

    curve_rows: list[dict[str, object]] = []
    for marker_name, observation in observed.set_index("marker_name").iterrows():
        marker = simulation_phase_line[simulation_phase_line["marker_name"] == marker_name].copy()
        marker["sim_r_c"] = pd.to_numeric(marker["sim_r_c"], errors="coerce")
        marker["s_phase_deg"] = pd.to_numeric(marker["s_phase_deg"], errors="coerce")
        marker = marker[np.isfinite(marker["sim_r_c"]) & np.isfinite(marker["s_phase_deg"])].sort_values("sim_r_c")
        marker = marker.drop_duplicates("sim_r_c", keep="last")
        if len(marker) < 2:
            raise ValueError(f"simulation_phase_line has no usable phase curve for marker: {marker_name}")
        radii = marker["sim_r_c"].to_numpy(dtype=float)
        if not radii.min() <= baseline_r_c_mm <= radii.max():
            raise ValueError(f"baseline_r_c_mm={baseline_r_c_mm:g} is outside the simulated {marker_name} range")
        if not radii.min() <= target_r_c_mm <= radii.max():
            raise ValueError(f"target_r_c_mm={target_r_c_mm:g} is outside the simulated {marker_name} range")
        phase = np.rad2deg(np.unwrap(np.deg2rad(marker["s_phase_deg"].to_numpy(dtype=float))))
        baseline_phase = float(np.interp(baseline_r_c_mm, radii, phase))
        raw_anchored_phase = float(observation["before_phase_deg"]) + phase - baseline_phase
        for radius, sim_phase, anchored_phase in zip(radii, phase, raw_anchored_phase, strict=True):
            curve_rows.append(
                {
                    "marker_name": marker_name,
                    "sim_r_c": float(radius),
                    "sim_unwrapped_phase_deg": float(sim_phase),
                    "raw_anchored_phase_deg": float(anchored_phase),
                }
            )
    return position, pd.DataFrame(curve_rows)


def build_experiment_simulation_phase_comparison(
    simulation_phase_line: pd.DataFrame,
    raw_phase_observation: pd.DataFrame,
    *,
    before_r_c_mm: float,
    current_r_c_mm: float,
) -> pd.DataFrame:
    """Compare raw and simulated phases at two candidate radius positions."""

    _require_columns(simulation_phase_line, ("sim_r_c", "marker_name", "s_phase_deg"), "simulation_phase_line")
    _require_columns(
        raw_phase_observation,
        ("marker_name", "before_phase_deg", "after_phase_deg"),
        "raw_phase_observation",
    )
    rows: list[dict[str, object]] = []
    for observation in raw_phase_observation.itertuples(index=False):
        marker_name = str(observation.marker_name)
        marker = simulation_phase_line[simulation_phase_line["marker_name"] == marker_name].copy()
        marker["sim_r_c"] = pd.to_numeric(marker["sim_r_c"], errors="coerce")
        marker["s_phase_deg"] = pd.to_numeric(marker["s_phase_deg"], errors="coerce")
        marker = marker[np.isfinite(marker["sim_r_c"]) & np.isfinite(marker["s_phase_deg"])].sort_values("sim_r_c")
        marker = marker.drop_duplicates("sim_r_c", keep="last")
        if len(marker) < 2:
            raise ValueError(f"simulation_phase_line has no usable phase curve for marker: {marker_name}")
        radii = marker["sim_r_c"].to_numpy(dtype=float)
        if before_r_c_mm < radii.min() or current_r_c_mm > radii.max():
            raise ValueError(f"candidate r_c positions are outside the simulated {marker_name} range")
        phase = np.rad2deg(np.unwrap(np.deg2rad(marker["s_phase_deg"].to_numpy(dtype=float))))
        sim_before = float(np.interp(before_r_c_mm, radii, phase))
        sim_current = float(np.interp(current_r_c_mm, radii, phase))
        measured_before = float(observation.before_phase_deg)
        measured_current = float(observation.after_phase_deg)
        measured_delta = float(_wrap180(measured_current - measured_before))
        simulated_delta = float(_wrap180(sim_current - sim_before))
        rows.append(
            {
                "marker_name": marker_name,
                "measured_before_phase_deg": measured_before,
                "simulated_before_phase_deg": float(_wrap180(sim_before)),
                "measured_current_phase_deg": measured_current,
                "simulated_current_phase_deg": float(_wrap180(sim_current)),
                "measured_phase_delta_deg": measured_delta,
                "simulated_phase_delta_deg": simulated_delta,
                "delta_residual_deg": simulated_delta - measured_delta,
                "before_r_c_mm": before_r_c_mm,
                "current_r_c_mm": current_r_c_mm,
            }
        )
    return pd.DataFrame(rows)


def _simulated_phase_change_curves(
    simulation_phase_line: pd.DataFrame,
    marker_names: list[str],
    *,
    baseline_r_c_mm: float,
    delta_grid_um: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_radii = baseline_r_c_mm + delta_grid_um * 1e-3
    for marker_name in marker_names:
        marker = simulation_phase_line[simulation_phase_line["marker_name"] == marker_name].copy()
        marker["sim_r_c"] = pd.to_numeric(marker["sim_r_c"], errors="coerce")
        marker["s_phase_deg"] = pd.to_numeric(marker["s_phase_deg"], errors="coerce")
        marker = marker[np.isfinite(marker["sim_r_c"]) & np.isfinite(marker["s_phase_deg"])].sort_values("sim_r_c")
        marker = marker.drop_duplicates("sim_r_c", keep="last")
        if len(marker) < 2:
            continue
        radii = marker["sim_r_c"].to_numpy(dtype=float)
        if baseline_r_c_mm < radii.min() - 1e-12 or baseline_r_c_mm > radii.max() + 1e-12:
            raise ValueError(f"baseline_r_c_mm={baseline_r_c_mm:g} is outside the simulated {marker_name} range")
        if target_radii.min() < radii.min() - 1e-12 or target_radii.max() > radii.max() + 1e-12:
            raise ValueError(f"requested r_c projection range is outside the simulated {marker_name} line")
        phase_unwrapped_deg = np.rad2deg(np.unwrap(np.deg2rad(marker["s_phase_deg"].to_numpy(dtype=float))))
        baseline_phase = float(np.interp(baseline_r_c_mm, radii, phase_unwrapped_deg))
        predicted_phase = np.interp(np.clip(target_radii, radii.min(), radii.max()), radii, phase_unwrapped_deg)
        for delta_um, phase_delta in zip(delta_grid_um, predicted_phase - baseline_phase, strict=True):
            rows.append(
                {
                    "marker_name": marker_name,
                    "delta_r_c_um": float(delta_um),
                    "sim_phase_delta_deg": float(_wrap180(phase_delta)),
                }
            )
    return pd.DataFrame(rows)


def _delta_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    count = int(round((maximum - minimum) / step))
    grid = minimum + np.arange(count + 1, dtype=float) * step
    if grid[-1] < maximum:
        grid = np.append(grid, maximum)
    else:
        grid[-1] = maximum
    return grid


def _radius_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    count = max(0, int(np.floor((maximum - minimum) / step + 1e-9)))
    grid = minimum + np.arange(count + 1, dtype=float) * step
    if grid[-1] < maximum - 1e-12:
        grid = np.append(grid, maximum)
    else:
        grid[-1] = maximum
    return grid


def _marker_curve(
    simulation_phase_line: pd.DataFrame,
    marker_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    marker = simulation_phase_line[
        simulation_phase_line["marker_name"].astype(str) == marker_name
    ].copy()
    marker["sim_r_c"] = pd.to_numeric(marker["sim_r_c"], errors="coerce")
    marker["s_phase_deg"] = pd.to_numeric(marker["s_phase_deg"], errors="coerce")
    marker = marker[np.isfinite(marker["sim_r_c"]) & np.isfinite(marker["s_phase_deg"])]
    marker = marker.sort_values("sim_r_c").drop_duplicates("sim_r_c", keep="last")
    if len(marker) < 2:
        raise ValueError(f"simulation_phase_line has no usable phase curve for marker: {marker_name}")
    radii = marker["sim_r_c"].to_numpy(dtype=float)
    phase = np.rad2deg(np.unwrap(np.deg2rad(marker["s_phase_deg"].to_numpy(dtype=float))))
    return radii, phase


def _invert_monotonic_curve(
    radii: np.ndarray,
    phase: np.ndarray,
    target_phase: float,
    *,
    label: str,
) -> float:
    differences = np.diff(phase)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise PhaseRadiusMappingError(
            f"Simulated {label} phase is not monotonic in r_c"
        )
    if not phase.min() <= target_phase <= phase.max():
        raise PhaseRadiusMappingError(
            f"Measured current {label} phase maps outside the simulated r_c range"
        )
    order = np.argsort(phase)
    return float(np.interp(target_phase, phase[order], radii[order]))


def _wrap180(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], table_name: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
