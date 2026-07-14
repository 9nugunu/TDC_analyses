"""Coupler-cavity beta, frequency, and external-Q estimates from reflection phases."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from deflector_tuning.analysis.coupler_position_gate import (
    ALLOWED_COUPLER_CAVITY_TRANSITIONS,
    coupler_cavity_endpoint_metadata,
)

ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "tune_position",
    "cpl_pair",
    "cpl_pos_basis",
    "cpl_pos_from",
    "cpl_pos_to",
    "port_side",
    "s_name",
    "marker_role",
    "run_id",
    "scan_type",
)
REFERENCE_MARKER_NAME = "f_pi2"
OPERATION_MARKER_NAME = "f_2pi3"
PHASE_INPUT_CONVENTION = "raw_s11_reflection_phase_deg"
COUPLING_K_COLUMNS: tuple[str, ...] = ("coupling_k", "k", "cell_coupling_k")
OUTPUT_COLUMNS: list[str] = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "tune_position",
    "cpl_pair",
    "cpl_pos_basis",
    "cpl_pos_from",
    "cpl_pos_to",
    "port_side",
    "s_name",
    "marker_role",
    "ref_marker",
    "op_marker",
    "ref_freq_ghz",
    "op_freq_ghz",
    "ref_freq_target_ghz",
    "op_freq_target_ghz",
    "ref_phase_deg",
    "op_phase_deg",
    "ref_tan_half",
    "op_tan_half",
    "coupler_freq_ghz",
    "q_ext",
    "match_freq_ghz",
    "freq_delta_mhz",
    "coupling_k",
    "k_source",
    "beta",
    "q_ext_target",
    "op_mode_deg",
    "phase_convention",
    "beta_status",
    "is_valid",
    "invalid_reason",
]


def tan_half_phase(phase_deg: float) -> float:
    """Return ``tan(phase_deg / 2)`` using degrees."""

    phase = float(phase_deg)
    tangent = math.tan(math.radians(phase) / 2.0)
    if not math.isfinite(tangent):
        raise ValueError(f"phase_deg produces a non-finite half-phase tangent: {phase_deg!r}")
    return float(tangent)


def calculate_coupler_frequency_ghz(
    frequency_1_ghz: float,
    reflection_phase_1_deg: float,
    frequency_2_ghz: float,
    reflection_phase_2_deg: float,
) -> float:
    """Estimate coupler-cavity frequency from two reflection phases."""

    f1, f2, t1, t2 = _validated_frequency_phase_pair(
        frequency_1_ghz,
        reflection_phase_1_deg,
        frequency_2_ghz,
        reflection_phase_2_deg,
    )
    denominator = f2 * t1 - f1 * t2
    if abs(denominator) < 1e-15:
        raise ValueError("coupler frequency denominator is too close to zero")
    radicand = f1 * f2 * (f1 * t1 - f2 * t2) / denominator
    if radicand <= 0.0 or not math.isfinite(radicand):
        raise ValueError(f"coupler frequency radicand is not positive and finite: {radicand!r}")
    return float(math.sqrt(radicand))


def calculate_external_quality_factor(
    frequency_1_ghz: float,
    reflection_phase_1_deg: float,
    frequency_2_ghz: float,
    reflection_phase_2_deg: float,
) -> float:
    """Estimate external quality factor from two reflection phases."""

    f1, f2, t1, t2 = _validated_frequency_phase_pair(
        frequency_1_ghz,
        reflection_phase_1_deg,
        frequency_2_ghz,
        reflection_phase_2_deg,
    )
    numerator_product = (f2 * t1 - f1 * t2) * (f1 * t1 - f2 * t2)
    if numerator_product <= 0.0 or not math.isfinite(numerator_product):
        raise ValueError(f"external-Q square-root term is not positive and finite: {numerator_product!r}")
    denominator = abs(t1 * t2 * (f1**2 - f2**2))
    if denominator < 1e-15:
        raise ValueError("external-Q denominator is too close to zero")
    return float(math.sqrt(f1 * f2) * math.sqrt(numerator_product) / denominator)


def calculate_coupling_beta(
    frequency_1_ghz: float,
    phase_difference_1_deg: float,
    frequency_2_ghz: float,
    phase_difference_2_deg: float,
    *,
    pi_over_two_frequency_ghz: float,
    operation_mode_deg: float,
    coupling_k: float,
) -> float:
    """Calculate positive Kyhl/Zheng coupling coefficient beta."""

    f1, f2, t1, t2 = _validated_frequency_phase_pair(
        frequency_1_ghz,
        phase_difference_1_deg,
        frequency_2_ghz,
        phase_difference_2_deg,
    )
    k = _positive_coupling_k(coupling_k)
    f_pi2 = _positive_frequency(pi_over_two_frequency_ghz, name="pi_over_two_frequency_ghz")
    sin_theta = math.sin(math.radians(float(operation_mode_deg)))
    if abs(sin_theta) < 1e-15:
        raise ValueError("operation_mode_deg produces a zero sine term")
    denominator = t2 * f1 - t1 * f2
    if abs(denominator) < 1e-15:
        raise ValueError("coupling-beta denominator is too close to zero")
    beta = (
        1.0
        / ((k / 2.0) * f_pi2 * sin_theta)
        * (t1 * t2 * (f1**2 - f2**2))
        / denominator
    )
    if not math.isfinite(beta):
        raise ValueError(f"coupling beta is not finite: {beta!r}")
    return float(abs(beta))


def calculate_matching_frequency_ghz(pi_over_two_frequency_ghz: float, operation_frequency_ghz: float) -> float:
    """Return Zheng's approximate matching frequency target."""

    f_pi2 = _positive_frequency(pi_over_two_frequency_ghz, name="pi_over_two_frequency_ghz")
    f_operation = _positive_frequency(operation_frequency_ghz, name="operation_frequency_ghz")
    return float((f_pi2 + f_operation) / 2.0)


def calculate_target_external_quality_factor(operation_mode_deg: float, *, coupling_k: float) -> float:
    """Return Zheng's approximate target external quality factor."""

    k = _positive_coupling_k(coupling_k)
    sin_theta = math.sin(math.radians(float(operation_mode_deg)))
    if abs(sin_theta) < 1e-15:
        raise ValueError("operation_mode_deg produces a zero sine term")
    return float(2.0 / (k * abs(sin_theta)))


def estimate_coupling_k_from_marker_frequencies(
    *,
    pi_over_two_frequency_ghz: float,
    operation_frequency_ghz: float,
    operation_mode_deg: float,
) -> float:
    """Estimate positive cell-to-cell coupling coefficient from marker frequencies.

    Zheng's matching approximation gives
    ``frequency_theta / frequency_pi_over_two - 1 ~= -k*cos(theta)/2``.
    The coefficient itself is positive, so the returned value is the magnitude.
    """

    f_pi2 = _positive_frequency(pi_over_two_frequency_ghz, name="pi_over_two_frequency_ghz")
    f_operation = _positive_frequency(operation_frequency_ghz, name="operation_frequency_ghz")
    cos_theta = math.cos(math.radians(float(operation_mode_deg)))
    if abs(cos_theta) < 1e-15:
        raise ValueError("operation_mode_deg produces a zero cosine term")
    frequency_ratio_offset = f_operation / f_pi2 - 1.0
    return _positive_coupling_k(-2.0 * frequency_ratio_offset / cos_theta)


def build_coupler_cavity_parameter_table(
    marker_points: pd.DataFrame,
    markers: pd.DataFrame | None = None,
    *,
    reference_marker_name: str = REFERENCE_MARKER_NAME,
    operation_marker_name: str = OPERATION_MARKER_NAME,
    operation_mode_deg: float = 120.0,
    coupling_k: float | None = None,
) -> pd.DataFrame:
    """Build one coupler-cavity parameter estimate per marker-point run."""

    output_columns = _output_columns(marker_points)
    if marker_points.empty:
        return pd.DataFrame(columns=output_columns)

    required_columns = {"marker_name", "s_phase_deg"}
    missing = sorted(required_columns.difference(marker_points.columns))
    if missing:
        return pd.DataFrame(columns=output_columns)

    table = marker_points.copy()
    for column in ID_COLUMN_CANDIDATES:
        if column not in table:
            table[column] = pd.NA
    table = _filter_to_coupler_cavity_endpoints(table)
    if table.empty:
        return pd.DataFrame(columns=output_columns)

    grouping_columns = _grouping_columns(table)
    table_coupling_k, table_coupling_k_source = _first_valid_coupling_k_with_source(
        explicit_argument=coupling_k,
        column_values=[
            *_column_values(markers, COUPLING_K_COLUMNS),
            *_column_values(table, COUPLING_K_COLUMNS),
        ],
    )
    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(grouping_columns, dropna=False, sort=False):
        reference = _first_marker_row(group, reference_marker_name)
        operation = _first_marker_row(group, operation_marker_name)
        if reference is None or operation is None:
            continue

        row: dict[str, object] = {
            **dict(zip(grouping_columns, _as_tuple(group_values), strict=True)),
            "ref_marker": reference_marker_name,
            "op_marker": operation_marker_name,
            "ref_freq_ghz": _frequency_value(reference),
            "op_freq_ghz": _frequency_value(operation),
            "ref_freq_target_ghz": _target_frequency_value(reference),
            "op_freq_target_ghz": _target_frequency_value(operation),
            "ref_phase_deg": reference["s_phase_deg"],
            "op_phase_deg": operation["s_phase_deg"],
            "op_mode_deg": float(operation_mode_deg),
            "phase_convention": PHASE_INPUT_CONVENTION,
        }
        _add_estimates(row, coupling_k=table_coupling_k, coupling_k_source=table_coupling_k_source)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=output_columns)
    output = pd.DataFrame(rows, columns=output_columns)
    if "is_valid" in output:
        output["is_valid"] = output["is_valid"].astype(object)
    return _sort_output(output)


def _add_estimates(row: dict[str, object], *, coupling_k: float | None, coupling_k_source: str) -> None:
    try:
        reference_frequency = float(row["ref_freq_ghz"])
        operation_frequency = float(row["op_freq_ghz"])
        reference_phase = float(row["ref_phase_deg"])
        operation_phase = float(row["op_phase_deg"])
        row["ref_tan_half"] = tan_half_phase(reference_phase)
        row["op_tan_half"] = tan_half_phase(operation_phase)
        row["coupler_freq_ghz"] = calculate_coupler_frequency_ghz(
            reference_frequency,
            reference_phase,
            operation_frequency,
            operation_phase,
        )
        row["q_ext"] = calculate_external_quality_factor(
            reference_frequency,
            reference_phase,
            operation_frequency,
            operation_phase,
        )
        row["match_freq_ghz"] = calculate_matching_frequency_ghz(
            float(row["ref_freq_target_ghz"]),
            float(row["op_freq_target_ghz"]),
        )
        row["freq_delta_mhz"] = (
            float(row["coupler_freq_ghz"]) - float(row["match_freq_ghz"])
        ) * 1000.0
        row["is_valid"] = True
        row["invalid_reason"] = ""
    except (TypeError, ValueError, KeyError) as exc:
        _set_invalid_estimates(row, invalid_reason=str(exc))
        return

    resolved_coupling_k, resolved_coupling_k_source = _resolve_coupling_k(
        row,
        coupling_k=coupling_k,
        coupling_k_source=coupling_k_source,
    )
    if resolved_coupling_k is None:
        row["coupling_k"] = float("nan")
        row["k_source"] = ""
        row["beta"] = float("nan")
        row["q_ext_target"] = float("nan")
        row["beta_status"] = "missing_coupling_k"
        return

    try:
        positive_k = _positive_coupling_k(resolved_coupling_k)
        row["coupling_k"] = positive_k
        row["k_source"] = resolved_coupling_k_source
        row["beta"] = calculate_coupling_beta(
            reference_frequency,
            reference_phase,
            operation_frequency,
            operation_phase,
            pi_over_two_frequency_ghz=float(row["ref_freq_target_ghz"]),
            operation_mode_deg=float(row["op_mode_deg"]),
            coupling_k=positive_k,
        )
        row["q_ext_target"] = calculate_target_external_quality_factor(
            float(row["op_mode_deg"]),
            coupling_k=positive_k,
        )
        row["beta_status"] = "ok"
    except (TypeError, ValueError) as exc:
        row["coupling_k"] = float(resolved_coupling_k)
        row["k_source"] = resolved_coupling_k_source
        row["beta"] = float("nan")
        row["q_ext_target"] = float("nan")
        row["beta_status"] = f"invalid_coupling_beta: {exc}"


def _set_invalid_estimates(row: dict[str, object], *, invalid_reason: str) -> None:
    for column in (
        "ref_tan_half",
        "op_tan_half",
        "coupler_freq_ghz",
        "q_ext",
        "match_freq_ghz",
        "freq_delta_mhz",
        "coupling_k",
        "q_ext_target",
        "beta",
    ):
        row[column] = float("nan")
    row["k_source"] = ""
    row["beta_status"] = "not_calculated"
    row["is_valid"] = False
    row["invalid_reason"] = invalid_reason


def _validated_frequency_phase_pair(
    frequency_1_ghz: float,
    phase_1_deg: float,
    frequency_2_ghz: float,
    phase_2_deg: float,
) -> tuple[float, float, float, float]:
    f1 = _positive_frequency(frequency_1_ghz, name="frequency_1_ghz")
    f2 = _positive_frequency(frequency_2_ghz, name="frequency_2_ghz")
    if abs(f1 - f2) < 1e-15:
        raise ValueError("frequencies must be distinct")
    return f1, f2, tan_half_phase(phase_1_deg), tan_half_phase(phase_2_deg)


def _positive_frequency(value: float, *, name: str) -> float:
    frequency = float(value)
    if frequency <= 0.0 or not math.isfinite(frequency):
        raise ValueError(f"{name} must be positive and finite: {value!r}")
    return frequency


def _positive_coupling_k(value: float) -> float:
    coupling = abs(float(value))
    if coupling <= 0.0 or not math.isfinite(coupling):
        raise ValueError(f"coupling_k must be non-zero and finite: {value!r}")
    return coupling


def _output_columns(marker_points: pd.DataFrame) -> list[str]:
    metadata_columns = _metadata_columns(marker_points)
    insert_index = OUTPUT_COLUMNS.index("ref_marker")
    return [*OUTPUT_COLUMNS[:insert_index], *metadata_columns, *OUTPUT_COLUMNS[insert_index:]]


def _grouping_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in ID_COLUMN_CANDIDATES if column in table]
    columns.extend(column for column in _metadata_columns(table) if column not in columns)
    return columns or ["source_file"]


def _metadata_columns(table: pd.DataFrame) -> list[str]:
    if table is None or table.empty:
        return []
    excluded = set(OUTPUT_COLUMNS) | {
        "marker_name",
        "marker_source",
        "freq_target_ghz",
        "freq_ghz",
        "freq_error_ghz",
        "s_db",
        "s_phase_deg",
        "s_real",
        "s_imag",
        "source_format",
        "uncorrected_freq_ghz",
        "frequency_scale_factor",
        "temp_op_C",
        "temp_meas_C",
        "humidity_fraction",
    }
    return [column for column in table.columns if column.startswith("sim_") and column not in excluded]


def _filter_to_coupler_cavity_endpoints(table: pd.DataFrame) -> pd.DataFrame:
    if "tune_position" not in table:
        return _filter_to_geometry_sweep(table)
    endpoint_metadata = coupler_cavity_endpoint_metadata(table["tune_position"])
    if not endpoint_metadata:
        return _filter_to_geometry_sweep(table)

    table = table.copy()
    row_metadata = [_endpoint_metadata_for_position(value, endpoint_metadata) for value in table["tune_position"]]
    keep_mask = [metadata is not None for metadata in row_metadata]
    table = table.loc[keep_mask].copy()
    kept_metadata = [metadata for metadata in row_metadata if metadata is not None]
    if table.empty:
        return table
    for column in (
        "cpl_pair",
        "cpl_pos_basis",
        "cpl_pos_from",
        "cpl_pos_to",
    ):
        table[column] = [metadata[column] for metadata in kept_metadata]
    return table


def _filter_to_geometry_sweep(table: pd.DataFrame) -> pd.DataFrame:
    geometry_columns = [column for column in ("sim_r_c", "sim_w_c") if column in table]
    if not geometry_columns:
        return table.iloc[0:0].copy()
    if not any(pd.to_numeric(table[column], errors="coerce").nunique() > 1 for column in geometry_columns):
        return table.iloc[0:0].copy()
    if not _has_coupler_geometry_sweep_context(table):
        return table.iloc[0:0].copy()
    table = table.copy()
    table["cpl_pair"] = pd.NA
    table["cpl_pos_basis"] = "geometry_sweep"
    table["cpl_pos_from"] = pd.NA
    table["cpl_pos_to"] = pd.NA
    return table


def _has_coupler_geometry_sweep_context(table: pd.DataFrame) -> bool:
    if "scan_type" in table:
        scan_types = {str(value) for value in table["scan_type"].dropna().unique()}
        if scan_types == {"geometry_sweep"}:
            return True

    if "tune_position" not in table:
        return False
    tune_positions = pd.to_numeric(table["tune_position"], errors="coerce").dropna()
    if tune_positions.nunique() != 1:
        return False
    tune_position = float(tune_positions.iloc[0])
    endpoint_positions = {
        endpoint
        for start, end, _basis in ALLOWED_COUPLER_CAVITY_TRANSITIONS
        for endpoint in (start, end)
    }
    return any(abs(tune_position - endpoint) < 1e-9 for endpoint in endpoint_positions)


def _endpoint_metadata_for_position(
    tune_position: object,
    endpoint_metadata: dict[float, dict[str, object]],
) -> dict[str, object] | None:
    if pd.isna(tune_position):
        return None
    try:
        position = float(tune_position)
    except (TypeError, ValueError):
        return None
    for endpoint_position, metadata in endpoint_metadata.items():
        if abs(position - endpoint_position) < 1e-9:
            return metadata
    return None


def _first_marker_row(group: pd.DataFrame, marker_name: str) -> pd.Series | None:
    rows = group[group["marker_name"] == marker_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _frequency_value(row: pd.Series) -> float:
    if "freq_ghz" in row and pd.notna(row["freq_ghz"]):
        return float(row["freq_ghz"])
    return float(row["freq_target_ghz"])


def _target_frequency_value(row: pd.Series) -> float:
    if "freq_target_ghz" in row and pd.notna(row["freq_target_ghz"]):
        return float(row["freq_target_ghz"])
    return _frequency_value(row)


def _first_valid_coupling_k(*value_groups: Iterable[object]) -> float | None:
    for values in value_groups:
        for value in values:
            if pd.isna(value):
                continue
            try:
                return _positive_coupling_k(float(value))
            except (TypeError, ValueError):
                continue
    return None


def _first_valid_coupling_k_with_source(
    *,
    explicit_argument: object | None,
    column_values: Iterable[object],
) -> tuple[float | None, str]:
    argument_value = _first_valid_coupling_k([explicit_argument])
    if argument_value is not None:
        return argument_value, "explicit_argument"
    column_value = _first_valid_coupling_k(column_values)
    if column_value is not None:
        return column_value, "explicit_column"
    return None, ""


def _resolve_coupling_k(
    row: dict[str, object],
    *,
    coupling_k: float | None,
    coupling_k_source: str,
) -> tuple[float | None, str]:
    if coupling_k is not None and math.isfinite(float(coupling_k)):
        return float(coupling_k), coupling_k_source

    for reference_column, operation_column in (
        ("ref_freq_target_ghz", "op_freq_target_ghz"),
        ("ref_freq_ghz", "op_freq_ghz"),
    ):
        try:
            return (
                estimate_coupling_k_from_marker_frequencies(
                    pi_over_two_frequency_ghz=float(row[reference_column]),
                    operation_frequency_ghz=float(row[operation_column]),
                    operation_mode_deg=float(row["op_mode_deg"]),
                ),
                "marker_frequency_ratio_abs",
            )
        except (TypeError, ValueError, KeyError):
            continue
    return None, ""


def _column_values(table: pd.DataFrame | None, columns: tuple[str, ...]) -> list[object]:
    if table is None or table.empty:
        return []
    values: list[object] = []
    for column in columns:
        if column in table:
            values.extend(table[column].dropna().tolist())
    return values


def _sort_output(table: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in (
            "dataset_id",
            "run_id",
            "tune_position",
            "sim_NumDepth",
            "source_file",
            "s_name",
        )
        if column in table
    ]
    if not sort_columns:
        return table.reset_index(drop=True)
    sortable = table.copy()
    helper_columns: list[str] = []
    for column in sort_columns:
        helper = f"_{column}_sort"
        sortable[helper] = pd.to_numeric(sortable[column], errors="coerce")
        helper_columns.append(helper)
    sorted_table = sortable.sort_values([*helper_columns, *sort_columns], kind="mergesort")
    return sorted_table.drop(columns=helper_columns).reset_index(drop=True)


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
