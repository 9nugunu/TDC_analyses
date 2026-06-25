"""Cell-center and iris-center field-energy sensitivity ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS: list[str] = [
    "field_kind",
    "component",
    "trace_label",
    "pair_index",
    "cell_label",
    "upstream_iris_label",
    "downstream_iris_label",
    "cell_center_z_mm",
    "upstream_iris_z_mm",
    "downstream_iris_z_mm",
    "cell_field_value",
    "upstream_iris_field_value",
    "downstream_iris_field_value",
    "cell_field_energy",
    "upstream_iris_field_energy",
    "downstream_iris_field_energy",
    "upstream_iris_to_cell_energy_ratio",
    "downstream_iris_to_cell_energy_ratio",
    "adjacent_average_iris_to_cell_energy_ratio",
    "adjacent_max_iris_to_cell_energy_ratio",
]
SUMMARY_COLUMNS: list[str] = [
    "ratio_definition",
    "pair_count",
    "mean_energy_ratio",
    "std_energy_ratio",
    "min_energy_ratio",
    "max_energy_ratio",
]


def compute_cell_iris_field_energy_ratios(
    field_profile_export,
    *,
    regular_cell_count: int = 9,
    field_kind: str = "e",
) -> pd.DataFrame:
    """Compute ``|E_iris|^2 / |E_cell|^2`` ratios from a z-profile export."""

    trace = _single_field_profile_trace(field_profile_export, expected_kind=field_kind)
    params = field_profile_export.parameters
    if "d" not in params or "t" not in params:
        raise ValueError("field_profile_export.parameters requires 'd' and 't'")
    period = float(params["d"]) + float(params["t"])
    iris_z = np.array([float(params["t"]) / 2.0 + index * period for index in range(regular_cell_count + 1)])
    cell_z = np.array([float(params["t"]) + float(params["d"]) / 2.0 + index * period for index in range(regular_cell_count)])

    iris_field = interpolate_profile(trace.z_mm, trace.values, iris_z)
    cell_field = interpolate_profile(trace.z_mm, trace.values, cell_z)

    rows: list[dict[str, object]] = []
    for index in range(regular_cell_count):
        upstream_field = float(iris_field[index])
        downstream_field = float(iris_field[index + 1])
        cell_value = float(cell_field[index])
        cell_energy = cell_value**2
        upstream_energy = upstream_field**2
        downstream_energy = downstream_field**2
        adjacent_average_energy = 0.5 * (upstream_energy + downstream_energy)
        adjacent_max_energy = max(upstream_energy, downstream_energy)
        rows.append(
            {
                "field_kind": trace.field_kind,
                "component": trace.component,
                "trace_label": trace.label,
                "pair_index": index + 1,
                "cell_label": f"R{index + 1}",
                "upstream_iris_label": "IN" if index == 0 else f"I{index}",
                "downstream_iris_label": "OUT" if index == regular_cell_count - 1 else f"I{index + 1}",
                "cell_center_z_mm": float(cell_z[index]),
                "upstream_iris_z_mm": float(iris_z[index]),
                "downstream_iris_z_mm": float(iris_z[index + 1]),
                "cell_field_value": cell_value,
                "upstream_iris_field_value": upstream_field,
                "downstream_iris_field_value": downstream_field,
                "cell_field_energy": cell_energy,
                "upstream_iris_field_energy": upstream_energy,
                "downstream_iris_field_energy": downstream_energy,
                "upstream_iris_to_cell_energy_ratio": safe_ratio(upstream_energy, cell_energy),
                "downstream_iris_to_cell_energy_ratio": safe_ratio(downstream_energy, cell_energy),
                "adjacent_average_iris_to_cell_energy_ratio": safe_ratio(adjacent_average_energy, cell_energy),
                "adjacent_max_iris_to_cell_energy_ratio": safe_ratio(adjacent_max_energy, cell_energy),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize_cell_iris_field_energy_ratios(field_energy_ratios: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate iris/cell field-energy ratios."""

    if field_energy_ratios.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for column, label in [
        ("upstream_iris_to_cell_energy_ratio", "upstream_iris_to_cell"),
        ("downstream_iris_to_cell_energy_ratio", "downstream_iris_to_cell"),
        ("adjacent_average_iris_to_cell_energy_ratio", "adjacent_average_iris_to_cell"),
        ("adjacent_max_iris_to_cell_energy_ratio", "adjacent_max_iris_to_cell"),
    ]:
        values = pd.to_numeric(field_energy_ratios[column], errors="coerce").dropna()
        rows.append(
            {
                "ratio_definition": label,
                "pair_count": int(values.count()),
                "mean_energy_ratio": float(values.mean()) if not values.empty else float("nan"),
                "std_energy_ratio": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "min_energy_ratio": float(values.min()) if not values.empty else float("nan"),
                "max_energy_ratio": float(values.max()) if not values.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def interpolate_profile(source_z: np.ndarray, source_y: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Interpolate a profile after sorting by z."""

    order = np.argsort(source_z)
    z_sorted = np.asarray(source_z, dtype=float)[order]
    y_sorted = np.asarray(source_y, dtype=float)[order]
    return np.interp(np.asarray(target_z, dtype=float), z_sorted, y_sorted)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator and guard zero denominators."""

    return float(numerator / denominator) if denominator else float("nan")


def _single_field_profile_trace(field_profile_export, *, expected_kind: str):
    normalized_expected = _normalized_field_kind(expected_kind)
    traces = [
        trace
        for trace in field_profile_export.traces
        if _normalized_field_kind(trace.field_kind) == normalized_expected and trace.value_kind != "phase"
    ]
    if not traces:
        raise ValueError(f"No {expected_kind} non-phase field profile trace found")
    return traces[0]


def _normalized_field_kind(field_kind: str) -> str:
    text = field_kind.strip().lower().replace("_", "-")
    if text in {"e", "e-field", "electric", "electric-field"}:
        return "e"
    if text in {"h", "h-field", "magnetic", "magnetic-field"}:
        return "h"
    return text
