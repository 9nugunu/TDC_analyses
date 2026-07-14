"""Direct one-port Y11/Z11 Touchstone loading and marker sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import pandas as pd

from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.markers.frequency_markers import MARKER_ORDER, extract_marker_frequencies


Lane = Literal["y11", "z11"]


@dataclass(frozen=True)
class OnePortMatrixSpec:
    lane: Lane
    parameter: str
    suffix: str
    element_column: str
    element_value: str
    real_column: str
    imag_column: str


SPECS: dict[Lane, OnePortMatrixSpec] = {
    "y11": OnePortMatrixSpec(
        "y11", "Y", ".y1p", "y_name", "Y11", "y_re_siemens", "y_im_siemens"
    ),
    "z11": OnePortMatrixSpec(
        "z11", "Z", ".z1p", "z_name", "Z11", "z_re_ohm", "z_im_ohm"
    ),
}


def detect_one_port_matrix_lane(path: str | Path) -> Lane | None:
    """Detect direct one-port Y or Z exports and reject mixed folders."""

    folder = Path(path)
    if not folder.is_dir():
        return None
    has_y = any(
        candidate.is_file() and candidate.suffix.lower() == ".y1p"
        for candidate in folder.iterdir()
    )
    has_z = any(
        candidate.is_file() and candidate.suffix.lower() == ".z1p"
        for candidate in folder.iterdir()
    )
    if has_y and has_z:
        raise ValueError(
            f"Direct matrix folder contains both .y1p and .z1p files: {folder}"
        )
    if has_y:
        return "y11"
    if has_z:
        return "z11"
    return None


def load_one_port_matrix_folder(path: str | Path, lane: Lane) -> pd.DataFrame:
    """Load direct CST one-port matrix exports for one explicit lane."""

    folder = Path(path)
    spec = SPECS[lane]
    files = sorted(
        (
            candidate
            for candidate in folder.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == spec.suffix
        ),
        key=lambda item: item.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No {spec.suffix} files found in {folder}")

    navigator = _result_navigator_by_run_id(folder)
    rows: list[dict[str, object]] = []
    for matrix_file in files:
        data = read_touchstone(matrix_file)
        if data.header.parameter != spec.parameter:
            raise ValueError(
                f"Expected {spec.parameter}-parameter Touchstone data in {matrix_file.name}"
            )
        if any(len(values) != 1 for values in data.s_values):
            raise ValueError(
                f"Expected one-port {spec.element_value} data in {matrix_file.name}"
            )
        run_id = _run_id_from_file_name(matrix_file.name, spec.suffix)
        metadata = navigator.get(run_id, {})
        for frequency, values in zip(data.frequency, data.s_values, strict=True):
            value = values[0]
            rows.append(
                {
                    "dataset_id": folder.name,
                    "data_kind": "sim",
                    "data_layer": "sim",
                    "source_file": matrix_file.name,
                    "run_id": run_id,
                    "freq_ghz": _frequency_to_ghz(
                        frequency, data.header.frequency_unit
                    ),
                    spec.element_column: spec.element_value,
                    spec.real_column: value.real,
                    spec.imag_column: value.imag,
                    "source_format": (
                        f"touchstone_{spec.parameter.lower()}_"
                        f"{data.header.data_format.lower()}"
                    ),
                    "reference_ohm": data.header.reference_ohm,
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def load_y11_touchstone_folder(path: str | Path) -> pd.DataFrame:
    """Load direct one-port admittance exports using compact Y11 columns."""

    return load_one_port_matrix_folder(path, "y11")


def load_z11_touchstone_folder(path: str | Path) -> pd.DataFrame:
    """Load direct one-port impedance exports using compact Z11 columns."""

    return load_one_port_matrix_folder(path, "z11")


def sample_one_port_markers(
    matrix_table: pd.DataFrame,
    markers: pd.DataFrame,
    lane: Lane,
) -> pd.DataFrame:
    """Select the nearest matrix value per source file and marker."""

    spec = SPECS[lane]
    required_matrix_columns = {
        "source_file",
        "freq_ghz",
        spec.real_column,
        spec.imag_column,
    }
    required_marker_columns = {"marker_name", "freq_ghz"}
    if missing := required_matrix_columns.difference(matrix_table.columns):
        raise ValueError(
            f"{spec.element_value} table is missing required columns: {sorted(missing)}"
        )
    if missing := required_marker_columns.difference(markers.columns):
        raise ValueError(f"Marker table is missing required columns: {sorted(missing)}")

    sampled: list[pd.DataFrame] = []
    for _, marker in markers.iterrows():
        target_frequency = float(marker["freq_ghz"])
        rows = matrix_table.copy()
        rows["freq_target_ghz"] = target_frequency
        rows["freq_error_ghz"] = rows["freq_ghz"] - target_frequency
        rows["_abs_freq_error_ghz"] = rows["freq_error_ghz"].abs()
        nearest_index = rows.groupby("source_file", dropna=False)[
            "_abs_freq_error_ghz"
        ].idxmin()
        rows = rows.loc[nearest_index].copy()
        rows["marker_name"] = marker["marker_name"]
        rows["marker_role"] = marker.get("marker_role", pd.NA)
        rows["marker_source"] = marker.get("marker_source", pd.NA)
        sampled.append(rows.drop(columns="_abs_freq_error_ghz"))
    if not sampled:
        return pd.DataFrame(
            columns=[
                *matrix_table.columns,
                "freq_target_ghz",
                "freq_error_ghz",
                "marker_name",
                "marker_role",
                "marker_source",
            ]
        )

    output = pd.concat(sampled, ignore_index=True)
    marker_order = {name: index for index, name in enumerate(MARKER_ORDER)}
    output["_marker_order"] = output["marker_name"].map(marker_order).fillna(
        len(marker_order)
    )
    if "sim_r_c" in output:
        output["_sweep_sort"] = pd.to_numeric(output["sim_r_c"], errors="coerce")
    elif "run_id" in output:
        output["_sweep_sort"] = pd.to_numeric(output["run_id"], errors="coerce")
    else:
        output["_sweep_sort"] = range(len(output))
    output = output.sort_values(
        ["_marker_order", "_sweep_sort", "source_file"],
        kind="mergesort",
    )
    return output.drop(columns=["_marker_order", "_sweep_sort"]).reset_index(drop=True)


def sample_y11_markers(y11_table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Sample direct Y11 values at the supplied dispersion markers."""

    return sample_one_port_markers(y11_table, markers, "y11")


def sample_z11_markers(z11_table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Sample direct Z11 values at the supplied dispersion markers."""

    return sample_one_port_markers(z11_table, markers, "z11")


def extract_one_port_marker_frequencies(
    path: str | Path,
    *,
    marker_role: str,
) -> pd.DataFrame:
    """Return direct-CST marker frequencies, preferring an explicit KYHL table."""

    folder = Path(path)
    marker_files = sorted((folder / "processed").glob("*kyhl_markers.csv"))
    if marker_role == "sim" and marker_files:
        table = pd.read_csv(marker_files[0])
        label_column = _first_present_column(table, ("phase_label", "label"))
        frequency_column = _first_present_column(
            table, ("cst_freq_GHz", "CST_freq_GHz")
        )
        if label_column is not None and frequency_column is not None:
            frequencies = {
                "f_2pi3": _kyhl_marker_frequency(
                    table, label_column, frequency_column, "2pi/3"
                ),
                "f_mean": _kyhl_marker_frequency(
                    table, label_column, frequency_column, "mean(pi/2,2pi/3)"
                ),
                "f_pi2": _kyhl_marker_frequency(
                    table, label_column, frequency_column, "pi/2"
                ),
            }
            return pd.DataFrame(
                [
                    {
                        "marker_name": marker_name,
                        "freq_ghz": frequencies[marker_name],
                        "marker_role": marker_role,
                        "marker_source": "explicit_cst_kyhl_marker_table",
                        "source_file": marker_files[0].relative_to(folder).as_posix(),
                    }
                    for marker_name in MARKER_ORDER
                ]
            )
    return extract_marker_frequencies(folder, marker_role=marker_role)


def _kyhl_marker_frequency(
    table: pd.DataFrame,
    label_column: str,
    frequency_column: str,
    label: str,
) -> float:
    values = table.loc[table[label_column] == label, frequency_column]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one CST KYHL marker row labeled {label!r}")
    return float(values.iloc[0])


def _first_present_column(
    table: pd.DataFrame, candidates: tuple[str, ...]
) -> str | None:
    return next((candidate for candidate in candidates if candidate in table.columns), None)


def _result_navigator_by_run_id(folder: Path) -> dict[int, dict[str, object]]:
    path = folder / "result_navigator.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, sep="\t")
    table = table.rename(columns=lambda name: str(name).strip().strip('"'))
    if "3D Run ID" not in table:
        return {}
    table = table.rename(columns={"3D Run ID": "run_id"})
    rows: dict[int, dict[str, object]] = {}
    for _, row in table.iterrows():
        run_id = int(row["run_id"])
        metadata = {
            f"sim_{column}": _to_number_if_possible(value)
            for column, value in row.items()
            if column != "run_id"
        }
        rows[run_id] = metadata
    return rows


def _run_id_from_file_name(file_name: str, suffix: str) -> int | None:
    match = re.search(
        rf"_(\d+){re.escape(suffix)}$", file_name, flags=re.IGNORECASE
    )
    return int(match.group(1)) if match is not None else None


def _frequency_to_ghz(frequency: float, unit: str) -> float:
    factors = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0}
    if unit not in factors:
        raise ValueError(f"Unsupported Touchstone frequency unit: {unit}")
    return float(frequency) * factors[unit]


def _to_number_if_possible(value: object) -> object:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
