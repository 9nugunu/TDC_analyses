"""Shared helper for small folder loaders."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.filename_metadata import metadata_from_filename
from deflector_tuning.data_loading.dataset_naming import validate_dataset_id
from deflector_tuning.data_loading.readers.touchstone_reader import (
    read_touchstone,
    touchstone_parameter_from_suffix,
)
from deflector_tuning.data_loading.records import DataFiles, DataFolder, DataKind
from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.progress import progress_iter


class FolderLoader:
    """Base class for loaders that only return a DataFolder in Step 2."""

    data_layer: DataLayer
    data_kind: DataKind

    def load(self, path: str | Path) -> DataFolder:
        data_root = _dataset_root_from_path(Path(path), self.data_layer)
        validate_dataset_id(data_root.name, self.data_layer)
        return DataFolder(
            dataset_id=data_root.name,
            path=data_root,
            data_layer=self.data_layer,
            data_kind=self.data_kind,
        )

    def list_files(self, path: str | Path) -> DataFiles:
        """Group files directly inside the requested dataset or export folder."""

        data_root = _dataset_root_from_path(Path(path), self.data_layer)
        requested_root = Path(path)
        file_root = requested_root if requested_root.is_dir() else data_root
        touchstone_files: list[Path] = []
        csv_files: list[Path] = []
        other_files: list[Path] = []

        for child in sorted(file_root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if _is_touchstone_file(child):
                touchstone_files.append(child)
            elif suffix == ".csv":
                csv_files.append(child)
            else:
                other_files.append(child)

        return DataFiles(
            touchstone_files=touchstone_files,
            csv_files=csv_files,
            other_files=other_files,
        )

    def load_touchstone(self, path: str | Path, *, file_workers: int = 1) -> pd.DataFrame:
        """Read Touchstone files into one simple long-form table."""

        data_folder = self.load(path)
        touchstone_files = self.list_files(path).touchstone_files
        parameter_families = {
            touchstone_parameter_from_suffix(touchstone_file)
            for touchstone_file in touchstone_files
        }
        if len(parameter_families) != 1:
            families = ", ".join(sorted(parameter_families)) or "none"
            raise ValueError(
                f"Expected one Touchstone parameter family in {data_folder.dataset_id}; "
                f"found mixed Touchstone parameter families: {families}"
            )
        parameter = next(iter(parameter_families))
        rows = _load_touchstone_file_rows(
            touchstone_files,
            dataset_id=data_folder.dataset_id,
            data_kind=data_folder.data_kind.value,
            data_layer=data_folder.data_layer.folder_name,
            strict_cell_position=self.data_layer is DataLayer.SIM,
            file_workers=file_workers,
        )
        return pd.DataFrame(rows, columns=_touchstone_columns(parameter))


def _load_touchstone_file_rows(
    touchstone_files: list[Path],
    *,
    dataset_id: str,
    data_kind: str,
    data_layer: str,
    strict_cell_position: bool,
    file_workers: int,
) -> list[dict[str, object]]:
    effective_workers = min(max(int(file_workers), 1), len(touchstone_files) or 1)
    if effective_workers <= 1:
        rows: list[dict[str, object]] = []
        for touchstone_file in progress_iter(
            touchstone_files,
            desc=f"Loading Touchstone {dataset_id}",
            total=len(touchstone_files),
        ):
            rows.extend(
                _read_one_touchstone_file_rows(
                    touchstone_file,
                    dataset_id=dataset_id,
                    data_kind=data_kind,
                    data_layer=data_layer,
                    strict_cell_position=strict_cell_position,
                )
            )
        return rows

    indexed_rows: list[tuple[int, list[dict[str, object]]]] = []
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        future_to_index = {
            executor.submit(
                _read_one_touchstone_file_rows,
                touchstone_file,
                dataset_id=dataset_id,
                data_kind=data_kind,
                data_layer=data_layer,
                strict_cell_position=strict_cell_position,
            ): index
            for index, touchstone_file in enumerate(touchstone_files)
        }
        for future in progress_iter(
            as_completed(future_to_index),
            desc=f"Loading Touchstone {dataset_id}",
            total=len(future_to_index),
        ):
            indexed_rows.append((future_to_index[future], future.result()))

    rows: list[dict[str, object]] = []
    for _, file_rows in sorted(indexed_rows, key=lambda item: item[0]):
        rows.extend(file_rows)
    return rows


def _read_one_touchstone_file_rows(
    touchstone_file: Path,
    *,
    dataset_id: str,
    data_kind: str,
    data_layer: str,
    strict_cell_position: bool,
) -> list[dict[str, object]]:
    touchstone_data = read_touchstone(touchstone_file)
    metadata = metadata_from_filename(
        touchstone_file,
        dataset_id,
        strict_cell_position=strict_cell_position,
    )
    rows: list[dict[str, object]] = []
    parameter = touchstone_data.header.parameter
    for freq, values_at_freq in zip(
        touchstone_data.frequency,
        touchstone_data.values,
        strict=True,
    ):
        freq_ghz = _frequency_to_ghz(freq, touchstone_data.header.frequency_unit)
        names = _parameter_names(parameter, len(values_at_freq))
        for name, value in zip(names, values_at_freq, strict=True):
            row = {
                "dataset_id": dataset_id,
                "data_kind": data_kind,
                "data_layer": data_layer,
                "source_file": touchstone_file.name,
                "freq_ghz": freq_ghz,
            }
            row.update(_parameter_value_columns(parameter, name, value))
            row.update(
                {
                    "source_format": _source_format(
                        parameter,
                        touchstone_data.header.data_format,
                    ),
                    "reference_ohm": touchstone_data.header.reference_ohm,
                    "tune_position": metadata["tune_position"],
                    "port_side": metadata["port_side"],
                }
            )
            if parameter == "S":
                row["is_normalized"] = touchstone_data.header.is_normalized
            rows.append(row)
    return rows


_S_TOUCHSTONE_COLUMNS = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "freq_ghz",
    "s_name",
    "s_real",
    "s_imag",
    "s_db",
    "s_phase_deg",
    "source_format",
    "reference_ohm",
    "is_normalized",
    "tune_position",
    "port_side",
]

_Y_TOUCHSTONE_COLUMNS = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "freq_ghz",
    "y_name",
    "y_real_siemens",
    "y_imag_siemens",
    "source_format",
    "reference_ohm",
    "tune_position",
    "port_side",
]

_Z_TOUCHSTONE_COLUMNS = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "freq_ghz",
    "z_name",
    "z_real_ohm",
    "z_imag_ohm",
    "source_format",
    "reference_ohm",
    "tune_position",
    "port_side",
]


def _safe_db(value: complex) -> float:
    magnitude = abs(value)
    if magnitude == 0.0:
        return float("-inf")
    return float(20.0 * np.log10(magnitude))


def _frequency_to_ghz(frequency: float, frequency_unit: str) -> float:
    factors = {
        "Hz": 1e-9,
        "kHz": 1e-6,
        "MHz": 1e-3,
        "GHz": 1.0,
    }
    if frequency_unit not in factors:
        raise ValueError(f"Unsupported Touchstone frequency unit: {frequency_unit}")
    return frequency * factors[frequency_unit]


def _parameter_names(parameter: str, value_count: int) -> list[str]:
    if value_count == 1:
        return [f"{parameter}11"]
    if value_count == 4:
        return [
            f"{parameter}11",
            f"{parameter}21",
            f"{parameter}12",
            f"{parameter}22",
        ]
    port_count = int(value_count**0.5)
    return [
        f"{parameter}{row}{column}"
        for column in range(1, port_count + 1)
        for row in range(1, port_count + 1)
    ]


def _parameter_value_columns(
    parameter: str,
    name: str,
    value: complex,
) -> dict[str, object]:
    if parameter == "S":
        return {
            "s_name": name,
            "s_real": value.real,
            "s_imag": value.imag,
            "s_db": _safe_db(value),
            "s_phase_deg": float(np.angle(value, deg=True)),
        }
    if parameter == "Y":
        return {
            "y_name": name,
            "y_real_siemens": value.real,
            "y_imag_siemens": value.imag,
        }
    if parameter == "Z":
        return {
            "z_name": name,
            "z_real_ohm": value.real,
            "z_imag_ohm": value.imag,
        }
    raise ValueError(f"Unsupported Touchstone parameter family: {parameter}")


def _touchstone_columns(parameter: str) -> list[str]:
    columns = {
        "S": _S_TOUCHSTONE_COLUMNS,
        "Y": _Y_TOUCHSTONE_COLUMNS,
        "Z": _Z_TOUCHSTONE_COLUMNS,
    }
    try:
        return columns[parameter]
    except KeyError as error:
        raise ValueError(f"Unsupported Touchstone parameter family: {parameter}") from error


def _source_format(parameter: str, data_format: str) -> str:
    if parameter == "S":
        return f"touchstone_{data_format.lower()}"
    return f"touchstone_{parameter.lower()}_{data_format.lower()}"


def _is_touchstone_file(path: Path) -> bool:
    try:
        touchstone_parameter_from_suffix(path)
    except ValueError:
        return False
    return True


def _dataset_root_from_path(path: Path, data_layer: DataLayer) -> Path:
    parts = path.parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts[:-2]):
        if part == "data" and lower_parts[index + 1] == data_layer.folder_name:
            return Path(*parts[: index + 3])
    raise ValueError(f"Expected dataset path under data/{data_layer.folder_name}; got {path!s}")
