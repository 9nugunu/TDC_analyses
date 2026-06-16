"""Shared helper for small folder loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.filename_metadata import metadata_from_filename
from deflector_tuning.data_loading.readers.touchstone_reader import read_touchstone
from deflector_tuning.data_loading.records import DataFiles, DataFolder, DataKind
from deflector_tuning.data_loading.source_layer import DataLayer


class FolderLoader:
    """Base class for loaders that only return a DataFolder in Step 2."""

    data_layer: DataLayer
    data_kind: DataKind

    def load(self, path: str | Path) -> DataFolder:
        data_root = _dataset_root_from_path(Path(path), self.data_layer)
        return DataFolder(
            dataset_id=data_root.name,
            path=data_root,
            data_layer=self.data_layer,
            data_kind=self.data_kind,
        )

    def list_files(self, path: str | Path) -> DataFiles:
        """Group files directly inside a dataset folder by simple extension."""

        data_root = _dataset_root_from_path(Path(path), self.data_layer)
        touchstone_files: list[Path] = []
        csv_files: list[Path] = []
        other_files: list[Path] = []

        for child in sorted(data_root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix in {".s1p", ".s2p", ".s3p", ".s4p"}:
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

    def load_touchstone(self, path: str | Path) -> pd.DataFrame:
        """Read Touchstone files into one simple long-form table."""

        data_folder = self.load(path)
        rows: list[dict[str, object]] = []
        for touchstone_file in self.list_files(path).touchstone_files:
            touchstone_data = read_touchstone(touchstone_file)
            metadata = metadata_from_filename(touchstone_file, data_folder.dataset_id)
            for freq, s_values_at_freq in zip(
                touchstone_data.frequency,
                touchstone_data.s_values,
                strict=True,
            ):
                freq_ghz = _frequency_to_ghz(freq, touchstone_data.header.frequency_unit)
                for s_name, s_value in zip(_s_names(len(s_values_at_freq)), s_values_at_freq, strict=True):
                    rows.append(
                        {
                            "dataset_id": data_folder.dataset_id,
                            "data_kind": data_folder.data_kind.value,
                            "data_layer": data_folder.data_layer.folder_name,
                            "source_file": touchstone_file.name,
                            "freq_ghz": freq_ghz,
                            "s_name": s_name,
                            "s_real": s_value.real,
                            "s_imag": s_value.imag,
                            "s_db": _safe_db(s_value),
                            "s_phase_deg": float(np.angle(s_value, deg=True)),
                            "source_format": f"touchstone_{touchstone_data.header.data_format.lower()}",
                            "reference_ohm": touchstone_data.header.reference_ohm,
                            "is_normalized": touchstone_data.header.is_normalized,
                            "tune_position": metadata["tune_position"],
                            "port_side": metadata["port_side"],
                        }
                    )
        return pd.DataFrame(rows, columns=_TOUCHSTONE_COLUMNS)


_TOUCHSTONE_COLUMNS = [
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


def _safe_db(s_value: complex) -> float:
    magnitude = abs(s_value)
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


def _s_names(value_count: int) -> list[str]:
    if value_count == 1:
        return ["S11"]
    if value_count == 4:
        return ["S11", "S21", "S12", "S22"]
    port_count = int(value_count**0.5)
    return [f"S{row}{column}" for column in range(1, port_count + 1) for row in range(1, port_count + 1)]


def _dataset_root_from_path(path: Path, data_layer: DataLayer) -> Path:
    parts = path.parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts[:-2]):
        if part == "data" and lower_parts[index + 1] == data_layer.folder_name:
            return Path(*parts[: index + 3])
    raise ValueError(f"Expected dataset path under data/{data_layer.folder_name}; got {path!s}")
