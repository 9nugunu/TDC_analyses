"""Loader and small processor for original experiment folders under data/raw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.filename_metadata import metadata_from_filename
from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.records import DataKind
from deflector_tuning.data_loading.source_layer import DataLayer


class RawLoader(FolderLoader):
    data_layer = DataLayer.RAW
    data_kind = DataKind.EXP

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        data_folder = self.load(path)
        rows: list[pd.DataFrame] = []
        for csv_file in self.list_files(path).csv_files:
            rows.append(_read_raw_csv(csv_file, data_folder.dataset_id))
        if not rows:
            return pd.DataFrame(columns=_RAW_COLUMNS)
        return pd.concat(rows, ignore_index=True)[_RAW_COLUMNS]


def _read_raw_csv(csv_file: Path, dataset_id: str) -> pd.DataFrame:
    text = csv_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if _has_ri_header(lines):
        return _read_ri_csv(csv_file, dataset_id)
    if _has_formatted_data_header(lines):
        return _read_mag_phase_csv(csv_file, dataset_id)
    raise ValueError(f"Unsupported raw CSV schema in {csv_file.name}")


def _has_ri_header(lines: list[str]) -> bool:
    return any("freq[Hz];re:Trc1_S11;im:Trc1_S11" in line for line in lines[:10])


def _has_formatted_data_header(lines: list[str]) -> bool:
    return any("Frequency, Formatted Data, Formatted Data" in line for line in lines[:10])


def _read_ri_csv(csv_file: Path, dataset_id: str) -> pd.DataFrame:
    table = pd.read_csv(csv_file, sep=";", comment="#")
    table = table.dropna(axis="columns", how="all")
    freq_hz = table["freq[Hz]"]
    s_real = table["re:Trc1_S11"]
    s_imag = table["im:Trc1_S11"]
    s_complex = s_real + 1j * s_imag
    metadata = metadata_from_filename(csv_file, dataset_id)
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "data_kind": DataKind.EXP.value,
            "data_layer": DataLayer.RAW.folder_name,
            "source_file": csv_file.name,
            "freq_hz": freq_hz,
            "freq_ghz": freq_hz / 1e9,
            "s_name": "S11",
            "s_real": s_real,
            "s_imag": s_imag,
            "s_db": _safe_db(np.abs(s_complex)),
            "s_phase_deg": np.angle(s_complex, deg=True),
            "source_format": "raw_csv_ri",
            "tune_position": metadata["tune_position"],
            "port_side": metadata["port_side"],
        }
    )


def _read_mag_phase_csv(csv_file: Path, dataset_id: str) -> pd.DataFrame:
    table = pd.read_csv(csv_file, skiprows=2)
    table = table.rename(
        columns={
            "Frequency": "freq_hz",
            " Formatted Data": "s_db",
            " Formatted Data.1": "s_phase_deg",
        }
    )
    freq_hz = table["freq_hz"]
    metadata = metadata_from_filename(csv_file, dataset_id)
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "data_kind": DataKind.EXP.value,
            "data_layer": DataLayer.RAW.folder_name,
            "source_file": csv_file.name,
            "freq_hz": freq_hz,
            "freq_ghz": freq_hz / 1e9,
            "s_name": "S11",
            "s_real": pd.NA,
            "s_imag": pd.NA,
            "s_db": table["s_db"],
            "s_phase_deg": table["s_phase_deg"],
            "source_format": "raw_csv_db_phase",
            "tune_position": metadata["tune_position"],
            "port_side": metadata["port_side"],
        }
    )


def _safe_db(magnitude: pd.Series) -> pd.Series:
    return 20.0 * np.log10(magnitude.replace(0.0, np.nan))


_RAW_COLUMNS = [
    "dataset_id",
    "data_kind",
    "data_layer",
    "source_file",
    "freq_hz",
    "freq_ghz",
    "s_name",
    "s_real",
    "s_imag",
    "s_db",
    "s_phase_deg",
    "source_format",
    "tune_position",
    "port_side",
]
