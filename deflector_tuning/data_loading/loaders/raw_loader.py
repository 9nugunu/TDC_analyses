"""Loader and small processor for original experiment folders under data/raw."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.filename_metadata import metadata_from_filename
from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.records import DataKind
from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.progress import progress_iter


class RawLoader(FolderLoader):
    data_layer = DataLayer.RAW
    data_kind = DataKind.EXP

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        data_folder = self.load(path)
        rows: list[pd.DataFrame] = []
        csv_files = self.list_files(path).csv_files
        for csv_file in progress_iter(
            csv_files,
            desc=f"Loading raw CSV {data_folder.dataset_id}",
            total=len(csv_files),
        ):
            if not _is_supported_raw_csv(csv_file):
                continue
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


def _is_supported_raw_csv(csv_file: Path) -> bool:
    text = csv_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return _has_ri_header(lines) or _has_formatted_data_header(lines)


def _has_ri_header(lines: list[str]) -> bool:
    return any(_ri_s11_columns_from_header(line) is not None for line in lines[:10])


def _has_formatted_data_header(lines: list[str]) -> bool:
    return any("Frequency, Formatted Data, Formatted Data" in line for line in lines[:10])


def _read_ri_csv(csv_file: Path, dataset_id: str) -> pd.DataFrame:
    table = pd.read_csv(csv_file, sep=";", comment="#")
    table = table.dropna(axis="columns", how="all")
    column_pair = _find_ri_s11_column_pair(table.columns)
    if column_pair is None:
        raise ValueError(f"Unsupported raw RI CSV schema in {csv_file.name}: missing S11 real/imag columns")
    real_column, imag_column = column_pair
    freq_hz = table["freq[Hz]"]
    s_real = table[real_column]
    s_imag = table[imag_column]
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


def _ri_s11_columns_from_header(line: str) -> tuple[str, str] | None:
    columns = [column.strip() for column in line.split(";") if column.strip()]
    return _find_ri_s11_column_pair(columns)


def _find_ri_s11_column_pair(columns) -> tuple[str, str] | None:
    pattern = re.compile(r"^(re|im):Trc(\d+)_S11$")
    pairs: dict[str, dict[str, str]] = {}
    for column in columns:
        match = pattern.match(str(column).strip())
        if match is None:
            continue
        component, trace_id = match.groups()
        pair = pairs.setdefault(trace_id, {})
        pair[component] = str(column)
    for trace_id in sorted(pairs, key=int):
        pair = pairs[trace_id]
        if "re" in pair and "im" in pair:
            return pair["re"], pair["im"]
    return None


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
