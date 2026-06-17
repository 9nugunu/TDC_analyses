"""Loader for corrected experiment folders under data/prepro."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deflector_tuning.data_loading.filename_metadata import metadata_from_filename
from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.records import DataKind
from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.progress import progress_iter


class PreproLoader(FolderLoader):
    data_layer = DataLayer.PREPRO
    data_kind = DataKind.EXP

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        data_folder = self.load(path)
        rows: list[pd.DataFrame] = []
        csv_files = self.list_files(path).csv_files
        for csv_file in progress_iter(
            csv_files,
            desc=f"Loading prepro CSV {data_folder.dataset_id}",
            total=len(csv_files),
        ):
            csv_table = pd.read_csv(csv_file)
            if "freq[Hz]" not in csv_table.columns or "Magnitude" not in csv_table.columns:
                raise ValueError(f"Unsupported prepro CSV schema in {csv_file.name}")
            metadata = metadata_from_filename(csv_file, data_folder.dataset_id)
            table = pd.DataFrame(
                {
                    "dataset_id": data_folder.dataset_id,
                    "data_kind": data_folder.data_kind.value,
                    "data_layer": data_folder.data_layer.folder_name,
                    "source_file": csv_file.name,
                    "freq_hz": csv_table["freq[Hz]"],
                    "freq_ghz": csv_table["freq[Hz]"] / 1e9,
                    "s_name": "S11",
                    "s_real": pd.NA,
                    "s_imag": pd.NA,
                    "s_db": csv_table["Magnitude"],
                    "s_phase_deg": csv_table["Phase_deg"] if "Phase_deg" in csv_table.columns else pd.NA,
                    "source_format": "processed_csv_db_phase",
                    "tune_position": metadata["tune_position"],
                    "port_side": metadata["port_side"],
                }
            )
            rows.append(table)
        if not rows:
            return pd.DataFrame(columns=_PREPRO_COLUMNS)
        return pd.concat(rows, ignore_index=True)[_PREPRO_COLUMNS]


_PREPRO_COLUMNS = [
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
