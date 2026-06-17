"""One simple entry point for loading data folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.loaders.prepro_loader import PreproLoader
from deflector_tuning.data_loading.loaders.raw_loader import RawLoader
from deflector_tuning.data_loading.loaders.sim_loader import SimLoader
from deflector_tuning.data_loading.records import DataFolder
from deflector_tuning.data_loading.source_layer import DataLayer, detect_data_layer


class DataLoader:
    """Route a path by ``data/sim``, ``data/raw``, or ``data/prepro``."""

    def __init__(self, *, file_workers: int = 1) -> None:
        self.file_workers = max(int(file_workers), 1)

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load a folder into the default analysis table.

        For now, sim/raw Touchstone folders and prepro processed CSV folders
        are supported.
        """

        loader = self.select_loader(path)
        files = loader.list_files(path)
        if files.touchstone_files:
            return loader.load_touchstone(path, file_workers=self.file_workers)
        if files.other_files and isinstance(loader, SimLoader):
            return loader.load_cst_sparameter_txt(path)
        if files.csv_files and isinstance(loader, PreproLoader):
            return loader.load_csv(path)
        if files.csv_files and isinstance(loader, RawLoader):
            return loader.load_csv(path)
        raise NotImplementedError(f"No default loader for {path!s}")

    def summarize(self, path: str | Path) -> pd.DataFrame:
        """Return a one-row sanity-check summary for a loaded dataset."""

        table = self.load(path)
        return pd.DataFrame(
            [
                {
                    "dataset_id": _first_value(table, "dataset_id"),
                    "data_kind": _first_value(table, "data_kind"),
                    "data_layer": _first_value(table, "data_layer"),
                    "row_count": len(table),
                    "file_count": table["source_file"].nunique() if "source_file" in table else 0,
                    "freq_min_ghz": table["freq_ghz"].min() if "freq_ghz" in table else None,
                    "freq_max_ghz": table["freq_ghz"].max() if "freq_ghz" in table else None,
                    "source_formats": _unique_values(table, "source_format"),
                    "s_names": _unique_values(table, "s_name"),
                    "tune_positions": _unique_values(table, "tune_position"),
                    "port_sides": _unique_values(table, "port_side"),
                }
            ]
        )

    def extract_nearest(self, path: str | Path, target_freq_ghz: float) -> pd.DataFrame:
        """Return nearest-frequency S-parameter rows per file/metadata group."""

        table = self.load(path).copy()
        table["target_freq_ghz"] = target_freq_ghz
        table["freq_error_ghz"] = table["freq_ghz"] - target_freq_ghz
        table["_abs_freq_error_ghz"] = table["freq_error_ghz"].abs()

        group_columns = ["source_file", "s_name", "tune_position", "port_side"]
        for column in group_columns:
            if column not in table:
                table[column] = pd.NA

        nearest_index = table.groupby(group_columns, dropna=False)["_abs_freq_error_ghz"].idxmin()
        output_columns = [
            "dataset_id",
            "data_kind",
            "data_layer",
            "source_file",
            "tune_position",
            "port_side",
            "s_name",
            "target_freq_ghz",
            "freq_ghz",
            "freq_error_ghz",
            "s_db",
            "s_phase_deg",
            "source_format",
        ]
        return table.loc[nearest_index, output_columns].reset_index(drop=True)

    def load_folder(self, path: str | Path) -> DataFolder:
        """Return only folder identity, without reading data files."""

        return self.select_loader(path).load(path)

    def select_loader(self, path: str | Path) -> FolderLoader:
        """Choose the first loader from the top-level data folder."""

        data_layer = detect_data_layer(path)
        if data_layer is DataLayer.SIM:
            return SimLoader()
        if data_layer is DataLayer.RAW:
            return RawLoader()
        if data_layer is DataLayer.PREPRO:
            return PreproLoader()
        raise ValueError(f"No loader for data layer {data_layer!r}")


def _first_value(table: pd.DataFrame, column: str) -> object:
    if column not in table or table.empty:
        return None
    return table[column].iloc[0]


def _unique_values(table: pd.DataFrame, column: str) -> list[object]:
    if column not in table:
        return []
    values = table[column].dropna().unique().tolist()
    return sorted(values)
