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

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load a folder into the default analysis table.

        For now, sim/raw Touchstone folders and prepro processed CSV folders
        are supported.
        """

        loader = self.select_loader(path)
        files = loader.list_files(path)
        if files.touchstone_files:
            return loader.load_touchstone(path)
        if files.csv_files and isinstance(loader, PreproLoader):
            return loader.load_csv(path)
        if files.csv_files and isinstance(loader, RawLoader):
            return loader.load_csv(path)
        raise NotImplementedError(f"No default loader for {path!s}")

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
