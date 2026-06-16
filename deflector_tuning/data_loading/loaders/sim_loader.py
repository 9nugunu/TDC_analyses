"""Loader for simulation folders under data/sim."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.records import DataKind
from deflector_tuning.data_loading.source_layer import DataLayer


class SimLoader(FolderLoader):
    data_layer = DataLayer.SIM
    data_kind = DataKind.SIM

    def load_touchstone(self, path: str | Path) -> pd.DataFrame:
        table = super().load_touchstone(path)
        navigator = _read_result_navigator(Path(path))
        if navigator.empty:
            return table
        table = table.copy()
        table["run_id"] = table["source_file"].map(_run_id_from_file_name)
        return _merge_result_navigator(table, navigator)


def _merge_result_navigator(table: pd.DataFrame, navigator: pd.DataFrame) -> pd.DataFrame:
    if table["run_id"].notna().any():
        table = table.copy()
        table["run_id"] = table["run_id"].astype("Int64")
        navigator = navigator.copy()
        navigator["run_id"] = navigator["run_id"].astype("Int64")
        return table.merge(navigator, on="run_id", how="left")
    if "sim_NumTune" in navigator and "tune_position" in table:
        table = table.copy()
        table["sim_NumTune"] = table["tune_position"].map(_num_tune_from_cell_position)
        navigator_without_empty_tunes = navigator.dropna(subset=["sim_NumTune"]).copy()
        navigator_without_empty_tunes["sim_NumTune"] = navigator_without_empty_tunes["sim_NumTune"].astype("Int64")
        table["sim_NumTune"] = table["sim_NumTune"].astype("Int64")
        table = table.drop(columns=["run_id"])
        return table.merge(navigator_without_empty_tunes, on="sim_NumTune", how="left")
    return table


def _read_result_navigator(path: Path) -> pd.DataFrame:
    navigator_path = _dataset_root(path) / "result_navigator.csv"
    if not navigator_path.exists():
        return pd.DataFrame()
    navigator = pd.read_csv(navigator_path, sep="\t")
    navigator = navigator.rename(columns=lambda name: str(name).strip().strip('"'))
    if "3D Run ID" not in navigator.columns:
        return pd.DataFrame()
    navigator = navigator.rename(columns={"3D Run ID": "run_id"})
    navigator["run_id"] = navigator["run_id"].astype(int)
    metadata_columns = [column for column in navigator.columns if column != "run_id"]
    navigator = navigator.rename(columns={column: f"sim_{column}" for column in metadata_columns})
    for column in navigator.columns:
        navigator[column] = _to_number_if_possible(navigator[column])
    return navigator


def _to_number_if_possible(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError):
        return series


def _dataset_root(path: Path) -> Path:
    parts = path.parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts[:-2]):
        if part == "data" and lower_parts[index + 1] == "sim":
            return Path(*parts[: index + 3])
    raise ValueError(f"Expected simulation path under data/sim; got {path!s}")


def _run_id_from_file_name(file_name: str) -> int | None:
    match = re.search(r"_(\d+)\.s\d+p$", file_name, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d+)\.s\d+p$", file_name, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _num_tune_from_cell_position(tune_position: object) -> int | None:
    if pd.isna(tune_position):
        return None
    value = float(tune_position) - 0.5
    if abs(value - round(value)) > 1e-9:
        return None
    return int(round(value))
