"""Loader for simulation folders under data/sim."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.data_loading.loaders.folder_loader import FolderLoader
from deflector_tuning.data_loading.records import DataKind
from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.progress import progress_iter


class SimLoader(FolderLoader):
    data_layer = DataLayer.SIM
    data_kind = DataKind.SIM

    def load_cst_sparameter_txt(self, path: str | Path) -> pd.DataFrame:
        """Read CST text S-parameter exports into the common long-form table."""

        data_folder = self.load(path)
        rows: list[dict[str, object]] = []
        other_files = self.list_files(path).other_files
        for txt_file in progress_iter(
            other_files,
            desc=f"Loading CST text {data_folder.dataset_id}",
            total=len(other_files),
        ):
            if txt_file.suffix.lower() != ".txt":
                continue
            parameters = _read_cst_parameter_header(txt_file)
            for freq_ghz, s_db in _iter_cst_magnitude_rows(txt_file):
                row = {
                    "dataset_id": data_folder.dataset_id,
                    "data_kind": data_folder.data_kind.value,
                    "data_layer": data_folder.data_layer.folder_name,
                    "source_file": txt_file.name,
                    "freq_ghz": freq_ghz,
                    "s_name": "S11",
                    "s_real": float(10.0 ** (s_db / 20.0)),
                    "s_imag": 0.0,
                    "s_db": s_db,
                    "s_phase_deg": 0.0,
                    "source_format": "cst_txt_magnitude",
                    "reference_ohm": 50.0,
                    "is_normalized": True,
                    "tune_position": None,
                    "port_side": None,
                }
                row.update(parameters)
                rows.append(row)
        if not rows:
            raise NotImplementedError(f"No CST S-parameter txt files in {path!s}")
        return pd.DataFrame(rows)

    def load_touchstone(self, path: str | Path, *, file_workers: int = 1) -> pd.DataFrame:
        table = super().load_touchstone(path, file_workers=file_workers)
        navigator = _read_result_navigator(Path(path))
        if navigator.empty:
            return _assign_scan_type(table)
        table = table.copy()
        table["run_id"] = table["source_file"].map(_run_id_from_file_name)
        return _assign_scan_type(_assign_num_depth_tune_positions(_merge_result_navigator(table, navigator)))


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


def _assign_scan_type(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    output["scan_type"] = _scan_type(output)
    return output


def _assign_num_depth_tune_positions(table: pd.DataFrame) -> pd.DataFrame:
    """Use CST NumDepth as a cell-like tune axis when filenames lack one."""

    if "sim_NumDepth" not in table:
        return table
    if "tune_position" not in table:
        table = table.copy()
        table["tune_position"] = pd.NA
    missing_tune_position = table["tune_position"].isna()
    if not missing_tune_position.any():
        return table

    num_depth = pd.to_numeric(table.loc[missing_tune_position, "sim_NumDepth"], errors="coerce")
    if num_depth.notna().sum() == 0:
        return table

    output = table.copy()
    output.loc[missing_tune_position, "tune_position"] = num_depth - 0.5
    return output


def _scan_type(table: pd.DataFrame) -> str:
    if "tune_position" in table and table["tune_position"].dropna().nunique() > 1:
        return "tune_position"
    if "sim_r_c" in table and "sim_w_c" in table:
        grid_points = table[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
        if len(grid_points) > 1:
            return "grid_2d"
    return "single_point"


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


def _read_cst_parameter_header(path: Path) -> dict[str, object]:
    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    match = re.search(r"\{(?P<body>.*)\}", first_line)
    if match is None:
        return {}
    parameters: dict[str, object] = {}
    for item in match.group("body").split(";"):
        if "=" not in item:
            continue
        name, value = [part.strip() for part in item.split("=", maxsplit=1)]
        if not name:
            continue
        parameters[f"sim_{name}"] = _parse_cst_parameter_value(value)
    return parameters


def _parse_cst_parameter_value(value: str) -> object:
    try:
        number = float(value)
    except ValueError:
        return value
    if np.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _iter_cst_magnitude_rows(path: Path):
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            yield float(parts[0]), float(parts[1])
        except ValueError:
            continue
