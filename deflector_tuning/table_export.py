"""Projection and transactional persistence for named analysis-table contracts."""

from __future__ import annotations

import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from deflector_tuning.table_schema import (
    FORBIDDEN_LEGACY_COLUMNS,
    TableNameSpec,
    table_specs,
)
from deflector_tuning.workflows.models import AnalysisPaths

CONTEXT_COLUMNS = frozenset(
    {
        "data_kind",
        "data_layer",
        "marker_role",
        "port_side",
        "s_name",
        "scan_type",
        "marker_source",
        "source_format",
        "frequency_scale_factor",
        "temp_op_C",
        "temp_meas_C",
        "humidity_fraction",
        "reference_ohm",
        "is_normalized",
        "op_mode_deg",
        "op_admit_scale",
        "op_admit_axes_deg",
        "axis_sign",
        "convention_note",
        "formula_note",
        "kyhl_source_note",
        "phase_convention",
        "y_name",
        "z_name",
    }
)


@dataclass(frozen=True)
class TableSaveResult:
    paths: AnalysisPaths
    constants: dict[str, dict[str, object]]


def project_table(
    table_key: str,
    table: pd.DataFrame,
    *,
    contract_name: str = "standard",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove empty/constant context columns while preserving all result columns."""

    specs = table_specs(contract_name)
    if table_key not in specs:
        raise KeyError(f"Unknown {contract_name} table: {table_key}")

    keep: list[str] = []
    constants: dict[str, object] = {}
    for column in table.columns:
        is_context = column in CONTEXT_COLUMNS or column.startswith("sim_")
        if column == "dataset_id" or not is_context:
            keep.append(column)
            continue
        values = table[column].dropna()
        if values.empty:
            continue
        if values.nunique(dropna=True) == 1:
            constants[column] = json_scalar(values.iloc[0])
            continue
        keep.append(column)
    return table.loc[:, keep].copy(), constants


def json_scalar(value: object) -> object:
    """Convert a pandas/numpy scalar into a JSON-compatible scalar."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported manifest scalar: {type(value).__name__}")


def save_standard_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> TableSaveResult:
    """Stage, validate, and atomically replace the 13 managed CSV tables."""

    return save_table_contract(tables, output_dir, "standard")


def save_y11_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> TableSaveResult:
    """Save the compact direct-Y11 table contract."""

    return save_table_contract(tables, output_dir, "y11")


def save_z11_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> TableSaveResult:
    """Save the compact direct-Z11 table contract."""

    return save_table_contract(tables, output_dir, "z11")


def save_table_contract(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    contract_name: str,
) -> TableSaveResult:
    """Stage, validate, and atomically replace one named table contract."""

    specs = table_specs(contract_name)
    expected_keys = list(specs)
    actual_keys = list(tables)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{contract_name} tables must use canonical key order: "
            f"expected {expected_keys}, got {actual_keys}"
        )

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".tables-stage-", dir=folder))
    staged_dir = stage_root / "files"
    backup_dir = stage_root / "backups"
    staged_dir.mkdir()
    backup_dir.mkdir()

    projected_tables: OrderedDict[str, pd.DataFrame] = OrderedDict()
    constants: dict[str, dict[str, object]] = {}
    try:
        for key, spec in specs.items():
            projected, table_constants = project_table(
                key,
                tables[key],
                contract_name=contract_name,
            )
            _validate_projected_columns(key, projected)
            staged_path = staged_dir / spec.filename
            projected.to_csv(staged_path, index=False)
            projected_tables[key] = projected
            constants[key] = table_constants

        _validate_staged_tables(staged_dir, projected_tables, specs)
        previous_files = _backup_managed_files(folder, backup_dir, specs)
        try:
            paths = _replace_managed_files(staged_dir, folder, specs)
            _remove_differing_legacy_files(folder, specs)
            _validate_final_tables(paths, projected_tables, specs)
        except Exception:
            _restore_managed_files(folder, backup_dir, previous_files, specs)
            raise
        return TableSaveResult(paths=paths, constants=constants)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _validate_projected_columns(table_key: str, table: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_LEGACY_COLUMNS.intersection(table.columns))
    if forbidden:
        raise ValueError(f"{table_key} contains forbidden legacy columns: {forbidden}")


def _validate_staged_tables(
    staged_dir: Path,
    tables: OrderedDict[str, pd.DataFrame],
    specs: OrderedDict[str, TableNameSpec],
) -> None:
    for key, spec in specs.items():
        path = staged_dir / spec.filename
        if not path.is_file():
            raise OSError(f"staged table is missing: {path}")
        header = pd.read_csv(path, nrows=0)
        if list(header.columns) != list(tables[key].columns):
            raise ValueError(f"staged table header mismatch: {key}")


def _managed_filenames(
    specs: OrderedDict[str, TableNameSpec],
) -> tuple[str, ...]:
    names: list[str] = []
    for spec in specs.values():
        for name in (spec.filename, spec.legacy_filename):
            if name is not None and name not in names:
                names.append(name)
    return tuple(names)


def _backup_managed_files(
    folder: Path,
    backup_dir: Path,
    specs: OrderedDict[str, TableNameSpec],
) -> frozenset[str]:
    previous: set[str] = set()
    for filename in _managed_filenames(specs):
        source = folder / filename
        if not source.is_file():
            continue
        shutil.copy2(source, backup_dir / filename)
        previous.add(filename)
    return frozenset(previous)


def _replace_managed_files(
    staged_dir: Path,
    folder: Path,
    specs: OrderedDict[str, TableNameSpec],
) -> AnalysisPaths:
    paths: AnalysisPaths = OrderedDict()
    for key, spec in specs.items():
        destination = folder / spec.filename
        (staged_dir / spec.filename).replace(destination)
        paths[key] = destination
    return paths


def _remove_differing_legacy_files(
    folder: Path,
    specs: OrderedDict[str, TableNameSpec],
) -> None:
    for spec in specs.values():
        if spec.legacy_filename is None or spec.legacy_filename == spec.filename:
            continue
        legacy_path = folder / spec.legacy_filename
        if legacy_path.exists():
            legacy_path.unlink()


def _restore_managed_files(
    folder: Path,
    backup_dir: Path,
    previous_files: frozenset[str],
    specs: OrderedDict[str, TableNameSpec],
) -> None:
    for filename in _managed_filenames(specs):
        path = folder / filename
        if filename in previous_files:
            shutil.copy2(backup_dir / filename, path)
        elif path.exists():
            path.unlink()


def _validate_final_tables(
    paths: AnalysisPaths,
    tables: OrderedDict[str, pd.DataFrame],
    specs: OrderedDict[str, TableNameSpec],
) -> None:
    if list(paths) != list(specs):
        raise ValueError("final table paths are not in canonical key order")
    for key, path in paths.items():
        if not path.is_file():
            raise OSError(f"final table is missing: {path}")
        header = pd.read_csv(path, nrows=0)
        if list(header.columns) != list(tables[key].columns):
            raise ValueError(f"final table header mismatch: {key}")
