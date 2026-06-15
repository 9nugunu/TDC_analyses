"""Small records returned by the data loader."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from deflector_tuning.data_loading.source_layer import DataLayer


class DataKind(Enum):
    """Where the data came from, independent of file format."""

    SIM = "sim"
    EXP = "experiment"


@dataclass(frozen=True)
class DataFolder:
    """A dataset folder found under ``data/sim``, ``data/raw``, or ``data/prepro``."""

    dataset_id: str
    path: Path
    data_layer: DataLayer
    data_kind: DataKind


@dataclass(frozen=True)
class DataFiles:
    """Simple file groups found directly inside one dataset folder."""

    touchstone_files: list[Path]
    csv_files: list[Path]
    other_files: list[Path]
