"""Find the project data layer from a path.

Simple rule
-----------
The folder under ``data`` decides the first loader:

- ``data/sim``: CST or other simulation files.
- ``data/raw``: original experiment files.
- ``data/prepro``: corrected experiment data, usually CSV-style products.

Raw experiment Touchstone files do not need a prepro copy. Later, raw
Touchstone and sim Touchstone can share one parser while keeping different
``DataKind`` values.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path, PurePath


class DataLayer(Enum):
    """Top-level folders under ``data``."""

    SIM = ("sim", "simulation data")
    RAW = ("raw", "original experiment data")
    PREPRO = ("prepro", "corrected or csv-style experiment data")

    def __init__(self, folder_name: str, note: str) -> None:
        self.folder_name = folder_name
        self.note = note


def detect_data_layer(path: str | Path) -> DataLayer:
    """Return the ``data/<layer>`` folder for a path."""

    parts = PurePath(path).parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts[:-1]):
        if part != "data":
            continue
        return _data_layer_from_folder(lower_parts[index + 1])
    raise ValueError(
        f"Expected path under data/sim, data/raw, or data/prepro; got {path!s}"
    )


def _data_layer_from_folder(folder_name: str) -> DataLayer:
    for layer in DataLayer:
        if folder_name == layer.folder_name:
            return layer
    raise ValueError(
        f"Expected path under data/sim, data/raw, or data/prepro; got data/{folder_name}"
    )
