"""Dataset folder naming rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePath

from deflector_tuning.data_loading.source_layer import DataLayer

DATASET_NAMING_DOC = "data/NAMING.md"
DATASET_NAME_PATTERN = re.compile(
    r"^(?P<layer>raw|sim|prepro)_(?P<date>\d{6}|undated)_(?P<category>sweep|grid|dispersion)_[A-Za-z0-9][A-Za-z0-9_]*$"
)


@dataclass(frozen=True)
class DatasetIdentity:
    """Parsed identity encoded in a dataset folder name."""

    dataset_id: str
    layer: str
    date: str
    category: str


def parse_dataset_id(dataset_id: str) -> DatasetIdentity:
    """Return the naming fields encoded in a dataset id."""

    match = DATASET_NAME_PATTERN.fullmatch(dataset_id)
    if match is None:
        raise ValueError(
            f"Dataset id does not follow the naming rule: {dataset_id!r}. "
            "Expected <layer>_<YYMMDD|undated>_<sweep|grid|dispersion>_<object>_<condition>; "
            f"see {DATASET_NAMING_DOC}."
        )
    return DatasetIdentity(
        dataset_id=dataset_id,
        layer=match.group("layer"),
        date=match.group("date"),
        category=match.group("category"),
    )


def dataset_identity_from_path(path: str | Path, data_layer: DataLayer) -> DatasetIdentity:
    """Return the dataset identity for a path under ``data/<layer>/<dataset_id>``."""

    dataset_id = _dataset_id_from_layer_path(Path(path), data_layer)
    identity = parse_dataset_id(dataset_id)
    if identity.layer != data_layer.folder_name:
        raise ValueError(
            f"Dataset id layer prefix {identity.layer!r} does not match parent data/{data_layer.folder_name}: "
            f"{dataset_id!r}. Rename the folder or move it to data/{identity.layer}; see {DATASET_NAMING_DOC}."
        )
    return identity


def validate_dataset_id(dataset_id: str, data_layer: DataLayer) -> None:
    """Raise when a dataset id does not follow the data naming contract."""

    identity = parse_dataset_id(dataset_id)
    expected_layer = data_layer.folder_name
    if identity.layer != expected_layer:
        raise ValueError(
            f"Dataset id layer prefix {identity.layer!r} does not match parent data/{expected_layer}: {dataset_id!r}. "
            f"Rename the folder or move it to data/{identity.layer}; see {DATASET_NAMING_DOC}."
        )


def _dataset_id_from_layer_path(path: Path, data_layer: DataLayer) -> str:
    parts = PurePath(path).parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts[:-2]):
        if part == "data" and lower_parts[index + 1] == data_layer.folder_name:
            return parts[index + 2]
    raise ValueError(f"Expected dataset path under data/{data_layer.folder_name}; got {path!s}")
