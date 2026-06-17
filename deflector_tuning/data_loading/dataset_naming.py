"""Dataset folder naming rules."""

from __future__ import annotations

import re

from deflector_tuning.data_loading.source_layer import DataLayer

DATASET_NAMING_DOC = "data/NAMING.md"
DATASET_NAME_PATTERN = re.compile(
    r"^(?P<layer>raw|sim|prepro)_(?P<date>\d{6}|undated)_(?P<category>sweep|grid|dispersion)_[A-Za-z0-9][A-Za-z0-9_]*$"
)


def validate_dataset_id(dataset_id: str, data_layer: DataLayer) -> None:
    """Raise when a dataset id does not follow the data naming contract."""

    match = DATASET_NAME_PATTERN.fullmatch(dataset_id)
    if match is None:
        raise ValueError(
            f"Dataset id does not follow the naming rule: {dataset_id!r}. "
            "Expected <layer>_<YYMMDD|undated>_<sweep|grid|dispersion>_<object>_<condition>; "
            f"see {DATASET_NAMING_DOC}."
        )
    expected_layer = data_layer.folder_name
    actual_layer = match.group("layer")
    if actual_layer != expected_layer:
        raise ValueError(
            f"Dataset id layer prefix {actual_layer!r} does not match parent data/{expected_layer}: {dataset_id!r}. "
            f"Rename the folder or move it to data/{actual_layer}; see {DATASET_NAMING_DOC}."
        )
