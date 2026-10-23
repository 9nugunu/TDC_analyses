"""Resolve publication marker configuration against its source evidence."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import numpy as np
from deflector_tuning.data_loading.cst_metadata import sha256


def resolve_markers(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify configured markers against the actual dispersion-summary row."""
    source = root / config["marker_source"]
    with source.open(newline="", encoding="utf-8") as stream:
        candidates = [row for row in csv.DictReader(stream)
                      if int(row["mode_index"]) == config["marker_mode_index"]]
    if len(candidates) != 1:
        raise ValueError("Dispersion mode must identify exactly one source row")
    row = candidates[0]
    markers = []
    for item in config["markers"]:
        source_values = [float(row[column]) for column in item.get(
            "mean_of_columns", [item.get("source_column")])]
        derived = float(np.mean(source_values))
        if not np.isclose(derived, item["requested_GHz"], atol=1e-12, rtol=0):
            raise ValueError(f"Configured marker disagrees with dispersion source: {item['name']}")
        markers.append({**item, "source_derived_GHz": derived})
    return markers, {"path": source.relative_to(root).as_posix(), "sha256": sha256(source),
                     "mode_index": config["marker_mode_index"], "source_row": row,
                     "markers": markers}
