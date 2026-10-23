"""Filename metadata helpers for experiment/simulation data files."""

from __future__ import annotations

import re
from pathlib import Path

from deflector_tuning.data_loading.filename_policy import FilenameMetadataPolicy, legacy_filename_policy


def metadata_from_filename(
    file_path: str | Path,
    dataset_id: str,
    *,
    strict_cell_position: bool = False,
    policy: FilenameMetadataPolicy | None = None,
) -> dict[str, object]:
    """Extract lightweight metadata from known tuning filenames.

    An explicit policy controls whether ``in``/``out`` prefixes identify ports.
    Omitting it preserves the legacy assembly-name policy for existing callers.
    """

    stem = Path(file_path).stem.replace("_processed", "")
    side = _side_from_stem(stem)
    if not (policy or legacy_filename_policy(dataset_id)).preserve_port_side:
        side = None
    return {
        "tune_position": _first_number(stem, strict_cell_position=strict_cell_position),
        "port_side": side,
    }


def _first_number(text: str, *, strict_cell_position: bool = False) -> float | None:
    position_match = re.search(
        r"(?:^|[_-])(?:cell|iris)[_-]?(\d+(?:\.\d+)?)|(?:^|[_-])(\d+(?:\.\d+)?)(?:cell|iris)(?:$|[_-])",
        text,
        flags=re.IGNORECASE,
    )
    if position_match is not None:
        return float(next(group for group in position_match.groups() if group is not None))
    if strict_cell_position:
        return None
    for match in re.finditer(r"\d+(?:\.\d+)?", text):
        token = match.group(0)
        if _looks_like_leading_date_token(text, match):
            continue
        return float(token)
    return None


def _looks_like_leading_date_token(text: str, match: re.Match[str]) -> bool:
    token = match.group(0)
    if "." in token or len(token) not in {6, 8}:
        return False
    return match.start() == 0 and len(text) > match.end() and text[match.end()] in {"_", "-"}


def _side_from_stem(stem: str) -> str | None:
    lower = stem.lower()
    if lower.startswith("in_"):
        return "in"
    if lower.startswith("out_"):
        return "out"
    return None
