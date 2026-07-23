"""Phase response of experimental plunger-offset measurements."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


OUTPUT_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "source_file",
    "tune_position",
    "marker_name",
    "plunger_offset_mm",
    "phase_deg",
    "phase_delta_deg",
    "reference_phase_deg",
)
_OFFSET_PATTERN = re.compile(
    r"(?P<offset>[+-]?\d+(?:\.\d+)?)mmoffset",
    flags=re.IGNORECASE,
)
_ZERO_OFFSET_PATTERN = re.compile(r"zerooffset", flags=re.IGNORECASE)


def build_phase_offset_response(
    marker_points: pd.DataFrame,
    *,
    reference_offset_mm: float = 0.0,
) -> pd.DataFrame:
    """Return marker phase changes relative to the zero-plunger-offset mean.

    The source filename supplies the experimental displacement through either
    ``<signed value>mmoffset`` or ``zerooffset``.  Each marker uses its own
    circular mean of all reference-offset repeats as the phase reference.
    """

    required = ("source_file", "marker_name", "s_phase_deg")
    missing = [column for column in required if column not in marker_points]
    if missing:
        raise ValueError(f"marker_points is missing required columns: {missing}")

    table = marker_points.copy()
    table["plunger_offset_mm"] = table["source_file"].map(
        _plunger_offset_mm_from_source_file
    )
    unresolved = table["plunger_offset_mm"].isna()
    if unresolved.any():
        files = ", ".join(
            sorted(table.loc[unresolved, "source_file"].astype(str).unique())
        )
        raise ValueError(
            "Could not read a plunger offset from source file(s): " + files
        )
    table["phase_deg"] = pd.to_numeric(table["s_phase_deg"], errors="coerce")
    if table["phase_deg"].isna().any():
        raise ValueError("marker_points contains non-numeric S-parameter phase values")

    rows: list[pd.DataFrame] = []
    for _, group in table.groupby("marker_name", sort=False, dropna=False):
        reference = group[
            np.isclose(group["plunger_offset_mm"], reference_offset_mm)
        ]
        if reference.empty:
            marker_name = str(group["marker_name"].iloc[0])
            raise ValueError(
                f"Missing {reference_offset_mm:g} mm plunger-offset reference for {marker_name}"
            )
        reference_phase_deg = _circular_mean_deg(reference["phase_deg"].to_numpy())
        response = group.copy()
        response["reference_phase_deg"] = reference_phase_deg
        response["phase_delta_deg"] = _wrap180(
            response["phase_deg"].to_numpy() - reference_phase_deg
        )
        rows.append(response)

    output = pd.concat(rows, ignore_index=True)
    for column in ("dataset_id", "tune_position"):
        if column not in output:
            output[column] = pd.NA
    return (
        output.loc[:, OUTPUT_COLUMNS]
        .sort_values(["marker_name", "plunger_offset_mm", "source_file"], kind="mergesort")
        .reset_index(drop=True)
    )


def _plunger_offset_mm_from_source_file(source_file: object) -> float | None:
    source_text = str(source_file)
    if _ZERO_OFFSET_PATTERN.search(source_text) is not None:
        return 0.0
    match = _OFFSET_PATTERN.search(source_text)
    return None if match is None else float(match.group("offset"))


def _circular_mean_deg(phases_deg: np.ndarray) -> float:
    radians = np.deg2rad(phases_deg.astype(float))
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * radians)))))


def _wrap180(phases_deg: np.ndarray) -> np.ndarray:
    return (phases_deg + 180.0) % 360.0 - 180.0
