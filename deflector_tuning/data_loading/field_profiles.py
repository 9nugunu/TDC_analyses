"""Load CST one-dimensional field phase and profile exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAMETER_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)")


@dataclass(frozen=True)
class FieldPhaseTrace:
    """One CST field-phase trace sampled along Z."""

    label: str
    field_kind: str
    component: str
    z_mm: np.ndarray
    phase_deg: np.ndarray


@dataclass(frozen=True)
class FieldProfileTrace:
    """One CST field profile trace sampled along Z."""

    label: str
    field_kind: str
    component: str
    value_kind: str
    z_mm: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class FieldPhaseExport:
    """Parsed CST field-phase export with repeated header/data blocks."""

    parameters: dict[str, float]
    traces: tuple[FieldPhaseTrace, ...]


@dataclass(frozen=True)
class FieldProfileExport:
    """Parsed CST field profile export with repeated header/data blocks."""

    parameters: dict[str, float]
    traces: tuple[FieldProfileTrace, ...]


def load_field_phase_export(path: str | Path) -> FieldPhaseExport:
    """Load a CST-style field phase text export containing one or more traces."""

    parameters: dict[str, float] = {}
    traces: list[FieldPhaseTrace] = []
    current_header: str | None = None
    current_rows: list[tuple[float, float]] = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#Parameters"):
            _append_trace(traces, current_header, current_rows)
            current_header = None
            current_rows = []
            parameters.update(_parse_parameters(line))
            continue
        if line.startswith('#"Z / mm"'):
            _append_trace(traces, current_header, current_rows)
            current_header = line
            current_rows = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        current_rows.append((float(parts[0]), float(parts[1])))

    _append_trace(traces, current_header, current_rows)
    if not traces:
        raise ValueError(f"No field phase traces found in {path}")
    return FieldPhaseExport(parameters=parameters, traces=tuple(traces))


def load_field_profile_export(path: str | Path) -> FieldProfileExport:
    """Load a CST-style one-dimensional field profile text export."""

    parameters: dict[str, float] = {}
    traces: list[FieldProfileTrace] = []
    current_header: str | None = None
    current_rows: list[tuple[float, float]] = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#Parameters"):
            _append_profile_trace(traces, current_header, current_rows)
            current_header = None
            current_rows = []
            parameters.update(_parse_parameters(line))
            continue
        if line.startswith('#"Z / mm"'):
            _append_profile_trace(traces, current_header, current_rows)
            current_header = line
            current_rows = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        current_rows.append((float(parts[0]), float(parts[1])))

    _append_profile_trace(traces, current_header, current_rows)
    if not traces:
        raise ValueError(f"No field profile traces found in {path}")
    return FieldProfileExport(parameters=parameters, traces=tuple(traces))


def _append_trace(
    traces: list[FieldPhaseTrace],
    header: str | None,
    rows: list[tuple[float, float]],
) -> None:
    if header is None or not rows:
        return
    data = np.asarray(rows, dtype=float)
    field_kind = _field_kind(header)
    component = _component(header)
    traces.append(
        FieldPhaseTrace(
            label=f"{field_kind.upper()} field {component} phase",
            field_kind=field_kind,
            component=component,
            z_mm=data[:, 0],
            phase_deg=data[:, 1],
        )
    )


def _append_profile_trace(
    traces: list[FieldProfileTrace],
    header: str | None,
    rows: list[tuple[float, float]],
) -> None:
    if header is None or not rows:
        return
    data = np.asarray(rows, dtype=float)
    field_kind = _field_kind(header)
    component = _component(header)
    value_kind = _value_kind(header)
    label = f"{field_kind.upper()} field {component} {value_kind}"
    traces.append(
        FieldProfileTrace(
            label=label,
            field_kind=field_kind,
            component=component,
            value_kind=value_kind,
            z_mm=data[:, 0],
            values=data[:, 1],
        )
    )


def _parse_parameters(line: str) -> dict[str, float]:
    return {match.group(1): float(match.group(2)) for match in PARAMETER_PATTERN.finditer(line)}


def _field_kind(header: str) -> str:
    match = re.search(r'"([eh])-field', header, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "field"


def _component(header: str) -> str:
    match = re.search(r"_([XYZ]) \(Z\)(?:_phase)?", header)
    return match.group(1) if match else "?"


def _value_kind(header: str) -> str:
    return "phase" if "_phase" in header.lower() else "real"
