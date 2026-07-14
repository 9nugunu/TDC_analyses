"""Small Touchstone helpers.

Project default: RI data is the default format when the header omits the format.
Touchstone rows are normalized into complex S-parameter values for downstream
loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TouchstoneHeader:
    frequency_unit: str
    parameter: str
    data_format: str
    reference_ohm: float
    is_normalized: bool


@dataclass(frozen=True)
class TouchstoneData:
    header: TouchstoneHeader
    frequency: list[float]
    s_values: list[list[complex]]


def read_touchstone(path: str | Path) -> TouchstoneData:
    """Read a simple RI or DB Touchstone file."""

    touchstone_path = Path(path)
    header: TouchstoneHeader | None = None
    frequency: list[float] = []
    s_values: list[list[complex]] = []
    value_count = _value_count_from_suffix(touchstone_path)

    for raw_line in touchstone_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("#"):
            header = parse_touchstone_header(line)
            if header.data_format not in {"RI", "DB"}:
                raise ValueError("Only RI and DB Touchstone data are supported for now")
            continue
        if header is None:
            continue
        row = [float(item) for item in line.split()]
        expected_number_count = 1 + 2 * value_count
        if len(row) != expected_number_count:
            raise ValueError(
                f"Expected {expected_number_count} numeric values in {touchstone_path.name}; got {len(row)}"
            )
        frequency.append(row[0])
        s_values.append(
            [
                _pair_to_complex(row[index], row[index + 1], header.data_format)
                for index in range(1, len(row), 2)
            ]
        )

    if header is None:
        raise ValueError(f"Missing Touchstone header in {touchstone_path}")
    return TouchstoneData(header=header, frequency=frequency, s_values=s_values)


def parse_touchstone_header(line: str) -> TouchstoneHeader:
    """Parse a Touchstone option line such as ``# GHz S RI R 50``.

    Defaults follow the project convention for this refactor:
    - data format: RI
    - reference impedance: 50 ohm when the header omits ``R``

    If the header says ``R 0``, keep ``reference_ohm=0.0`` and mark the data as
    not normalized.
    """

    words = line.strip().split()
    if not words or words[0] != "#":
        raise ValueError(f"Touchstone header must start with '#': {line!r}")

    tokens = [word.upper() for word in words[1:]]
    frequency_unit = _format_frequency_unit(tokens[0]) if tokens else "GHz"
    parameter = tokens[1] if len(tokens) >= 2 else "S"
    data_format = _find_data_format(tokens)
    reference_ohm = _find_reference_ohm(tokens)
    return TouchstoneHeader(
        frequency_unit=frequency_unit,
        parameter=parameter,
        data_format=data_format,
        reference_ohm=reference_ohm,
        is_normalized=reference_ohm != 0.0,
    )


def _strip_inline_comment(line: str) -> str:
    return line.split("!", maxsplit=1)[0]


def _value_count_from_suffix(path: Path) -> int:
    suffix = path.suffix.lower()
    if not suffix.startswith((".s", ".y", ".z")) or not suffix.endswith("p"):
        raise ValueError(
            f"Expected Touchstone extension like .s1p, .s2p, .y1p, or .z1p; got {path.name}"
        )
    port_count = int(suffix[2:-1])
    return port_count * port_count


def _pair_to_complex(first_value: float, second_value: float, data_format: str) -> complex:
    if data_format == "RI":
        return complex(first_value, second_value)
    if data_format == "DB":
        magnitude = 10.0 ** (first_value / 20.0)
        phase_rad = math.radians(second_value)
        return complex(magnitude * math.cos(phase_rad), magnitude * math.sin(phase_rad))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def _find_data_format(tokens: list[str]) -> str:
    for token in tokens:
        if token in {"RI", "MA", "DB"}:
            return token
    return "RI"


def _find_reference_ohm(tokens: list[str]) -> float:
    for index, token in enumerate(tokens[:-1]):
        if token == "R":
            return float(tokens[index + 1])
    return 50.0


def _format_frequency_unit(token: str) -> str:
    units = {
        "HZ": "Hz",
        "KHZ": "kHz",
        "MHZ": "MHz",
        "GHZ": "GHz",
    }
    return units.get(token, token)
