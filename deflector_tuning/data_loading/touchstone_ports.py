"""Touchstone port-order correction helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def write_s2p_with_ports_swapped(source_path: str | Path, output_path: str | Path) -> Path:
    """Write a two-port Touchstone file with port 1 and port 2 exchanged.

    For Touchstone v1 ``.s2p`` rows, values are ordered as
    ``S11, S21, S12, S22``. Swapping port numbers maps them to
    ``S22, S12, S21, S11``.
    """

    source = Path(source_path)
    output = Path(output_path)
    if source.suffix.lower() != ".s2p" or output.suffix.lower() != ".s2p":
        raise ValueError("Port swapping is only supported for .s2p files")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    swapped_lines = [_swap_line(line) for line in _swap_port_assignment_comments(lines)]
    output.write_text("\n".join(swapped_lines) + "\n", encoding="utf-8")
    return output


def write_folder_with_s2p_ports_swapped(source_folder: str | Path, output_folder: str | Path) -> list[Path]:
    """Copy a folder, swapping port order for direct child ``.s2p`` files."""

    source = Path(source_folder)
    output = Path(output_folder)
    if not source.is_dir():
        raise NotADirectoryError(source)
    output.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for child in sorted(source.iterdir(), key=lambda item: item.name.lower()):
        target = output / child.name
        if child.is_dir():
            continue
        if child.suffix.lower() == ".s2p":
            written.append(write_s2p_with_ports_swapped(child, target))
        else:
            shutil.copy2(child, target)
            written.append(target)
    return written


def _swap_port_assignment_comments(lines: list[str]) -> list[str]:
    assignments: dict[int, str] = {}
    pattern = re.compile(r"^(!\s*Touchstone port\s+)([12])(\s*=.*)$", flags=re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match is None:
            continue
        assignments[int(match.group(2))] = match.group(3)

    if set(assignments) != {1, 2}:
        return lines

    swapped: list[str] = []
    for line in lines:
        match = pattern.match(line)
        if match is None:
            swapped.append(line)
            continue
        port = int(match.group(2))
        other_port = 2 if port == 1 else 1
        swapped.append(f"{match.group(1)}{port}{assignments[other_port]}")
    return swapped


def _swap_line(line: str) -> str:
    stripped = line.lstrip()
    if not stripped or stripped.startswith(("!", "#")):
        return line

    if "!" in line:
        numeric_part, comment = line.split("!", maxsplit=1)
        comment_suffix = f" !{comment}"
    else:
        numeric_part = line
        comment_suffix = ""

    fields = numeric_part.split()
    if len(fields) != 9:
        return line

    swapped_fields = [
        fields[0],
        fields[7],
        fields[8],
        fields[5],
        fields[6],
        fields[3],
        fields[4],
        fields[1],
        fields[2],
    ]
    return " ".join(swapped_fields) + comment_suffix
