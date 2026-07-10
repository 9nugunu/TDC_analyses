from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

AnalysisPaths = OrderedDict[str, Path]
FigurePaths = OrderedDict[str, OrderedDict[str, Path]]
DetectionValue = bool | int | str
DetectionReport = dict[str, dict[str, DetectionValue]]


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    tables: AnalysisPaths
    figures: FigurePaths
    analysis_modes: tuple[str, ...]
    manifest_path: Path
