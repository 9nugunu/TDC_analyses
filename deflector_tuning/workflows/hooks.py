from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from deflector_tuning.dispersion import DispersionOutputs
from deflector_tuning.visualization.em_field_structure_plots import FieldProfileExport
from deflector_tuning.workflows.models import (
    AnalysisPaths,
    DetectionReport,
    FigurePaths,
)


class ManifestWriter(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        sparameter_path: Path,
        dispersion_path: Path,
        marker_role: str,
        modes: tuple[str, ...],
        detection: DetectionReport,
        tables: AnalysisPaths,
        figures: FigurePaths,
    ) -> None: ...


class ProfilePlotter(Protocol):
    def __call__(self, export: FieldProfileExport, output_path: str | Path) -> Path: ...


class DispersionProcessor(Protocol):
    def __call__(
        self,
        path: str | Path,
        output_dir: str | Path | None = None,
    ) -> DispersionOutputs: ...


class DispersionPlotter(Protocol):
    def __call__(
        self,
        dispersion_table: pd.DataFrame,
        output_path: str | Path,
        *,
        mode_indices: tuple[int, ...] | None = None,
        title: str,
    ) -> Path: ...
