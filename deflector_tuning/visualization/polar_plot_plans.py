"""Build ordered polar output descriptions without rendering figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from deflector_tuning.visualization.marker_styles import MARKER_LABELS
from deflector_tuning.visualization.polar_grouping import (
    _grouping_mode,
    _is_no_port_extension_position,
    _iter_family_overlay_groups,
    _iter_position_groups,
    _kyhl_phase_pair_overlay_table,
    _kyhl_phase_pair_overlay_title,
    _position_output_stem,
    _position_plot_title,
)


NO_PORT_EXTENSION_FOLDER = "No_portExtension"


@dataclass(frozen=True)
class PolarPositionPlan:
    """Describe one per-position polar output before rendering."""

    key: str
    output_path: Path
    position_label: str
    position_table: pd.DataFrame
    title: str

    @property
    def kind(self) -> str:
        return "position"


@dataclass(frozen=True)
class PolarOverviewPlan:
    """Describe the multi-position polar overview before rendering."""

    key: str
    output_path: Path
    groups: list[tuple[str, pd.DataFrame]]
    grouping_mode: str
    title_prefix: str

    @property
    def kind(self) -> str:
        return "overview"


@dataclass(frozen=True)
class PolarFamilyOverlayPlan:
    """Describe a family overlay or marker-specific family overlay."""

    key: str
    output_path: Path
    family: str
    family_table: pd.DataFrame
    title: str
    markers: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "family_overlay"


@dataclass(frozen=True)
class PolarKyhlPairPlan:
    """Describe one Kyhl phase-pair overlay before rendering."""

    key: str
    output_path: Path
    pair_table: pd.DataFrame
    pair_label: str
    title: str

    @property
    def kind(self) -> str:
        return "kyhl_pair_overlay"


PolarPlotPlan: TypeAlias = (
    PolarPositionPlan
    | PolarOverviewPlan
    | PolarFamilyOverlayPlan
    | PolarKyhlPairPlan
)


def build_polar_plot_plans(
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    title_prefix: str = "Polar phase",
) -> list[PolarPlotPlan]:
    """Build polar output plans without creating figures or files."""

    folder = Path(output_dir)
    grouping_mode = _grouping_mode(marker_points)
    groups = list(_iter_position_groups(marker_points))
    plans: list[PolarPlotPlan] = []
    if len(groups) > 1:
        for position_label, position_table in groups:
            position_folder = (
                folder / NO_PORT_EXTENSION_FOLDER
                if _is_no_port_extension_position(position_table)
                else folder
            )
            plans.append(
                PolarPositionPlan(
                    key=position_label,
                    output_path=position_folder / f"{_position_output_stem(position_table, position_label, grouping_mode=grouping_mode)}.png",
                    position_label=position_label,
                    position_table=position_table,
                    title=_position_plot_title(
                        position_label,
                        position_table,
                        grouping_mode=grouping_mode,
                        title_prefix=title_prefix,
                    ),
                )
            )

    if grouping_mode not in {"grid_point", "sim_sweep"}:
        plans.append(
            PolarOverviewPlan(
                key="overview",
                output_path=folder / "all_positions.png",
                groups=groups,
                grouping_mode=grouping_mode,
                title_prefix=title_prefix,
            )
        )

    if grouping_mode == "tune_position":
        for family, family_table in _iter_family_overlay_groups(marker_points):
            plans.append(
                PolarFamilyOverlayPlan(
                    key=f"{family}_overlay",
                    output_path=folder / f"{family}_overlay.png",
                    family=family,
                    family_table=family_table,
                    title=f"{family.title()} overlay: {title_prefix}",
                )
            )
            plans.append(
                PolarFamilyOverlayPlan(
                    key=f"{family}_f_2pi3_overlay",
                    output_path=folder / f"{family}_f_2pi3_overlay.png",
                    family=family,
                    family_table=family_table,
                    title=f"{family.title()} {MARKER_LABELS['f_2pi3']} overlay: {title_prefix}",
                    markers=("f_2pi3",),
                )
            )

        kyhl_pair_table = _kyhl_phase_pair_overlay_table(marker_points)
        if not kyhl_pair_table.empty:
            for _, pair_table in kyhl_pair_table.groupby("_pair_order", sort=True):
                pair_label = str(pair_table["_pair_label"].iloc[0])
                family = str(pair_table["_pair_family"].iloc[0])
                start = float(pair_table["_pair_start"].iloc[0])
                end = float(pair_table["_pair_end"].iloc[0])
                plans.append(
                    PolarKyhlPairPlan(
                        key=f"kyhl_phase_{family}_overlay",
                        output_path=folder / f"kyhl_phase_{family}_overlay.png",
                        pair_table=pair_table,
                        pair_label=pair_label,
                        title=_kyhl_phase_pair_overlay_title(family, start, end),
                    )
                )
    return plans
