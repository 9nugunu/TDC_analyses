"""Project-specific scientific defaults used by analysis and visualizations."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, cast


@dataclass(frozen=True)
class ProjectDefaults:
    """Named assumptions for the current deflector-tuning study."""

    default_dispersion_subpath: Path
    design_point_by_axis: Mapping[str, float] = field(default_factory=dict)
    ideal_phase_guide_angles_deg: tuple[float, ...] = ()


DEFAULT_PROJECT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "project_defaults.toml"


def load_project_defaults(path: str | Path = DEFAULT_PROJECT_CONFIG_PATH) -> ProjectDefaults:
    """Load scientific analysis defaults from a TOML configuration file."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    analysis = _required_section(config, "analysis", config_path)
    visualization = _required_section(config, "visualization", config_path)
    dispersion_subpath = _required_text(analysis, "default_dispersion_subpath", config_path)
    design_points = _numeric_mapping(
        _required_section(visualization, "design_point_by_axis", config_path),
        "visualization.design_point_by_axis",
        config_path,
    )
    guide_angles = _numeric_angles(
        visualization.get("ideal_phase_guide_angles_deg"),
        config_path,
    )
    return ProjectDefaults(
        default_dispersion_subpath=Path(dispersion_subpath),
        design_point_by_axis=design_points,
        ideal_phase_guide_angles_deg=guide_angles,
    )


def _required_section(
    config: Mapping[str, object], name: str, config_path: Path
) -> Mapping[str, object]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Expected [{name}] section in project configuration: {config_path}")
    return cast(Mapping[str, object], section)


def _required_text(section: Mapping[str, object], name: str, config_path: Path) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty {name} in project configuration: {config_path}")
    return value


def _numeric_mapping(
    values: Mapping[str, object], name: str, config_path: Path
) -> dict[str, float]:
    if not values:
        raise ValueError(f"Expected at least one value in {name}: {config_path}")
    return {
        axis: _numeric_value(value, setting=f"{name}.{axis}", config_path=config_path)
        for axis, value in values.items()
    }


def _numeric_angles(values: object, config_path: Path) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Expected non-empty visualization.ideal_phase_guide_angles_deg list "
            f"in project configuration: {config_path}"
        )
    return tuple(
        _numeric_value(
            value,
            setting="visualization.ideal_phase_guide_angles_deg",
            config_path=config_path,
        )
        for value in cast(list[object], values)
    )


def _numeric_value(value: object, *, setting: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Expected numeric {setting} in project configuration: {config_path}"
        )
    return float(value)


DEFAULT_PROJECT_DEFAULTS = load_project_defaults()
