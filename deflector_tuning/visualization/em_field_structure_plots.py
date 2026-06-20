"""EM field phase plots aligned with an inferred TDC half-section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)

PARAMETER_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)")
DEFAULT_REGULAR_CELL_COUNT = 9
GUIDE_COLOR = "#7C00FF"


@dataclass(frozen=True)
class FieldPhaseTrace:
    """One CST field-phase trace sampled along Z."""

    label: str
    field_kind: str
    component: str
    z_mm: np.ndarray
    phase_deg: np.ndarray


@dataclass(frozen=True)
class FieldPhaseExport:
    """Parsed CST field-phase export with repeated header/data blocks."""

    parameters: dict[str, float]
    traces: tuple[FieldPhaseTrace, ...]


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


def plot_field_phase_with_tdc_structure(
    export: FieldPhaseExport,
    output_path: str | Path,
    *,
    regular_cell_count: int = DEFAULT_REGULAR_CELL_COUNT,
    config: PlotConfig | None = None,
) -> Path:
    """Plot E/H field phase traces under an inferred TDC half-section."""

    if not export.traces:
        raise ValueError("field phase export has no traces")

    config = config or PlotConfig()
    apply_plot_style(config)

    z_min = min(float(np.min(trace.z_mm)) for trace in export.traces)
    z_max = max(float(np.max(trace.z_mm)) for trace in export.traces)
    fig, phase_ax = plt.subplots(figsize=(12.4, 5.8))
    _draw_phase_traces(phase_ax, export.traces, config=config)
    _draw_tdc_half_section_band(
        phase_ax,
        export.parameters,
        z_min=z_min,
        z_max=z_max,
        regular_cell_count=regular_cell_count,
        config=config,
    )
    _draw_regular_group_guides(
        phase_ax,
        export.parameters,
        regular_cell_count=regular_cell_count,
        z_min=z_min,
        z_max=z_max,
    )

    phase_ax.set_xlim(z_min, z_max)
    phase_ax.set_ylim(-270.0, 235.0)
    phase_ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
    phase_ax.grid(True, color="0.88", linewidth=0.8)
    phase_ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    apply_axis_text_style(
        phase_ax,
        xlabel="Z [mm]",
        ylabel="Field phase [deg]",
        title=f"TDC field phase with coupler + {regular_cell_count} regular cells + coupler",
        config=config,
        compact=True,
    )
    legend = phase_ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=config.compact_legend_size,
        facecolor="white",
        edgecolor="0.6",
        framealpha=0.72,
    )
    apply_legend_text_style(legend, config)
    _apply_legend_line_text_colors(legend)

    return save_figure(fig, output_path, config)


def _draw_phase_traces(ax: plt.Axes, traces: tuple[FieldPhaseTrace, ...], *, config: PlotConfig) -> None:
    colors = {"e": "#1f5f99", "h": "#b54a00"}
    styles = {"e": "-", "h": "--"}
    for trace in traces:
        key = trace.field_kind.lower()[:1]
        ax.plot(
            trace.z_mm,
            trace.phase_deg,
            label=_trace_legend_label(trace),
            color=colors.get(key, "0.25"),
            linestyle=styles.get(key, "-"),
            linewidth=config.line_width,
        )


def _trace_legend_label(trace: FieldPhaseTrace) -> str:
    if trace.field_kind.lower().startswith("e"):
        return "$E_x$ phase"
    if trace.field_kind.lower().startswith("h"):
        return "$H_y$ phase"
    return trace.label


def _apply_legend_line_text_colors(legend) -> None:
    if legend is None:
        return
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
        color = getattr(handle, "get_color", lambda: None)()
        if color is not None:
            text.set_color(color)


def _draw_tdc_half_section_band(
    ax: plt.Axes,
    parameters: dict[str, float],
    *,
    z_min: float,
    z_max: float,
    regular_cell_count: int,
    config: PlotConfig,
) -> None:
    if regular_cell_count < 1:
        raise ValueError("regular_cell_count must be positive")
    pitch = parameters.get("d")
    iris_thickness = parameters.get("t")
    iris_radius = parameters.get("a")
    cell_radius = parameters.get("b")
    if pitch is None or iris_thickness is None or iris_radius is None or cell_radius is None:
        raise ValueError("TDC structure overlay requires d, t, a, and b parameters")

    segments = _structure_segments(
        z_min=z_min,
        z_max=z_max,
        cell_length=pitch,
        iris_thickness=iris_thickness,
        iris_radius=iris_radius,
        cell_radius=cell_radius,
        regular_cell_count=regular_cell_count,
    )
    axis_y = -262.0
    iris_y = -240.0
    cell_y = -210.0
    radius_to_y = {
        iris_radius: iris_y,
        cell_radius: cell_y,
    }
    ax.plot([z_min, z_max], [axis_y, axis_y], color="0.2", linewidth=0.9, zorder=0.8)
    for z0, z1, radius, kind, label in segments:
        y_top = radius_to_y.get(radius, iris_y)
        fill_color = _segment_fill_color(kind)
        ax.fill_between(
            [z0, z1],
            [axis_y, axis_y],
            [y_top, y_top],
            color=fill_color,
            alpha=0.82,
            linewidth=0,
            zorder=0.5,
        )
        ax.plot([z0, z1], [y_top, y_top], color="0.12", linewidth=1.05, zorder=0.9)
        ax.plot([z0, z0], [axis_y, y_top], color="0.35", linewidth=0.55, alpha=0.65, zorder=0.9)
        ax.plot([z1, z1], [axis_y, y_top], color="0.35", linewidth=0.55, alpha=0.65, zorder=0.9)
        if label and z1 - z0 > 0.42 * pitch:
            ax.text(
                (z0 + z1) / 2.0,
                axis_y + (y_top - axis_y) * 0.48,
                _structure_display_label(label),
                ha="center",
                va="center",
                fontsize=config.compact_annotation_size,
                fontweight=config.label_weight,
                zorder=1.0,
            )
    for transition_z, y0, y1 in _outline_verticals(segments):
        ax.plot(
            [transition_z, transition_z],
            [radius_to_y.get(y0, iris_y), radius_to_y.get(y1, cell_y)],
            color="0.12",
            linewidth=1.05,
            zorder=0.9,
        )


def _structure_segments(
    *,
    z_min: float,
    z_max: float,
    cell_length: float,
    iris_thickness: float,
    iris_radius: float,
    cell_radius: float,
    regular_cell_count: int,
) -> list[tuple[float, float, float, str, str]]:
    segments: list[tuple[float, float, float, str, str]] = []
    input_coupler_start = -cell_length
    input_coupler_end = 0.0
    _add_clipped_segment(segments, z_min, input_coupler_start, iris_radius, "beam", "beam", z_min, z_max)
    _add_clipped_segment(
        segments,
        input_coupler_start,
        input_coupler_end,
        cell_radius,
        "coupler",
        "Coupler IN",
        z_min,
        z_max,
    )

    cursor = input_coupler_end
    for cell_index in range(1, regular_cell_count + 1):
        _add_clipped_segment(segments, cursor, cursor + iris_thickness, iris_radius, "iris", "", z_min, z_max)
        cursor += iris_thickness
        _add_clipped_segment(
            segments,
            cursor,
            cursor + cell_length,
            cell_radius,
            "regular",
            f"R{cell_index}",
            z_min,
            z_max,
        )
        cursor += cell_length

    _add_clipped_segment(segments, cursor, cursor + iris_thickness, iris_radius, "iris", "", z_min, z_max)
    cursor += iris_thickness
    _add_clipped_segment(
        segments,
        cursor,
        cursor + cell_length,
        cell_radius,
        "coupler",
        "Coupler OUT",
        z_min,
        z_max,
    )
    cursor += cell_length
    _add_clipped_segment(segments, cursor, z_max, iris_radius, "beam", "beam", z_min, z_max)
    return sorted(segments, key=lambda item: (item[0], item[1]))


def _add_clipped_segment(
    segments: list[tuple[float, float, float, str, str]],
    start: float,
    end: float,
    radius: float,
    kind: str,
    label: str,
    z_min: float,
    z_max: float,
) -> None:
    clipped_start = max(z_min, start)
    clipped_end = min(z_max, end)
    if clipped_start < clipped_end:
        segments.append((clipped_start, clipped_end, radius, kind, label))


def _outline_verticals(segments: list[tuple[float, float, float, str, str]]) -> list[tuple[float, float, float]]:
    verticals = []
    for left, right in zip(segments, segments[1:], strict=False):
        if abs(left[1] - right[0]) > 1e-6:
            continue
        if abs(left[2] - right[2]) < 1e-9:
            continue
        verticals.append((left[1], min(left[2], right[2]), max(left[2], right[2])))
    return verticals


def _segment_fill_color(kind: str) -> str:
    return {
        "beam": "#e7e7e7",
        "coupler": "#d8e4f5",
        "regular": "#d9eadf",
        "iris": "#f8d8c2",
    }.get(kind, "0.9")


def _structure_display_label(label: str) -> str:
    return {
        "beam": "pipe",
        "Coupler IN": "Coupler \n IN",
        "Coupler OUT": "Coupler \n OUT",
    }.get(label, label)


def _regular_group_guides(
    *,
    cell_length: float,
    iris_thickness: float,
    regular_cell_count: int,
) -> list[tuple[float, str]]:
    guides = []
    cursor = 0.0
    for cell_index in range(1, regular_cell_count + 1):
        cursor += iris_thickness + cell_length
        if cell_index % 3 == 0 and cell_index < regular_cell_count:
            guides.append((cursor + iris_thickness / 2.0, f"R{cell_index}"))
    return guides


def _phase_curve_guides(
    *,
    cell_length: float,
    iris_thickness: float,
    regular_cell_count: int,
) -> list[tuple[float, str]]:
    input_iris_center = iris_thickness / 2.0
    final_iris_start = regular_cell_count * (iris_thickness + cell_length)
    final_iris_center = final_iris_start + iris_thickness / 2.0
    return [
        (input_iris_center, "IN iris"),
        *_regular_group_guides(
            cell_length=cell_length,
            iris_thickness=iris_thickness,
            regular_cell_count=regular_cell_count,
        ),
        (final_iris_center, "OUT iris"),
    ]


def _draw_regular_group_guides(
    ax: plt.Axes,
    parameters: dict[str, float],
    *,
    regular_cell_count: int,
    z_min: float,
    z_max: float,
) -> None:
    cell_length = parameters.get("d")
    iris_thickness = parameters.get("t")
    if cell_length is None or iris_thickness is None:
        return
    label_offset = 0.028 * (z_max - z_min)
    for guide_z, label in _phase_curve_guides(
        cell_length=cell_length,
        iris_thickness=iris_thickness,
        regular_cell_count=regular_cell_count,
    ):
        if z_min <= guide_z <= z_max:
            ax.axvline(guide_z, color=GUIDE_COLOR, linestyle=":", linewidth=3.2, alpha=0.98)
            ax.text(
                guide_z - label_offset,
                218.0,
                label,
                ha="right",
                va="top",
                fontsize=11.0,
                color=GUIDE_COLOR,
                fontweight="bold",
            )


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


def _parse_parameters(line: str) -> dict[str, float]:
    return {match.group(1): float(match.group(2)) for match in PARAMETER_PATTERN.finditer(line)}


def _field_kind(header: str) -> str:
    match = re.search(r'"([eh])-field', header, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "field"


def _component(header: str) -> str:
    match = re.search(r"_([XYZ]) \(Z\)_phase", header)
    return match.group(1) if match else "?"
