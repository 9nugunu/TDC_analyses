"""EM field phase plots aligned with an inferred TDC half-section."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deflector_tuning.data_loading.field_profiles import (
    FieldPhaseExport,
    FieldPhaseTrace,
    FieldProfileExport,
    FieldProfileTrace,
    load_field_phase_export,
    load_field_profile_export,
)

from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    match_legend_text_colors_to_handles,
    save_figure,
)

DEFAULT_REGULAR_CELL_COUNT = 9
IRIS_GUIDE_COLOR = "#c77855"
REGULAR_CELL_GUIDE_COLOR = "#4f8d63"
PHASE_GUIDE_COLOR = IRIS_GUIDE_COLOR
PHASE_STRUCTURE_Y_MIN = -270.0
PHASE_STRUCTURE_Y_MAX = 235.0
PHASE_STRUCTURE_AXIS_Y = -262.0
PHASE_STRUCTURE_CELL_Y = -210.0
PROFILE_STRUCTURE_BOTTOM_AXIS_FRACTION = 0.0
PROFILE_STRUCTURE_BAND_AXIS_FRACTION = (PHASE_STRUCTURE_CELL_Y - PHASE_STRUCTURE_AXIS_Y) / (
    PHASE_STRUCTURE_Y_MAX - PHASE_STRUCTURE_Y_MIN
)
PROFILE_STRUCTURE_DATA_GAP_AXIS_FRACTION = 0.06
PROFILE_STRUCTURE_IRIS_HEIGHT_FRACTION = 0.42
PROFILE_TRACE_LINE_WIDTH = 3.0
PROFILE_GUIDE_LINE_WIDTH = 3.0
PROFILE_GUIDE_LABEL_X_OFFSET_FRACTION = 0.008


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
    match_legend_text_colors_to_handles(legend)

    return save_figure(fig, output_path, config)


def plot_field_profile_with_tdc_structure(
    export: FieldProfileExport,
    output_path: str | Path,
    *,
    regular_cell_count: int = DEFAULT_REGULAR_CELL_COUNT,
    config: PlotConfig | None = None,
) -> Path:
    """Plot one-dimensional E/H field profiles with TDC structure guides."""

    if not export.traces:
        raise ValueError("field profile export has no traces")

    config = _fixed_canvas_config(config or PlotConfig())
    apply_plot_style(config)

    z_min = min(float(np.min(trace.z_mm)) for trace in export.traces)
    z_max = max(float(np.max(trace.z_mm)) for trace in export.traces)
    data_y_min = min(float(np.min(trace.values)) for trace in export.traces)
    data_y_max = max(float(np.max(trace.values)) for trace in export.traces)
    data_span = data_y_max - data_y_min
    y_padding = max(data_span * 0.08, 1.0 if data_y_max == data_y_min else 0.0)
    y_min = data_y_min - y_padding
    y_max = data_y_max + y_padding

    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    if _has_tdc_structure_parameters(export.parameters):
        data_display_span = (y_max - data_y_min) if data_span > 0.0 else max(abs(data_y_max), 1.0)
        structure_fraction = (
            PROFILE_STRUCTURE_BOTTOM_AXIS_FRACTION
            + PROFILE_STRUCTURE_BAND_AXIS_FRACTION
            + PROFILE_STRUCTURE_DATA_GAP_AXIS_FRACTION
        )
        profile_axis_span = data_display_span / (1.0 - structure_fraction)
        band_span = profile_axis_span * PROFILE_STRUCTURE_BAND_AXIS_FRACTION
        axis_y = data_y_min - (
            PROFILE_STRUCTURE_DATA_GAP_AXIS_FRACTION + PROFILE_STRUCTURE_BAND_AXIS_FRACTION
        ) * profile_axis_span
        iris_y = axis_y + band_span * PROFILE_STRUCTURE_IRIS_HEIGHT_FRACTION
        cell_y = axis_y + band_span
        y_min = min(y_min, axis_y - PROFILE_STRUCTURE_BOTTOM_AXIS_FRACTION * profile_axis_span)
        _draw_tdc_half_section_band(
            ax,
            export.parameters,
            z_min=z_min,
            z_max=z_max,
            regular_cell_count=regular_cell_count,
            config=config,
            axis_y=axis_y,
            iris_y=iris_y,
            cell_y=cell_y,
        )
    _draw_profile_guides(
        ax,
        export.parameters,
        regular_cell_count=regular_cell_count,
        z_min=z_min,
        z_max=z_max,
        y_min=y_min,
        y_max=y_max,
        guide_mode=_profile_guide_mode(export.traces),
        config=config,
    )
    for trace in export.traces:
        (line,) = ax.plot(
            trace.z_mm,
            trace.values,
            label=_profile_legend_label(trace),
            color=_profile_color(trace),
            linestyle="--" if trace.value_kind == "phase" and trace.field_kind == "h" else "-",
            linewidth=PROFILE_TRACE_LINE_WIDTH,
        )
        line.set_gid("field_profile_trace")

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color="0.88", linewidth=0.8)
    apply_axis_text_style(
        ax,
        xlabel="Z [mm]",
        ylabel=_profile_y_label(export.traces),
        title="TDC field profile with structure guides",
        config=config,
        compact=False,
    )
    legend = ax.legend(
        frameon=True,
        loc="best",
        fontsize=config.compact_legend_size,
        facecolor="white",
        edgecolor="0.6",
        framealpha=0.72,
    )
    apply_legend_text_style(legend, config)
    match_legend_text_colors_to_handles(legend)
    return save_figure(fig, output_path, config)


def _fixed_canvas_config(config: PlotConfig) -> PlotConfig:
    """Keep sibling profile figures at the same final pixel dimensions."""

    if config.save_bbox_inches is None:
        return config
    return replace(config, save_bbox_inches=None)


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


def _profile_legend_label(trace: FieldProfileTrace) -> str:
    symbol = "E" if trace.field_kind.lower().startswith("e") else "H" if trace.field_kind.lower().startswith("h") else "field"
    suffix = "phase" if trace.value_kind == "phase" else "profile"
    component = f"_{trace.component.lower()}" if trace.component != "?" and symbol in {"E", "H"} else ""
    return f"${symbol}{component}$ {suffix}" if symbol in {"E", "H"} else trace.label


def _profile_color(trace: FieldProfileTrace) -> str:
    return {"e": "#1f5f99", "h": "#b54a00"}.get(trace.field_kind.lower()[:1], "0.25")


def _profile_y_label(traces: tuple[FieldProfileTrace, ...]) -> str:
    value_kinds = {trace.value_kind for trace in traces}
    if value_kinds == {"phase"}:
        return "Field phase [deg]"
    if value_kinds == {"real"}:
        return "Field component [a.u.]"
    return "Profile value"


def _draw_tdc_half_section_band(
    ax: plt.Axes,
    parameters: dict[str, float],
    *,
    z_min: float,
    z_max: float,
    regular_cell_count: int,
    config: PlotConfig,
    axis_y: float = -262.0,
    iris_y: float = -240.0,
    cell_y: float = -210.0,
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


def _regular_cell_center_guides(
    *,
    cell_length: float,
    iris_thickness: float,
    regular_cell_count: int,
) -> list[tuple[float, str]]:
    guides = []
    cursor = 0.0
    for cell_index in range(1, regular_cell_count + 1):
        regular_start = cursor + iris_thickness
        regular_center = regular_start + cell_length / 2.0
        guides.append((regular_center, f"R{cell_index}"))
        cursor += iris_thickness + cell_length
    return guides


def _iris_center_guides(
    *,
    cell_length: float,
    iris_thickness: float,
    regular_cell_count: int,
) -> list[tuple[float, str]]:
    guides = [(iris_thickness / 2.0, "IN iris")]
    for iris_index in range(1, regular_cell_count):
        iris_center = iris_index * (iris_thickness + cell_length) + iris_thickness / 2.0
        guides.append((iris_center, f"I{iris_index}"))
    final_iris_start = regular_cell_count * (iris_thickness + cell_length)
    guides.append((final_iris_start + iris_thickness / 2.0, "OUT iris"))
    return guides


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
            ax.axvline(guide_z, color=PHASE_GUIDE_COLOR, linestyle=":", linewidth=3.2, alpha=0.98)
            ax.text(
                guide_z - label_offset,
                218.0,
                label,
                ha="right",
                va="top",
                fontsize=11.0,
                color=PHASE_GUIDE_COLOR,
                fontweight="bold",
            )


def _draw_profile_guides(
    ax: plt.Axes,
    parameters: dict[str, float],
    *,
    regular_cell_count: int,
    z_min: float,
    z_max: float,
    y_min: float,
    y_max: float,
    guide_mode: str = "phase",
    config: PlotConfig | None = None,
) -> None:
    config = config or PlotConfig()
    cell_length = parameters.get("d")
    iris_thickness = parameters.get("t")
    if cell_length is None or iris_thickness is None:
        return
    label_y = (y_min + y_max) / 2.0
    label_offset = (z_max - z_min) * PROFILE_GUIDE_LABEL_X_OFFSET_FRACTION
    guides = _profile_guides(
        cell_length=cell_length,
        iris_thickness=iris_thickness,
        regular_cell_count=regular_cell_count,
        guide_mode=guide_mode,
    )
    guide_color = _profile_guide_color(guide_mode)
    for guide_z, label in guides:
        if z_min <= guide_z <= z_max:
            guide_line = ax.axvline(
                guide_z,
                color=guide_color,
                linestyle=":",
                linewidth=PROFILE_GUIDE_LINE_WIDTH,
                alpha=0.8,
            )
            guide_line.set_gid("profile_structure_guide")
            guide_label = ax.text(
                guide_z - label_offset,
                label_y,
                label,
                ha="right",
                va="center",
                fontsize=config.compact_annotation_size,
                color=guide_color,
                fontweight="bold",
            )
            guide_label.set_gid("profile_structure_guide_label")


def _profile_guides(
    *,
    cell_length: float,
    iris_thickness: float,
    regular_cell_count: int,
    guide_mode: str,
) -> list[tuple[float, str]]:
    if guide_mode == "h_cell_centers":
        return _regular_cell_center_guides(
            cell_length=cell_length,
            iris_thickness=iris_thickness,
            regular_cell_count=regular_cell_count,
        )
    if guide_mode == "e_iris_centers":
        return _iris_center_guides(
            cell_length=cell_length,
            iris_thickness=iris_thickness,
            regular_cell_count=regular_cell_count,
        )
    return _phase_curve_guides(
        cell_length=cell_length,
        iris_thickness=iris_thickness,
        regular_cell_count=regular_cell_count,
    )


def _profile_guide_mode(traces: tuple[FieldProfileTrace, ...]) -> str:
    if traces and all(trace.field_kind.lower().startswith("h") for trace in traces):
        return "h_cell_centers"
    if traces and any(trace.field_kind.lower().startswith("e") for trace in traces):
        return "e_iris_centers"
    return "phase"


def _profile_guide_color(guide_mode: str) -> str:
    if guide_mode == "h_cell_centers":
        return REGULAR_CELL_GUIDE_COLOR
    if guide_mode == "e_iris_centers":
        return IRIS_GUIDE_COLOR
    return PHASE_GUIDE_COLOR


def _has_tdc_structure_parameters(parameters: dict[str, float]) -> bool:
    return all(name in parameters for name in ("d", "t", "a", "b"))
