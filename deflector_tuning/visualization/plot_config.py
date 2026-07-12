"""Shared visualization configuration and save helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt

from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS

MATPLOTLIB_MATHTEXT_LOGGER = "matplotlib.mathtext"

BEST_MARKER_COLOR = "#c51b7d"
DEFAULT_DESIGN_POINT_BY_AXIS = DEFAULT_PROJECT_DEFAULTS.design_point_by_axis
REFERENCE_GUIDE_ALPHA = 0.85
REFERENCE_GUIDE_COLOR = "0.45"
REFERENCE_GUIDE_LABEL_COLOR = "0.35"
REFERENCE_GUIDE_LINESTYLE = "--"
DESIGN_REFERENCE_LINEWIDTH = 0.9
IDEAL_PHASE_GUIDE_ANGLES_DEG = DEFAULT_PROJECT_DEFAULTS.ideal_phase_guide_angles_deg


@dataclass(frozen=True)
class PlotConfig:
    """Shared plotting defaults for deflector-tuning visualizations."""

    dpi: int = 300
    title_size: int = 20
    compact_title_size: int = 16
    label_size: int = 17
    compact_label_size: int = 14
    tick_size: int = 14
    compact_tick_size: int = 13
    annotation_size: int = 13
    compact_annotation_size: int = 11
    legend_size: int = 13
    compact_legend_size: int = 12
    contour_line_width: float = 1.6
    contour_label_size: int = 12
    contour_label_weight: str = "bold"
    contour_line_alpha: float = 0.85
    contour_error_cmap: str = "RdYlGn_r"
    contour_signed_cmap: str = "coolwarm"
    contour_magnitude_cmap: str = "viridis"
    contour_light_line_color: str = "white"
    contour_dark_line_color: str = "black"
    contour_luminance_threshold: float = 0.54
    label_weight: str = "bold"
    tick_weight: str = "bold"
    title_weight: str = "bold"
    legend_weight: str = "bold"
    math_bold: bool = True
    line_width: float = 2.4
    marker_size: int = 52
    compact_marker_size: int = 32
    figure_size: tuple[float, float] = (5.6, 5.6)
    overview_panel_size: tuple[float, float] = (5.8, 5.2)
    font_family: tuple[str, ...] = ("Pretendard", "Noto Sans", "Malgun Gothic", "DejaVu Sans")
    save_bbox_inches: str | None = "tight"
    save_pad_inches: float = 0.1
    design_point_by_axis: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_DESIGN_POINT_BY_AXIS))
    ideal_phase_guide_angles_deg: tuple[float, ...] = IDEAL_PHASE_GUIDE_ANGLES_DEG


def apply_plot_style(config: PlotConfig | None = None) -> None:
    """Apply shared Matplotlib rcParams for project figures."""

    config = config or PlotConfig()
    suppress_matplotlib_mathtext_info_logs()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(config.font_family),
            "axes.unicode_minus": False,
            "axes.titlesize": config.title_size,
            "axes.titleweight": config.title_weight,
            "axes.labelsize": config.label_size,
            "axes.labelweight": config.label_weight,
            "xtick.labelsize": config.tick_size,
            "ytick.labelsize": config.tick_size,
            "legend.fontsize": config.legend_size,
            "mathtext.default": "regular",
            "savefig.dpi": config.dpi,
        }
    )


def suppress_matplotlib_mathtext_info_logs() -> None:
    """Hide noisy mathtext font-substitution INFO messages during rendering."""

    logging.getLogger(MATPLOTLIB_MATHTEXT_LOGGER).setLevel(logging.WARNING)


def math_label(expression: str, *, bold: bool = True) -> str:
    """Return a mathtext label, optionally using bold-compatible math syntax."""

    if expression.startswith("$") and expression.endswith("$"):
        label = expression
    else:
        label = f"${expression}$"
    return bold_math(label) if bold else label


def bold_math(label: str) -> str:
    r"""Best-effort bold mathtext conversion for simple project labels.

    Matplotlib mathtext does not inherit ``fontweight='bold'`` into all math
    glyphs. This helper converts common label forms such as ``$S_{11}$`` and
    ``$f_{2\pi/3}$`` into explicit ``\mathbf{...}`` mathtext segments.
    """

    if not (label.startswith("$") and label.endswith("$")):
        return label
    body = label[1:-1]
    protected_commands = {
        r"\partial": "§∂§",
        r"\nabla": "§∇§",
        r"\phi": "§φ§",
        r"\beta": "§β§",
        r"\pi": "§π§",
    }
    for command, token in protected_commands.items():
        body = body.replace(command, token)
    body = re.sub(r"([A-Za-z])", r"\\mathbf{\1}", body)
    body = re.sub(r"(\d+)", r"\\mathbf{\1}", body)
    body = body.replace("§∂§", r"\mathbf{\partial}")
    body = body.replace("§∇§", r"\mathbf{\nabla}")
    body = body.replace("§φ§", r"\mathbf{\phi}")
    body = body.replace("§β§", r"\mathbf{\beta}")
    body = body.replace("§π§", r"\mathbf{\pi}")
    return f"${body}$"


def apply_axis_text_style(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    config: PlotConfig | None = None,
    compact: bool = False,
) -> None:
    """Apply shared bold axis/title/tick styling."""

    config = config or PlotConfig()
    label_size = config.compact_label_size if compact else config.label_size
    tick_size = config.compact_tick_size if compact else config.tick_size
    title_size = config.compact_title_size if compact else config.title_size
    if xlabel is not None:
        ax.set_xlabel(_bold_label_if_math(xlabel, config), fontsize=label_size, fontweight=config.label_weight)
    if ylabel is not None:
        ax.set_ylabel(_bold_label_if_math(ylabel, config), fontsize=label_size, fontweight=config.label_weight)
    if title is not None:
        ax.set_title(_bold_label_if_math(title, config), fontsize=title_size, fontweight=config.title_weight)
    ax.tick_params(axis="both", labelsize=tick_size)
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight(config.tick_weight)


def apply_legend_text_style(legend, config: PlotConfig | None = None) -> None:
    """Apply shared legend text styling."""

    if legend is None:
        return
    config = config or PlotConfig()
    for text in legend.get_texts():
        text.set_fontweight(config.legend_weight)


def match_legend_text_colors_to_handles(legend) -> None:
    """Color each legend label with the matching legend handle color."""

    if legend is None:
        return
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
        color = getattr(handle, "get_color", lambda: None)()
        if color is not None:
            text.set_color(color)


def contour_contrast_color(cmap: str, config: PlotConfig | None = None) -> str:
    """Return a readable contour color for the average brightness of a colormap."""

    config = config or PlotConfig()
    colors = colormaps[cmap](np.linspace(0.08, 0.92, 17))[:, :3]
    luminance = colors @ np.array([0.2126, 0.7152, 0.0722])
    if float(luminance.mean()) >= config.contour_luminance_threshold:
        return config.contour_dark_line_color
    return config.contour_light_line_color


def save_figure(fig, output_path: str | Path, config: PlotConfig | None = None) -> Path:
    """Save a Matplotlib figure with the shared PNG policy."""

    config = config or PlotConfig()
    apply_plot_style(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=config.dpi,
        bbox_inches=config.save_bbox_inches,
        pad_inches=config.save_pad_inches,
    )
    return path


def _bold_label_if_math(label: str, config: PlotConfig) -> str:
    if config.math_bold and "$" in label:
        return _bold_math_segments(label)
    return label


def _bold_math_segments(text: str) -> str:
    parts = text.split("$")
    for index in range(1, len(parts), 2):
        parts[index] = bold_math(f"${parts[index]}$")[1:-1]
    return "$".join(parts)
