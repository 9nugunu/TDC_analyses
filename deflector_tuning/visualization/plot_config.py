"""Shared visualization configuration and save helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    """Shared plotting defaults for deflector-tuning visualizations."""

    dpi: int = 300
    title_size: int = 18
    compact_title_size: int = 13
    label_size: int = 15
    compact_label_size: int = 12
    tick_size: int = 12
    compact_tick_size: int = 11
    annotation_size: int = 11
    compact_annotation_size: int = 9
    legend_size: int = 11
    compact_legend_size: int = 10
    label_weight: str = "bold"
    tick_weight: str = "bold"
    title_weight: str = "bold"
    legend_weight: str = "bold"
    math_bold: bool = True
    line_width: float = 1.8
    marker_size: int = 52
    compact_marker_size: int = 32
    figure_size: tuple[float, float] = (5.6, 5.6)
    overview_panel_size: tuple[float, float] = (5.8, 5.2)
    font_family: tuple[str, ...] = ("Pretendard", "Noto Sans", "Malgun Gothic", "DejaVu Sans")


def apply_plot_style(config: PlotConfig | None = None) -> None:
    """Apply shared Matplotlib rcParams for project figures."""

    config = config or PlotConfig()
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
    body = body.replace(r"\pi", "§§")
    body = re.sub(r"([A-Za-z])", r"\\mathbf{\1}", body)
    body = re.sub(r"(\d+)", r"\\mathbf{\1}", body)
    body = body.replace("§§", r"\mathbf{\pi}")
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


def save_figure(fig, output_path: str | Path, config: PlotConfig | None = None) -> Path:
    """Save a Matplotlib figure with the shared PNG policy."""

    config = config or PlotConfig()
    apply_plot_style(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
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
