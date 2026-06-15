"""Shared visualization configuration and save helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    """Shared plotting defaults for deflector-tuning visualizations."""

    dpi: int = 300
    title_size: int = 11
    compact_title_size: int = 8
    label_size: int = 8
    compact_label_size: int = 7
    annotation_size: int = 7
    compact_annotation_size: int = 6
    line_width: float = 0.8
    marker_size: int = 26
    compact_marker_size: int = 18
    figure_size: tuple[float, float] = (4.2, 4.2)
    overview_panel_size: tuple[float, float] = (4.8, 4.4)
    font_family: tuple[str, ...] = ("Pretendard", "Noto Sans", "Malgun Gothic", "DejaVu Sans")


def apply_plot_style(config: PlotConfig | None = None) -> None:
    """Apply shared Matplotlib rcParams for project figures."""

    config = config or PlotConfig()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(config.font_family),
            "axes.unicode_minus": False,
            "savefig.dpi": config.dpi,
        }
    )


def save_figure(fig, output_path: str | Path, config: PlotConfig | None = None) -> Path:
    """Save a Matplotlib figure with the shared PNG policy."""

    config = config or PlotConfig()
    apply_plot_style(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    return path
