"""LPS publication typography adapted to the placed size of RF figures.

The checked-in preset records the actual LPS PlotConfig and source hashes.
Its reconstruction-specific charge cut and cumulative colors are deliberately
not applied to RF amplitudes. Existing plotted values are never transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from cycler import cycler

PRESET_PATH = Path(__file__).resolve().parents[2] / "config/lps_publication_style.json"


@dataclass(frozen=True)
class PaperStyle:
    label: float
    tick: float
    legend: float
    line: float
    marker: float
    dpi: int
    pad: float


def apply_lps_style() -> PaperStyle:
    """Apply the recorded LPS bold-serif/TeX settings at publication size."""
    record = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    source, app = record["source_config"], record["application"]
    scale = app["label_points"] / source["label_size"]
    style = PaperStyle(app["label_points"], source["font_size"] * .8 * scale,
                       app["label_points"] * record["legend_to_label_ratio"],
                       app["line_points"], app["marker_points"], source["figure_dpi"],
                       app["save_pad_inches"])
    plt.rcParams.update({
        "text.usetex": True, "text.latex.preamble": record["tex_preamble"],
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.weight": source["font_weight"], "font.size": style.tick,
        "axes.labelsize": style.label, "axes.labelweight": source["label_weight"],
        "axes.titlesize": style.label, "axes.titleweight": source["title_weight"],
        "xtick.labelsize": style.tick, "ytick.labelsize": style.tick,
        "xtick.major.width": app["major_tick_width"], "ytick.major.width": app["major_tick_width"],
        "axes.linewidth": app["spine_width"], "axes.unicode_minus": False,
        "axes.prop_cycle": cycler(color=app["line_colors"]),
        "lines.linewidth": style.line, "lines.markersize": style.marker,
        "legend.fontsize": style.legend, "legend.frameon": True,
        "legend.framealpha": .9, "legend.edgecolor": ".7",
        "legend.fancybox": False, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": style.dpi,
    })
    return style


def field_colormap():
    """Use the LPS color family on an unchanged linear RF-amplitude scale."""
    record = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    colors = record["source_reconstruction_style"]["colors"]
    return LinearSegmentedColormap.from_list("lps_rf_linear", list(reversed(colors))).with_extremes(bad=".92")


def finish_axes(ax, *, scale: float = 1.0, grid: bool = True) -> None:
    """Apply LPS axes/legend formatting without touching values or limits."""
    style = apply_lps_style()
    if not ax.axison:
        return
    ax.minorticks_on()
    ax.grid(False, which="both")
    if grid:
        ax.grid(which="major", color=".70", lw=.55, ls="-", alpha=.7)
        ax.grid(which="minor", color=".82", lw=.45, ls=":", alpha=.6)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", labelsize=style.tick*scale, width=1.1, length=3.5)
    ax.tick_params(which="minor", width=.65, length=2)
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_weight("bold")
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_size(style.label*scale)
        label.set_weight("bold")
    ax.title.set_size(style.label*scale)
    ax.title.set_weight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_visible(True)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(.9)
        legend.get_frame().set_edgecolor(".7")
        for text in legend.get_texts():
            text.set_size(style.legend*scale)
            text.set_weight("bold")


def prepare_text(fig) -> None:
    """Use TeX-safe literal labels while preserving embedded math expressions."""
    for text in fig.findobj(match=Text):
        parts = text.get_text().split("$")
        for index in range(0, len(parts), 2):
            parts[index] = re.sub(r"(?<!\\)_", r"\_", parts[index])
            parts[index] = re.sub(r"(?<!\\)%", r"\%", parts[index])
        text.set_text("$".join(parts))
        text.set_weight("bold")
        text.set_family("serif")
        text.set_usetex(True)


def save_paper_figure(fig, path, *, dpi: int | None = None, metadata=None) -> None:
    """Save with the LPS vector/600-dpi policy and scaled publication padding."""
    style = apply_lps_style()
    prepare_text(fig)
    fig.savefig(path, dpi=dpi or style.dpi, bbox_inches="tight", pad_inches=style.pad,
                metadata=metadata)
