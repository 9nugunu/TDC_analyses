"""Line plots for local phase-advance transition profiles."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.plot_config import PlotConfig, apply_axis_text_style, apply_plot_style, save_figure

REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "from_tune_position",
    "to_tune_position",
    "phase_advance_0to360_deg",
    "phase_error_from_240_deg",
)
MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
MARKER_LABELS: dict[str, str] = {
    "f_2pi3": r"$f_{2\pi/3}$",
    "f_mean": r"$f_{mean}$",
    "f_pi2": r"$f_{\pi/2}$",
}
MARKER_COLORS: dict[str, str] = {
    "f_2pi3": "#d62728",
    "f_mean": "#1f77b4",
    "f_pi2": "#2ca02c",
}


def plot_phase_advance(
    phase_advance: pd.DataFrame,
    output_dir: str | Path,
    *,
    transition_scope: str = "all",
    split_by_family: bool = False,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write phase-advance line plots."""

    if phase_advance.empty:
        raise ValueError("phase_advance is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in phase_advance]
    if missing:
        raise ValueError(f"phase_advance is missing required columns: {missing}")
    if transition_scope not in {"all", "internal"}:
        raise ValueError("transition_scope must be 'all' or 'internal'")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table = _prepare_table(phase_advance, transition_scope=transition_scope)

    paths: OrderedDict[str, Path] = OrderedDict()
    if split_by_family and _has_named_position_families(table):
        for family in _position_family_order(table):
            family_table = table[table["position_family"] == family].copy()
            if family_table.empty:
                continue
            paths[f"phase_advance_{family}"] = _plot_metric(
                family_table,
                folder / f"phase_advance_{family}.png",
                value_column="phase_advance_0to360_deg",
                ylabel="Phase advance [deg]",
                title=f"{family.title()} phase advance by transition",
                reference_value=240.0,
                reference_label="ideal 240°",
                config=config,
                include_family_in_label=False,
            )
    return paths


def _prepare_table(phase_advance: pd.DataFrame, *, transition_scope: str) -> pd.DataFrame:
    table = phase_advance.copy()
    if "position_family" not in table:
        table["position_family"] = table["from_tune_position"].map(_position_family)
    table["_from_sort"] = pd.to_numeric(table["from_tune_position"], errors="coerce")
    table["_to_sort"] = pd.to_numeric(table["to_tune_position"], errors="coerce")
    if transition_scope == "internal":
        masks = []
        for _, family_group in table.groupby("position_family", dropna=False, sort=False):
            transition_order = family_group[["_from_sort", "_to_sort"]].drop_duplicates().sort_values(
                ["_from_sort", "_to_sort"]
            )
            if len(transition_order) <= 2:
                masks.append(pd.Series(True, index=family_group.index))
                continue
            first = tuple(transition_order.iloc[0])
            last = tuple(transition_order.iloc[-1])
            masks.append(
                ~(
                    ((family_group["_from_sort"] == first[0]) & (family_group["_to_sort"] == first[1]))
                    | ((family_group["_from_sort"] == last[0]) & (family_group["_to_sort"] == last[1]))
                )
            )
        table = table.loc[pd.concat(masks).sort_index()].copy()
    table["transition_label"] = table.apply(
        lambda row: f"{_format_position(row['from_tune_position'])}→{_format_position(row['to_tune_position'])}",
        axis=1,
    )
    return table.sort_values(["position_family", "_from_sort", "_to_sort", "marker_name"], kind="mergesort")


def _plot_metric(
    table: pd.DataFrame,
    output_path: Path,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    reference_value: float,
    reference_label: str,
    config: PlotConfig,
    include_family_in_label: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = list(dict.fromkeys(table["transition_label"].tolist()))
    x_by_label = {label: index for index, label in enumerate(labels)}
    marker_order = [marker for marker in MARKER_ORDER if marker in set(table["marker_name"])]
    marker_order.extend(marker for marker in table["marker_name"].dropna().unique() if marker not in marker_order)
    family_order = _position_family_order(table)

    for marker in marker_order:
        for family in family_order:
            group = table[(table["marker_name"] == marker) & (table["position_family"] == family)].copy()
            if group.empty:
                continue
            x_values = [x_by_label[label] for label in group["transition_label"]]
            ax.plot(
                x_values,
                group[value_column],
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                linestyle="-" if family == "cell" else "--",
                label=(
                    f"{MARKER_LABELS.get(marker, marker)} {family}"
                    if include_family_in_label
                    else MARKER_LABELS.get(marker, marker)
                ),
                color=MARKER_COLORS.get(marker),
            )

    ax.axhline(reference_value, color="0.25", linestyle="--", linewidth=1.0, label=reference_label)
    apply_axis_text_style(ax, xlabel="Transition", ylabel=ylabel, title=title, config=config)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="best", fontsize=config.legend_size)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _format_position(position: object) -> str:
    try:
        value = float(position)
    except (TypeError, ValueError):
        return str(position)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _position_family(tune_position: object) -> str:
    if pd.isna(tune_position):
        return "unknown"
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{fractional:g}"


def _position_family_order(table: pd.DataFrame) -> list[str]:
    family_order = [family for family in ("cell", "iris") if family in set(table["position_family"])]
    family_order.extend(family for family in table["position_family"].dropna().unique() if family not in family_order)
    return family_order


def _has_named_position_families(table: pd.DataFrame) -> bool:
    if "position_family" not in table:
        return False
    families = set(table["position_family"].dropna().astype(str))
    return bool(families.difference({"unknown", "nan", "offset_nan"}))
