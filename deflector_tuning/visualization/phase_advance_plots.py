"""Line plots for local phase-advance transition profiles."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.progress import progress_iter
from deflector_tuning.visualization.simulation_grouping import (
    format_simulation_parameter_token as _format_simulation_parameter_token,
    format_simulation_parameter_value as _format_simulation_parameter_value,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "marker_name",
    "from_tune_position",
    "to_tune_position",
    "phase_advance_0to360_deg",
    "phase_error_from_240_deg",
)
MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
SERIES_ID_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "data_kind",
    "data_layer",
    "marker_role",
    "port_side",
    "s_name",
)
SERIES_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")
FACET_COLUMNS: tuple[str, ...] = ("sim_r_c", "sim_w_c")


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
    table = _prepare_table(_select_s11_rows(phase_advance), transition_scope=transition_scope)

    paths: OrderedDict[str, Path] = OrderedDict()
    if split_by_family and _has_named_position_families(table):
        families = _position_family_order(table)
        for family in progress_iter(families, desc="Rendering phase advance figures", total=len(families)):
            family_table = table[table["position_family"] == family].copy()
            if family_table.empty:
                continue
            for facet_stem, facet_title, facet_table in _iter_plot_facets(family_table):
                key = _phase_advance_key(family, facet_stem)
                paths[key] = _plot_metric(
                    facet_table,
                    folder / f"{key}.png",
                    value_column="phase_advance_0to360_deg",
                    ylabel="Phase advance [deg]",
                    title=f"{family.title()} phase advance by transition{facet_title}",
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
    series_columns = _series_columns(table)

    for marker in marker_order:
        for family in family_order:
            group = table[(table["marker_name"] == marker) & (table["position_family"] == family)].copy()
            if group.empty:
                continue
            series_groups = list(group.groupby(series_columns, dropna=False, sort=False)) if series_columns else [((), group)]
            for series_index, (series_values, series_group) in enumerate(series_groups):
                series_group = series_group.sort_values(["_from_sort", "_to_sort"], kind="mergesort")
                x_values = [x_by_label[label] for label in series_group["transition_label"]]
                ax.plot(
                    x_values,
                    series_group[value_column],
                    marker="o",
                    linewidth=1.6,
                    markersize=4.5,
                    linestyle=SERIES_LINESTYLES[series_index % len(SERIES_LINESTYLES)],
                    label=_series_label(
                        marker,
                        family,
                        include_family=include_family_in_label,
                        columns=series_columns,
                        values=series_values,
                    ),
                    color=MARKER_COLORS.get(marker),
                )

    ax.axhline(reference_value, color="0.25", linestyle="--", linewidth=1.0, label=reference_label)
    apply_axis_text_style(ax, xlabel="Transition", ylabel=ylabel, title=title, config=config)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    legend = ax.legend(frameon=False, loc="best", fontsize=config.legend_size)
    apply_legend_text_style(legend, config)
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


def _series_columns(table: pd.DataFrame) -> list[str]:
    """Return metadata columns that distinguish independent line series."""

    columns: list[str] = []
    for column in SERIES_ID_COLUMNS:
        if column not in table:
            continue
        values = table[column].fillna("<NA>").astype(str)
        if values.nunique(dropna=False) > 1:
            columns.append(column)
    return columns


def _series_label(
    marker: object,
    family: object,
    *,
    include_family: bool,
    columns: list[str],
    values: object,
) -> str:
    marker_label = MARKER_LABELS.get(str(marker), str(marker))
    parts = [marker_label]
    if include_family:
        parts.append(str(family))

    if columns:
        if len(columns) == 1:
            series_values = values if isinstance(values, tuple) else (values,)
        else:
            series_values = tuple(values)
        suffix = [
            str(value)
            for column, value in zip(columns, series_values, strict=True)
            if not pd.isna(value) and str(value) not in {"", "<NA>", "nan", "None"}
        ]
        parts.extend(suffix)
    return " ".join(parts)


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


def _select_s11_rows(table: pd.DataFrame) -> pd.DataFrame:
    if "s_name" not in table:
        return table.copy()
    return table[table["s_name"].astype(str).str.upper() == "S11"].copy()


def _iter_plot_facets(table: pd.DataFrame):
    facet_columns = [column for column in FACET_COLUMNS if _needs_faceting(table, column)]
    if not facet_columns:
        yield "", "", table
        return

    sort_columns = [*facet_columns, "_from_sort", "_to_sort", "marker_name"]
    sorted_table = table.sort_values(sort_columns, kind="mergesort")
    for facet_values, group in sorted_table.groupby(facet_columns, dropna=False, sort=False):
        if not isinstance(facet_values, tuple):
            facet_values = (facet_values,)
        parts = [
            (column, value)
            for column, value in zip(facet_columns, facet_values, strict=True)
            if not pd.isna(value)
        ]
        facet_stem = "_".join(_facet_suffix_part(column, value) for column, value in parts)
        title = ""
        if parts:
            title = " (" + ", ".join(
                f"{_facet_label(column)}={_facet_display_value(column, value)}" for column, value in parts
            ) + ")"
        yield facet_stem, title, group.copy()


def _phase_advance_key(family: str, facet_stem: str) -> str:
    if not facet_stem:
        return family
    if "r_c_" in facet_stem or "w_c_" in facet_stem:
        return f"{family}_{facet_stem}"
    return f"{family}_{facet_stem}"


def _needs_faceting(table: pd.DataFrame, column: str) -> bool:
    if column not in table:
        return False
    values = table[column].dropna().astype(str)
    return values.nunique() > 1


def _facet_label(column: str) -> str:
    return {"s_name": "S", "port_side": "port", "sim_r_c": "r_c", "sim_w_c": "w_c"}.get(column, column)


def _facet_suffix_part(column: str, value: object) -> str:
    prefix = {"sim_r_c": "r_c", "sim_w_c": "w_c"}.get(column)
    label = (
        _format_simulation_parameter_token(column, value)
        if prefix is not None
        else _safe_label(value)
    )
    if prefix is None:
        return label
    return f"{prefix}_{label}"


def _facet_display_value(column: str, value: object) -> object:
    if column in {"sim_r_c", "sim_w_c"}:
        return _format_simulation_parameter_value(column, value)
    return value


def _safe_label(value: object) -> str:
    sanitized = str(value).strip().lower().replace(" ", "_").replace(".", "p").replace("-", "m")
    sanitized = "".join(character if character.isalnum() or character == "_" else "_" for character in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "unknown"
