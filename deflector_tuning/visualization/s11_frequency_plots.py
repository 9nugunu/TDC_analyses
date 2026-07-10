"""S11 frequency-domain plots with marker-frequency point overlays."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deflector_tuning.visualization.finite_checks import require_finite_plot_columns
from deflector_tuning.visualization.plot_config import (
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    save_figure,
)
from deflector_tuning.visualization.marker_styles import MARKER_COLORS, MARKER_LABELS
from deflector_tuning.progress import progress_iter

REQUIRED_SPARAMETER_COLUMNS: tuple[str, ...] = ("source_file", "freq_ghz", "s_db", "s_phase_deg")
REQUIRED_MARKER_COLUMNS: tuple[str, ...] = (
    "source_file",
    "marker_name",
    "freq_ghz",
    "s_db",
    "s_phase_deg",
)
MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")
PORT_SIDE_STYLES: dict[str, dict[str, object]] = {
    "in": {"color": "#0d47a1", "linestyle": "-", "marker": "o"},
    "out": {"color": "#b71c1c", "linestyle": "--", "marker": "D"},
}
DEFAULT_TRACE_STYLE: dict[str, object] = {"color": "#1565c0", "linestyle": "-"}
MARKER_Y_OFFSETS: dict[str, int] = {"f_2pi3": 20, "f_mean": -34, "f_pi2": 50}
MAX_LEGEND_ENTRIES: int = 30
MAX_TRACE_POINTS_PER_SOURCE: int = 5000
DUPLICATE_ID_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "data_kind",
    "data_layer",
    "tune_position",
    "port_side",
    "s_name",
    "sim_offset_cell_03",
    "sim_r_c",
    "sim_w_c",
    "marker_name",
    "marker_role",
    "marker_source",
    "target_freq_ghz",
    "freq_ghz",
)


def plot_s11_with_markers(
    sparameter_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_by_position: bool = True,
    config: PlotConfig | None = None,
) -> OrderedDict[str, Path]:
    """Write S11 frequency plots with marker point overlays.

    The style follows the existing KYHL S11 frequency views: a blue S11 trace,
    colored marker vertical guides/points, and compact marker annotations.
    """

    if sparameter_table.empty:
        raise ValueError("sparameter_table is empty")
    missing_s = [column for column in REQUIRED_SPARAMETER_COLUMNS if column not in sparameter_table]
    if missing_s:
        raise ValueError(f"sparameter_table is missing required columns: {missing_s}")
    missing_m = [column for column in REQUIRED_MARKER_COLUMNS if column not in marker_points]
    if missing_m:
        raise ValueError(f"marker_points is missing required columns: {missing_m}")

    config = config or PlotConfig()
    apply_plot_style(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    s_table = _collapse_unlabelled_duplicate_rows(_select_s11_rows(sparameter_table))
    m_table = _collapse_unlabelled_duplicate_rows(_select_s11_rows(marker_points))
    if s_table.empty:
        raise ValueError("sparameter_table has no S11 rows")
    require_finite_plot_columns(s_table, columns=("freq_ghz", "s_db"), context="S11 sparameter_table")
    require_finite_plot_columns(m_table, columns=("freq_ghz", "s_db", "s_phase_deg"), context="S11 marker_points")
    paths: OrderedDict[str, Path] = OrderedDict()
    if _has_grid_point_groups(s_table, m_table):
        groups = list(_iter_grid_point_groups(s_table))
        for family, depth_label, sim_r_c, sim_w_c, group in progress_iter(
            groups,
            desc="Rendering S11 grid figures",
            total=len(groups),
        ):
            marker_group = _select_grid_point_rows(
                m_table,
                family=family,
                depth_label=depth_label,
                sim_r_c=sim_r_c,
                sim_w_c=sim_w_c,
            )
            key = _grid_point_output_key(family, depth_label, sim_r_c, sim_w_c)
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"{_grid_point_file_stem(family, depth_label, sim_r_c, sim_w_c)}.png",
                title=f"{_grid_point_title_prefix(family, depth_label, sim_r_c, sim_w_c)}: S11 magnitude",
                config=config,
            )
        return paths

    if _has_sim_sweep_groups(s_table, m_table):
        groups = list(_iter_sim_sweep_groups(s_table))
        for key, group in progress_iter(
            groups,
            desc="Rendering S11 sweep figures",
            total=len(groups),
        ):
            marker_group = _select_sim_sweep_rows(m_table, group)
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"{key}.png",
                title=f"{key.replace('_', ' ')}: S11 magnitude",
                config=config,
            )
        return paths

    if not _skip_overview_for_port_sides(s_table):
        paths["overview"] = _plot_one(
            s_table,
            m_table,
            folder / "with_markers.png",
            title="S11 magnitude with marker points",
            config=config,
        )
    if split_by_position and _has_named_tune_positions(s_table, m_table):
        positioned_s_table = s_table[s_table["tune_position"].notna()]
        position_groups = list(positioned_s_table.groupby("tune_position", sort=True))
        for tune_position, group in progress_iter(
            position_groups,
            desc="Rendering S11 position figures",
            total=len(position_groups),
        ):
            marker_group = m_table[m_table["tune_position"] == tune_position]
            key = _tune_position_output_key(tune_position, group=group)
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"{key}.png",
                title=f"{_tune_position_title_label(tune_position, group=group)}: S11 magnitude",
                config=config,
            )
    return paths


def _plot_one(
    s_table: pd.DataFrame,
    marker_points: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    config: PlotConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for source_file, group in s_table.groupby("source_file", sort=False):
        group = group.sort_values("freq_ghz")
        plot_group = _thin_trace_for_plot(group)
        label = _source_label(source_file, group)
        style = _source_style(group)
        ax.plot(
            plot_group["freq_ghz"],
            plot_group["s_db"],
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=2.4,
            label=label,
        )

    if not marker_points.empty:
        y_min, y_max = _axis_marker_bounds(s_table["s_db"])
        for _, point in marker_points.sort_values(["source_file", "freq_ghz", "marker_name"]).iterrows():
            marker_name = point["marker_name"]
            color = MARKER_COLORS.get(marker_name, "#333333")
            point_style = _marker_point_style(point)
            ax.axvline(point["freq_ghz"], color=color, linestyle="--", linewidth=1.0, alpha=0.45)
            ax.scatter(
                [point["freq_ghz"]],
                [point["s_db"]],
                s=72,
                color=color,
                marker=str(point_style["marker"]),
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
            offset_x, offset_y, ha = _annotation_offset(point)
            ax.annotate(
                _marker_annotation(point),
                xy=(point["freq_ghz"], point["s_db"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                color=color,
                fontsize=config.annotation_size,
                fontweight="bold",
                ha=ha,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.92},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.9},
            )
        ax.set_ylim(y_min, y_max)

    apply_axis_text_style(
        ax,
        xlabel="Freq. (GHz)",
        ylabel=r"$S_{11}$ (dB)",
        title=title,
        config=config,
    )
    ax.grid(True, which="major", color="0.78", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="0.90", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    _add_legend_if_readable(ax, config=config)
    fig.tight_layout()
    path = save_figure(fig, output_path, config)
    plt.close(fig)
    return path


def _thin_trace_for_plot(group: pd.DataFrame, *, max_points: int = MAX_TRACE_POINTS_PER_SOURCE) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    indices = np.linspace(0, len(group) - 1, num=max_points, dtype=int)
    return group.iloc[np.unique(indices)]


def _add_legend_if_readable(ax, *, config: PlotConfig) -> None:
    handles, labels = ax.get_legend_handles_labels()
    visible = [(handle, label) for handle, label in zip(handles, labels, strict=True) if not str(label).startswith("_")]
    if not visible or len(visible) > MAX_LEGEND_ENTRIES:
        return
    legend_handles, legend_labels = zip(*visible, strict=True)
    legend = ax.legend(legend_handles, legend_labels, frameon=True, loc="best", fontsize=config.legend_size)
    apply_legend_text_style(legend, config)


def _axis_marker_bounds(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * 0.25, 3.0)
    return low - padding, high + padding


def _has_named_tune_positions(s_table: pd.DataFrame, m_table: pd.DataFrame) -> bool:
    if "tune_position" not in s_table or "tune_position" not in m_table:
        return False
    return s_table["tune_position"].notna().any() and m_table["tune_position"].notna().any()


def _select_s11_rows(table: pd.DataFrame) -> pd.DataFrame:
    if "s_name" not in table:
        return table.copy()
    return table[table["s_name"].astype(str).str.upper() == "S11"].copy()


def _collapse_unlabelled_duplicate_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "source_file" not in table or "tune_position" not in table:
        return table.copy()
    if "port_side" in table and table["port_side"].notna().any():
        return table.copy()

    group_columns = [column for column in DUPLICATE_ID_COLUMNS if column in table]
    if not group_columns:
        return table.copy()

    rows: list[dict[str, object]] = []
    for _, group in table.groupby(group_columns, dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        if len(group) > 1:
            row["source_file"] = _combined_source_label(group)
            if "s_db" in group:
                row["s_db"] = float(pd.to_numeric(group["s_db"], errors="coerce").mean())
            if "s_phase_deg" in group:
                row["s_phase_deg"] = _circular_mean_deg(group["s_phase_deg"])
            if "freq_error_ghz" in group:
                row["freq_error_ghz"] = float(pd.to_numeric(group["freq_error_ghz"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=table.columns)


def _combined_source_label(group: pd.DataFrame) -> str:
    positions = group["tune_position"].dropna().unique()
    if len(positions) == 1:
        return _format_position(positions[0])
    return "combined"


def _circular_mean_deg(values: pd.Series) -> float:
    angles = pd.to_numeric(values, errors="coerce").dropna()
    if angles.empty:
        return float("nan")
    vectors = np.exp(1j * np.deg2rad(angles.to_numpy(dtype=float)))
    return float(np.angle(vectors.mean(), deg=True))


def _marker_annotation(point: pd.Series) -> str:
    label = MARKER_LABELS.get(str(point["marker_name"]), str(point["marker_name"]))
    return f"{label}\n{float(point['s_phase_deg']):+.1f}°"


def _source_label(source_file: object, group: pd.DataFrame) -> str:
    if "port_side" in group:
        port_sides = group["port_side"].dropna().astype(str).str.lower().unique()
        if len(port_sides) == 1:
            return port_sides[0]
    return _source_file_label(source_file)


def _source_file_label(source_file: object) -> str:
    text = str(source_file)
    run_label = _run_label_from_source_file(text)
    if run_label is not None:
        return run_label
    suffix = Path(text).suffix.lower()
    if suffix in {".csv", ".s1p", ".s2p", ".s3p", ".s4p"}:
        return Path(text).stem.replace("_processed", "")
    return text.replace("_processed", "")


def _run_label_from_source_file(source_file: str) -> str | None:
    stem = Path(source_file).stem
    match = re.search(r"(?:^run|[_\-\s](?:run)?)(\d+)$", stem, flags=re.IGNORECASE)
    if match is None:
        return None
    digits = match.group(1)
    number = int(digits)
    width = max(2, len(digits))
    return f"RUN {number:0{width}d}"


def _source_style(group: pd.DataFrame) -> dict[str, object]:
    if "port_side" not in group:
        return DEFAULT_TRACE_STYLE
    port_sides = group["port_side"].dropna().astype(str).str.lower().unique()
    if len(port_sides) != 1:
        return DEFAULT_TRACE_STYLE
    return PORT_SIDE_STYLES.get(port_sides[0], DEFAULT_TRACE_STYLE)


def _marker_point_style(point: pd.Series) -> dict[str, object]:
    port_side = str(point.get("port_side", "")).lower()
    return PORT_SIDE_STYLES.get(port_side, {"marker": "o"})


def _annotation_offset(point: pd.Series) -> tuple[int, int, str]:
    marker_name = str(point.get("marker_name", ""))
    port_side = str(point.get("port_side", "")).lower()
    base_y = MARKER_Y_OFFSETS.get(marker_name, 20)
    if port_side == "out":
        return -52, base_y - 12, "right"
    if port_side == "in":
        return 10, base_y, "left"
    return 10, base_y, "left"


def _format_position(position: object) -> str:
    value = float(position)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def _format_position_key(position: object) -> str:
    return _format_position(position).replace(".", "p")


def _tune_position_output_key(tune_position: object, *, group: pd.DataFrame | None = None) -> str:
    depth_label = _num_depth_output_label(group)
    if depth_label is not None:
        return depth_label
    return f"{_position_family(tune_position)}_{_format_position_key(tune_position)}"


def _tune_position_title_label(tune_position: object, *, group: pd.DataFrame | None = None) -> str:
    depth_label = _num_depth_output_label(group)
    if depth_label is not None:
        return f"Depth {depth_label.removeprefix('depth_')} (tune {_format_position(tune_position)})"
    return f"{_position_family(tune_position).title()} {_format_position(tune_position)}"


def _num_depth_output_label(group: pd.DataFrame | None) -> str | None:
    if group is None or "sim_NumDepth" not in group:
        return None
    values = pd.to_numeric(group["sim_NumDepth"], errors="coerce").dropna().unique()
    if len(values) != 1:
        return None
    return _num_depth_output_label_from_value(values[0])


def _num_depth_output_label_from_value(value: object) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    if abs(numeric - round(numeric)) < 1e-9:
        return f"depth_{int(round(numeric))}p0"
    return f"depth_{_format_position_key(numeric)}"


def _position_family(tune_position: object) -> str:
    value = float(tune_position)
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{str(fractional).replace('.', 'p')}"


def _skip_overview_for_port_sides(s_table: pd.DataFrame) -> bool:
    if "port_side" not in s_table:
        return False
    return s_table["port_side"].dropna().nunique() > 0


def _has_grid_point_groups(s_table: pd.DataFrame, m_table: pd.DataFrame) -> bool:
    if "sim_r_c" not in s_table or "sim_w_c" not in s_table:
        return False
    grid_points = s_table[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
    return len(grid_points) >= 1


def _iter_grid_point_groups(s_table: pd.DataFrame):
    table = s_table.copy()
    table["_r_sort"] = pd.to_numeric(table["sim_r_c"], errors="coerce")
    table["_w_sort"] = pd.to_numeric(table["sim_w_c"], errors="coerce")
    if "tune_position" in table and table["tune_position"].dropna().nunique() > 0:
        table["position_family"] = table["tune_position"].map(lambda value: "unknown" if pd.isna(value) else _position_family(value))
        table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
        group_columns = ["position_family", *_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
        table = table.sort_values(
            [*group_columns, "_r_sort", "_w_sort", "_position_sort", "source_file", "freq_ghz"],
            kind="mergesort",
        )
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            family = str(values[0])
            depth_label = _num_depth_output_label(group) if "sim_NumDepth" in group_columns else None
            sim_r_c = values[-2]
            sim_w_c = values[-1]
            yield family, depth_label, sim_r_c, sim_w_c, group
        return
    group_columns = [*_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
    table = table.sort_values(
        [*group_columns, "_r_sort", "_w_sort", "source_file", "freq_ghz"],
        kind="mergesort",
    )
    for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        depth_label = _num_depth_output_label(group) if "sim_NumDepth" in group_columns else None
        sim_r_c = values[-2]
        sim_w_c = values[-1]
        yield "grid", depth_label, sim_r_c, sim_w_c, group


def _select_grid_point_rows(
    table: pd.DataFrame,
    *,
    family: str,
    depth_label: str | None,
    sim_r_c: object,
    sim_w_c: object,
) -> pd.DataFrame:
    if "sim_r_c" not in table or "sim_w_c" not in table:
        return table.iloc[0:0].copy()
    mask = (table["sim_r_c"] == sim_r_c) & (table["sim_w_c"] == sim_w_c)
    if family not in {"grid", "unknown"} and "tune_position" in table:
        family_values = table["tune_position"].map(lambda value: "unknown" if pd.isna(value) else _position_family(value))
        mask &= family_values == family
    if depth_label is not None and "sim_NumDepth" in table:
        mask &= table["sim_NumDepth"].map(_num_depth_output_label_from_value) == depth_label
    return table.loc[mask].copy()


def _grid_point_output_key(family: str, depth_label: str | None, sim_r_c: object, sim_w_c: object) -> str:
    depth_prefix = f"_{depth_label}" if depth_label is not None else ""
    prefix = "grid" if family == "unknown" else family
    return f"{prefix}{depth_prefix}_{_format_grid_point_key(sim_r_c, sim_w_c)}"


def _grid_point_file_stem(family: str, depth_label: str | None, sim_r_c: object, sim_w_c: object) -> str:
    grid_key = _format_grid_point_key(sim_r_c, sim_w_c)
    depth_prefix = f"{depth_label}_" if depth_label is not None else ""
    if family in {"grid", "unknown"}:
        return f"{depth_prefix}{grid_key}"
    return f"{family}_{depth_prefix}{grid_key}"


def _grid_point_title_prefix(family: str, depth_label: str | None, sim_r_c: object, sim_w_c: object) -> str:
    grid_label = f"r_c={_format_grid_value(sim_r_c)}, w_c={_format_grid_value(sim_w_c)}"
    if depth_label is not None:
        grid_label = f"{depth_label.replace('_', ' ')} {grid_label}"
    if family in {"grid", "unknown"}:
        return grid_label
    return f"{family} {grid_label}"


def _grid_point_depth_group_columns(table: pd.DataFrame) -> list[str]:
    if "sim_NumDepth" in table and pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().nunique() > 1:
        return ["sim_NumDepth"]
    return []


def _has_sim_sweep_groups(s_table: pd.DataFrame, m_table: pd.DataFrame) -> bool:
    return bool(_varying_sim_sweep_columns(s_table)) and bool(_varying_sim_sweep_columns(m_table))


def _iter_sim_sweep_groups(s_table: pd.DataFrame):
    table = s_table.copy()
    group_columns = _sim_sweep_group_columns(table)
    sort_columns = [f"_{column}_sort" for column in group_columns]
    for column, sort_column in zip(group_columns, sort_columns, strict=True):
        table[sort_column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values([*sort_columns, *group_columns, "source_file", "freq_ghz"], kind="mergesort")
    for values, group in table.groupby(group_columns, sort=False, dropna=False):
        if len(group_columns) == 1 and not isinstance(values, tuple):
            values = (values,)
        yield _format_sim_sweep_key(group_columns, values, group), group


def _select_sim_sweep_rows(table: pd.DataFrame, s_group: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    mask = pd.Series(True, index=table.index)
    for column in _sim_sweep_group_columns(s_group):
        if column not in table:
            return table.iloc[0:0].copy()
        values = s_group[column].dropna().unique()
        if len(values) != 1:
            continue
        mask &= table[column] == values[0]
    return table.loc[mask].copy()


def _sim_sweep_group_columns(table: pd.DataFrame) -> list[str]:
    group_columns = []
    if "sim_NumDepth" in table and pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().nunique() > 1:
        group_columns.append("sim_NumDepth")
    group_columns.extend(column for column in _varying_sim_sweep_columns(table) if column not in group_columns)
    return group_columns


def _varying_sim_sweep_columns(table: pd.DataFrame) -> list[str]:
    columns = []
    for column in table.columns:
        if not column.startswith("sim_") or column in {"sim_r_c", "sim_w_c"}:
            continue
        metadata_name = column.removeprefix("sim_")
        if metadata_name.lower().startswith("num"):
            continue
        if table[column].dropna().nunique() > 1:
            columns.append(column)
    return columns


def _format_sim_sweep_key(group_columns: list[str], values: tuple[object, ...], group: pd.DataFrame) -> str:
    parts = []
    family = _sim_sweep_family(group)
    if family:
        parts.append(family)
    for column, value in zip(group_columns, values, strict=True):
        if column == "sim_NumDepth":
            label = _num_depth_output_label_from_value(value)
            if label is not None:
                parts.append(label)
        else:
            parts.append(f"{_sim_sweep_column_label(column)}_{_format_grid_value(value).replace('.', 'p').replace('-', 'm')}")
    return "_".join(parts)


def _sim_sweep_family(group: pd.DataFrame) -> str:
    if "tune_position" not in group:
        return ""
    tune_positions = group["tune_position"].dropna().unique()
    if len(tune_positions) != 1:
        return ""
    try:
        return _position_family(tune_positions[0])
    except (TypeError, ValueError):
        return ""


def _sim_sweep_column_label(column: str) -> str:
    label = column.removeprefix("sim_")
    if label.startswith("Depth") and len(label) > len("Depth"):
        label = label[len("Depth") :]
    return _camel_to_snake(label)


def _camel_to_snake(label: str) -> str:
    output = []
    for index, char in enumerate(label):
        if char.isupper() and index > 0 and not label[index - 1].isupper():
            output.append("_")
        output.append(char.lower())
    return "".join(output)


def _format_grid_point_key(sim_r_c: object, sim_w_c: object) -> str:
    return f"r_c_{_format_grid_value(sim_r_c).replace('.', 'p')}_w_c_{_format_grid_value(sim_w_c).replace('.', 'p')}"


def _format_grid_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:g}"
