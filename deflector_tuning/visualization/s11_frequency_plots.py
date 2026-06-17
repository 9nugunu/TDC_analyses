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
MARKER_LABELS: dict[str, str] = {
    "f_2pi3": r"$f_{2\pi/3}$",
    "f_mean": r"$f_{mean}$",
    "f_pi2": r"$f_{\pi/2}$",
}
MARKER_COLORS: dict[str, str] = {
    "f_2pi3": "#2e7d32",
    "f_mean": "#ff6f00",
    "f_pi2": "#1565c0",
}
PORT_SIDE_STYLES: dict[str, dict[str, object]] = {
    "in": {"color": "#0d47a1", "linestyle": "-", "marker": "o"},
    "out": {"color": "#b71c1c", "linestyle": "--", "marker": "D"},
}
DEFAULT_TRACE_STYLE: dict[str, object] = {"color": "#1565c0", "linestyle": "-"}
MARKER_Y_OFFSETS: dict[str, int] = {"f_2pi3": 20, "f_mean": -34, "f_pi2": 50}
MAX_LEGEND_ENTRIES: int = 30
DUPLICATE_ID_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "data_kind",
    "data_layer",
    "tune_position",
    "port_side",
    "s_name",
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
        for family, sim_r_c, sim_w_c, group in progress_iter(
            groups,
            desc="Rendering S11 grid figures",
            total=len(groups),
        ):
            marker_group = _select_grid_point_rows(m_table, family=family, sim_r_c=sim_r_c, sim_w_c=sim_w_c)
            key = _grid_point_output_key(family, sim_r_c, sim_w_c)
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"s11_{_grid_point_file_stem(family, sim_r_c, sim_w_c)}.png",
                title=f"{_grid_point_title_prefix(family, sim_r_c, sim_w_c)}: S11 magnitude",
                config=config,
            )
        return paths

    if not _skip_overview_for_port_sides(s_table):
        paths["overview"] = _plot_one(
            s_table,
            m_table,
            folder / "s11_with_markers.png",
            title="S11 magnitude with marker points",
            config=config,
        )
    if split_by_position and _has_named_tune_positions(s_table, m_table):
        position_groups = list(s_table.groupby("tune_position", dropna=False, sort=True))
        for tune_position, group in progress_iter(
            position_groups,
            desc="Rendering S11 position figures",
            total=len(position_groups),
        ):
            marker_group = m_table[m_table["tune_position"] == tune_position]
            key = _tune_position_output_key(tune_position)
            paths[key] = _plot_one(
                group,
                marker_group,
                folder / f"s11_{key}.png",
                title=f"{_position_family(tune_position).title()} {_format_position(tune_position)}: S11 magnitude",
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
        label = _source_label(source_file, group)
        style = _source_style(group)
        ax.plot(
            group["freq_ghz"],
            group["s_db"],
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


def _tune_position_output_key(tune_position: object) -> str:
    return f"{_position_family(tune_position)}_{_format_position_key(tune_position)}"


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
        table = table.sort_values(
            ["position_family", "_r_sort", "_w_sort", "sim_r_c", "sim_w_c", "_position_sort", "source_file", "freq_ghz"],
            kind="mergesort",
        )
        for (family, sim_r_c, sim_w_c), group in table.groupby(["position_family", "sim_r_c", "sim_w_c"], sort=False, dropna=False):
            yield str(family), sim_r_c, sim_w_c, group
        return
    table = table.sort_values(
        ["_r_sort", "_w_sort", "sim_r_c", "sim_w_c", "source_file", "freq_ghz"],
        kind="mergesort",
    )
    for (sim_r_c, sim_w_c), group in table.groupby(["sim_r_c", "sim_w_c"], sort=False, dropna=False):
        yield "grid", sim_r_c, sim_w_c, group


def _select_grid_point_rows(table: pd.DataFrame, *, family: str, sim_r_c: object, sim_w_c: object) -> pd.DataFrame:
    if "sim_r_c" not in table or "sim_w_c" not in table:
        return table.iloc[0:0].copy()
    mask = (table["sim_r_c"] == sim_r_c) & (table["sim_w_c"] == sim_w_c)
    if family not in {"grid", "unknown"} and "tune_position" in table:
        family_values = table["tune_position"].map(lambda value: "unknown" if pd.isna(value) else _position_family(value))
        mask &= family_values == family
    return table.loc[mask].copy()


def _grid_point_output_key(family: str, sim_r_c: object, sim_w_c: object) -> str:
    prefix = "grid" if family == "unknown" else family
    return f"{prefix}_{_format_grid_point_key(sim_r_c, sim_w_c)}"


def _grid_point_file_stem(family: str, sim_r_c: object, sim_w_c: object) -> str:
    grid_key = _format_grid_point_key(sim_r_c, sim_w_c)
    if family in {"grid", "unknown"}:
        return grid_key
    return f"{family}_{grid_key}"


def _grid_point_title_prefix(family: str, sim_r_c: object, sim_w_c: object) -> str:
    grid_label = f"r_c={_format_grid_value(sim_r_c)}, w_c={_format_grid_value(sim_w_c)}"
    if family in {"grid", "unknown"}:
        return grid_label
    return f"{family} {grid_label}"


def _format_grid_point_key(sim_r_c: object, sim_w_c: object) -> str:
    return f"r_c_{_format_grid_value(sim_r_c).replace('.', 'p')}_w_c_{_format_grid_value(sim_w_c).replace('.', 'p')}"


def _format_grid_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:g}"
