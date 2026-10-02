"""Select polar position groups and format their output labels.

This layer works with tables only; it does not create or save figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deflector_tuning.visualization.simulation_grouping import (
    format_grid_value as _shared_format_grid_value,
    format_simulation_parameter_value as _shared_format_simulation_parameter_value,
    grid_point_depth_group_columns as _shared_grid_point_depth_group_columns,
    varying_sim_sweep_columns as _shared_varying_sim_sweep_columns,
)


MARKER_ORDER: tuple[str, ...] = ("f_2pi3", "f_mean", "f_pi2")


KYHL_PHASE_PAIR_OVERLAY_STEPS: tuple[tuple[str, float, float], ...] = (
    ("cell", 0.5, 1.5),
    ("iris", 1.0, 2.0),
)


NO_PORT_EXTENSION_SOURCE_PATTERN = r"(?:^|[_-])noportE(?:[._-]|$)"


def _iter_position_groups(marker_points: pd.DataFrame):
    table = marker_points.copy()
    grouping_mode = _grouping_mode(table)
    if grouping_mode == "tune_position":
        table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
        group_columns = _position_group_columns(table, base_columns=("tune_position",))
        table = table.sort_values(["_position_sort", *group_columns, "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            yield _format_group_label(group_key, group_columns), group
        return
    if grouping_mode == "grid_point":
        table["_r_sort"] = pd.to_numeric(table["sim_r_c"], errors="coerce")
        table["_w_sort"] = pd.to_numeric(table["sim_w_c"], errors="coerce")
        if "tune_position" in table and table["tune_position"].dropna().nunique() > 0:
            table["position_family"] = table["tune_position"].map(lambda value: "unknown" if pd.isna(value) else _position_family(value))
            table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
            group_columns = ["position_family", *_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
            table = table.sort_values(
                [*group_columns, "_r_sort", "_w_sort", "_position_sort", "source_file", "marker_name"],
                kind="mergesort",
            )
            for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
                values = group_key if isinstance(group_key, tuple) else (group_key,)
                depth_label = _num_depth_label(group) if "sim_NumDepth" in group_columns else None
                yield _format_grid_family_label(str(values[0]), values[-2], values[-1], depth_label=depth_label), group
            return
        group_columns = [*_grid_point_depth_group_columns(table), "sim_r_c", "sim_w_c"]
        table = table.sort_values([*group_columns, "_r_sort", "_w_sort", "source_file", "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            depth_label = _num_depth_label(group) if "sim_NumDepth" in group_columns else None
            yield _format_grid_label(values[-2], values[-1], depth_label=depth_label), group
        return
    if grouping_mode == "sim_sweep":
        sweep_columns = _varying_sim_sweep_columns(table)
        group_columns = _sim_sweep_group_columns(table, sweep_columns)
        sort_columns = [f"_{column}_sort" for column in group_columns]
        for column, sort_column in zip(group_columns, sort_columns, strict=True):
            table[sort_column] = pd.to_numeric(table[column], errors="coerce")
        table = table.sort_values([*sort_columns, *group_columns, "source_file", "marker_name"], kind="mergesort")
        for values, group in table.groupby(group_columns, sort=False, dropna=False):
            if len(group_columns) == 1 and not isinstance(values, tuple):
                values = (values,)
            yield _format_sim_sweep_label(group_columns, values), group
        return
    if grouping_mode == "source_file":
        group_columns = _position_group_columns(table, base_columns=("source_file",))
        table = table.sort_values([*group_columns, "marker_name"], kind="mergesort")
        for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
            yield _format_group_label(group_key, group_columns), group
        return
    yield "all", table.sort_values(["source_file", "marker_name"], kind="mergesort")


def _grouping_mode(marker_points: pd.DataFrame) -> str:
    if _has_multiple_grid_points(marker_points):
        return "grid_point"
    if _has_multiple_sim_sweep_points(marker_points):
        return "sim_sweep"
    if _has_multiple_tune_positions(marker_points):
        return "tune_position"
    if _has_multiple_source_files(marker_points):
        return "source_file"
    return "all"


def _has_multiple_tune_positions(marker_points: pd.DataFrame) -> bool:
    if "tune_position" not in marker_points:
        return False
    return marker_points["tune_position"].dropna().nunique() > 1


def _has_multiple_grid_points(marker_points: pd.DataFrame) -> bool:
    if "sim_r_c" not in marker_points or "sim_w_c" not in marker_points:
        return False
    grid_points = marker_points[["sim_r_c", "sim_w_c"]].dropna().drop_duplicates()
    return len(grid_points) > 1


def _has_multiple_sim_sweep_points(marker_points: pd.DataFrame) -> bool:
    sweep_columns = _varying_sim_sweep_columns(marker_points)
    if not sweep_columns:
        return False
    sweep_points = marker_points[sweep_columns].dropna(how="all").drop_duplicates()
    return len(sweep_points) > 1


def _has_multiple_source_files(marker_points: pd.DataFrame) -> bool:
    if "source_file" not in marker_points:
        return False
    return marker_points["source_file"].dropna().nunique() > 1


def _is_no_port_extension_position(position_table: pd.DataFrame) -> bool:
    if "source_file" not in position_table:
        return False
    source_files = position_table["source_file"].dropna().astype(str)
    return bool(
        not source_files.empty
        and source_files.str.contains(
            NO_PORT_EXTENSION_SOURCE_PATTERN,
            case=False,
            regex=True,
        ).all()
    )


def _position_group_columns(table: pd.DataFrame, *, base_columns: tuple[str, ...]) -> list[str]:
    group_columns = list(base_columns)
    if not _has_duplicate_markers(table, group_columns):
        return group_columns
    for column in ("source_file", "s_name", "port_side"):
        if column in table and table[column].dropna().nunique() > 1 and column not in group_columns:
            group_columns.append(column)
        if not _has_duplicate_markers(table, group_columns):
            break
    return group_columns


def _has_duplicate_markers(table: pd.DataFrame, group_columns: list[str]) -> bool:
    required_columns = [*group_columns, "marker_name"]
    if any(column not in table for column in required_columns):
        return False
    return bool(table.duplicated(required_columns, keep=False).any())


def _iter_family_overlay_groups(marker_points: pd.DataFrame):
    table = marker_points.copy()
    table = table.dropna(subset=["tune_position"]).copy()
    if table.empty:
        return
    table["position_family"] = table["tune_position"].map(_position_family)
    table["_position_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table.sort_values(["position_family", "_position_sort", "marker_name"], kind="mergesort")
    for family, group in table.groupby("position_family", sort=False, dropna=False):
        if family not in {"cell", "iris"} or group["tune_position"].dropna().nunique() < 1:
            continue
        yield str(family), group.copy()


def _kyhl_phase_pair_overlay_table(marker_points: pd.DataFrame) -> pd.DataFrame:
    table = marker_points.copy()
    table = table.dropna(subset=["tune_position"]).copy()
    if table.empty:
        return pd.DataFrame()
    table["_tune_sort"] = pd.to_numeric(table["tune_position"], errors="coerce")
    table = table[table["_tune_sort"].notna() & table["marker_name"].isin(MARKER_ORDER)].copy()
    if table.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    marker_order = {marker: index for index, marker in enumerate(MARKER_ORDER)}
    for pair_order, (family, start, end) in enumerate(KYHL_PHASE_PAIR_OVERLAY_STEPS):
        is_start = np.isclose(table["_tune_sort"], start, atol=1e-9, equal_nan=False)
        is_end = np.isclose(table["_tune_sort"], end, atol=1e-9, equal_nan=False)
        pair_rows = table[is_start | is_end].copy()
        if pair_rows.empty or not bool(is_start.any() and is_end.any()):
            continue
        complete_markers = [
            marker
            for marker, group in pair_rows.groupby("marker_name", sort=False)
            if {"start", "end"}.issubset(
                set(np.where(np.isclose(group["_tune_sort"], start, atol=1e-9), "start", "end"))
            )
        ]
        if not complete_markers:
            continue
        pair_rows = pair_rows[pair_rows["marker_name"].isin(complete_markers)].copy()
        pair_rows["_pair_order"] = pair_order
        pair_rows["_pair_family"] = family
        pair_rows["_pair_start"] = start
        pair_rows["_pair_end"] = end
        pair_rows["_pair_label"] = f"{family} {_format_position(start)}->{_format_position(end)}"
        pair_rows["_pair_endpoint"] = np.where(np.isclose(pair_rows["_tune_sort"], start, atol=1e-9), "start", "end")
        pair_rows["_marker_order"] = pair_rows["marker_name"].map(marker_order)
        frames.append(pair_rows)

    if not frames:
        return pd.DataFrame()
    overlay = pd.concat(frames, ignore_index=True)
    return overlay.sort_values(["_pair_order", "_marker_order", "_pair_endpoint"], kind="mergesort")


def _kyhl_phase_pair_overlay_title(family: str, start: float, end: float) -> str:
    return f"KYHL {family} phase overlay: {_format_position(start)}->{_format_position(end)}"


def _format_position(position: object) -> str:
    try:
        value = _snap_tune_position(float(position))
    except (TypeError, ValueError):
        return str(position)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _format_group_label(group_key: object, group_columns: list[str]) -> str:
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    if tuple(group_columns) == ("sim_r_c", "sim_w_c"):
        return _format_grid_label(values[0], values[1])
    parts: list[str] = []
    for column, value in zip(group_columns, values, strict=True):
        if column == "tune_position":
            parts.append(_format_position(value))
        elif column == "sim_r_c":
            parts.append(f"r_c={_shared_format_simulation_parameter_value('sim_r_c', value)}")
        elif column == "sim_w_c":
            parts.append(f"w_c={_shared_format_simulation_parameter_value('sim_w_c', value)}")
        elif column == "source_file":
            parts.append(Path(str(value)).stem)
        elif column == "s_name":
            parts.append(str(value))
        elif column == "port_side" and not pd.isna(value):
            parts.append(f"port={value}")
        elif not pd.isna(value):
            parts.append(str(value))
    return " | ".join(parts)


def _position_output_stem(position_table: pd.DataFrame, position_label: str, *, grouping_mode: str) -> str:
    if grouping_mode == "tune_position" and " | " not in position_label:
        return _tune_position_filename_label(position_table)
    if grouping_mode == "grid_point":
        return _grid_point_filename_label(position_table, position_label)
    if grouping_mode == "sim_sweep":
        return f"{_simulation_family_label(position_table)}_{_safe_label(position_label)}"
    return f"position_{_safe_label(position_label)}"


def _position_plot_title(position_label: str, position_table: pd.DataFrame, *, grouping_mode: str, title_prefix: str) -> str:
    if grouping_mode == "grid_point":
        family = _grid_point_family(position_table)
        if family is not None:
            grid_label = _format_grid_label(
                position_table["sim_r_c"].iloc[0],
                position_table["sim_w_c"].iloc[0],
                depth_label=_num_depth_label(position_table),
            )
            return f"{family.title()} polar: {grid_label}"
        return f"Grid polar: {position_label}"
    if grouping_mode == "sim_sweep":
        return f"{_simulation_family_label(position_table).title()} polar: {position_label}"
    return f"{position_label}: {title_prefix}"


def _tune_position_filename_label(position_table: pd.DataFrame) -> str:
    tune_positions = position_table["tune_position"].dropna().unique()
    if len(tune_positions) != 1:
        return f"position_{_safe_label(_format_position(tune_positions[0] if len(tune_positions) else 'unknown'))}"
    tune_position = tune_positions[0]
    return f"{_position_family(tune_position)}_{_safe_label(_format_position(tune_position))}"


def _position_family(tune_position: object) -> str:
    value = _snap_tune_position(float(tune_position))
    fractional = value % 1.0
    if abs(fractional) < 1e-9:
        return "iris"
    if abs(fractional - 0.5) < 1e-9:
        return "cell"
    return f"offset_{_safe_label(f'{fractional:g}')}"


def _snap_tune_position(value: float) -> float:
    doubled = round(value * 2.0)
    snapped = doubled / 2.0
    if abs(value - snapped) < 1e-6:
        return snapped
    return value


def _format_grid_label(sim_r_c: object, sim_w_c: object, *, depth_label: str | None = None) -> str:
    grid_label = (
        f"r_c={_shared_format_simulation_parameter_value('sim_r_c', sim_r_c)}, "
        f"w_c={_shared_format_simulation_parameter_value('sim_w_c', sim_w_c)}"
    )
    if depth_label is None:
        return grid_label
    return f"depth={depth_label.removeprefix('depth_')} {grid_label}"


def _format_grid_family_label(family: str, sim_r_c: object, sim_w_c: object, *, depth_label: str | None = None) -> str:
    return f"{family} {_format_grid_label(sim_r_c, sim_w_c, depth_label=depth_label)}"


def _grid_point_filename_label(position_table: pd.DataFrame, position_label: str) -> str:
    family = _grid_point_family(position_table)
    depth_suffix = _grid_point_depth_filename_suffix(position_table)
    if family is not None:
        return (
            f"{family}{depth_suffix}_"
            f"{_safe_label(_format_grid_label(position_table['sim_r_c'].iloc[0], position_table['sim_w_c'].iloc[0]))}"
        )
    return f"grid_{_safe_label(position_label)}"


def _grid_point_family(position_table: pd.DataFrame) -> str | None:
    if "position_family" not in position_table:
        return None
    families = [str(family) for family in position_table["position_family"].dropna().unique()]
    if len(families) == 1 and families[0] != "unknown":
        return families[0]
    return None


def _grid_point_depth_group_columns(table: pd.DataFrame) -> list[str]:
    return _shared_grid_point_depth_group_columns(table)


def _grid_point_depth_filename_suffix(position_table: pd.DataFrame) -> str:
    depth_label = _num_depth_label(position_table)
    if depth_label is None:
        return ""
    return f"_{depth_label}"


def _num_depth_label(table: pd.DataFrame) -> str | None:
    if "sim_NumDepth" not in table:
        return None
    values = pd.to_numeric(table["sim_NumDepth"], errors="coerce").dropna().unique()
    if len(values) != 1:
        return None
    return f"depth_{_format_num_depth(values[0])}"


def _simulation_family_label(position_table: pd.DataFrame) -> str:
    text_parts: list[str] = []
    for column in ("dataset_id", "source_file"):
        if column in position_table:
            text_parts.extend(str(value).lower() for value in position_table[column].dropna().unique())
    joined = " ".join(text_parts)
    if "iris" in joined:
        return "iris"
    if "cell" in joined:
        return "cell"
    if "tune_position" in position_table:
        tune_positions = position_table["tune_position"].dropna().unique()
        if len(tune_positions) == 1:
            try:
                family = _position_family(tune_positions[0])
            except (TypeError, ValueError):
                family = ""
            if family in {"iris", "cell"}:
                return family
    return "sim"


def _format_sim_sweep_label(sweep_columns: list[str], values: tuple[object, ...]) -> str:
    parts = []
    for column, value in zip(sweep_columns, values, strict=True):
        if column == "sim_NumDepth":
            parts.append(f"depth={_format_num_depth(value)}")
        elif column in {"sim_r_c", "sim_w_c"}:
            parts.append(
                f"{_sim_sweep_column_label(column)}="
                f"{_shared_format_simulation_parameter_value(column, value)}"
            )
        else:
            parts.append(f"{_sim_sweep_column_label(column)}={_format_grid_value(value)}")
    return "_".join(parts)


def _sim_sweep_group_columns(table: pd.DataFrame, sweep_columns: list[str]) -> list[str]:
    group_columns = []
    if "sim_NumDepth" in table and table["sim_NumDepth"].dropna().nunique() > 1:
        group_columns.append("sim_NumDepth")
    group_columns.extend(column for column in sweep_columns if column not in group_columns)
    if "sim_NumDepth" not in group_columns and "tune_position" in table and table["tune_position"].dropna().nunique() > 1:
        group_columns.append("tune_position")
    return group_columns


def _varying_sim_sweep_columns(marker_points: pd.DataFrame) -> list[str]:
    return _shared_varying_sim_sweep_columns(marker_points)


def _sim_sweep_column_label(column: str) -> str:
    if column == "tune_position":
        return "tune_position"
    label = column.removeprefix("sim_")
    if label.startswith("Depth") and len(label) > len("Depth"):
        label = label[len("Depth") :]
    return _camel_to_snake(label)


def _format_num_depth(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}p0"
    return _format_grid_value(value)


def _camel_to_snake(label: str) -> str:
    converted = []
    previous_is_lower_or_digit = False
    for character in label:
        if character in {" ", "-", "."}:
            converted.append("_")
            previous_is_lower_or_digit = False
            continue
        if character.isupper() and previous_is_lower_or_digit:
            converted.append("_")
        converted.append(character.lower())
        previous_is_lower_or_digit = character.islower() or character.isdigit()
    return "".join(converted)


def _format_grid_value(value: object) -> str:
    return _shared_format_grid_value(value)


def _safe_label(label: str) -> str:
    sanitized = (
        label.replace("=", "_")
        .replace(",", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace(" ", "_")
        .replace("|", "_")
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized
