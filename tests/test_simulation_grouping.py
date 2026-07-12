import pandas as pd

from deflector_tuning.visualization.simulation_grouping import (
    format_grid_value,
    format_simulation_parameter_token,
    format_simulation_parameter_value,
    grid_point_depth_group_columns,
    varying_sim_sweep_columns,
)


def test_varying_sim_sweep_columns_includes_geometry_and_excludes_num_metadata() -> None:
    table = pd.DataFrame(
        {
            "sim_r_c": [54.59, 55.59],
            "sim_w_c": [18.3224, 19.3224],
            "sim_NumDepth": [1, 2],
            "sim_DepthPlunger_offset": [-3.0, 0.0],
            "sim_FixedOffset": [1.0, 1.0],
            "dataset_id": ["sim_sweep", "sim_sweep"],
        }
    )

    assert varying_sim_sweep_columns(table) == ["sim_r_c", "sim_w_c", "sim_DepthPlunger_offset"]


def test_grid_point_depth_group_columns_requires_multiple_depths() -> None:
    single_depth = pd.DataFrame({"sim_NumDepth": [1, 1]})
    multiple_depths = pd.DataFrame({"sim_NumDepth": [1, 2]})

    assert grid_point_depth_group_columns(single_depth) == []
    assert grid_point_depth_group_columns(multiple_depths) == ["sim_NumDepth"]


def test_format_grid_value_preserves_numeric_and_text_values() -> None:
    assert format_grid_value(54.5900) == "54.59"
    assert format_grid_value(-3.0) == "-3"
    assert format_grid_value("reference") == "reference"


def test_simulation_parameter_tokens_keep_sortable_decimal_width() -> None:
    assert format_simulation_parameter_value("sim_r_c", 56.1) == "56.10"
    assert format_simulation_parameter_token("sim_r_c", 56.1) == "56p10"
    assert format_simulation_parameter_token("sim_r_c", 56.01) == "56p01"
    assert format_simulation_parameter_token("sim_w_c", 19.32) == "19p3200"
