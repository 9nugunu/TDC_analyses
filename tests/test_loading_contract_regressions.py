from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.loaders.sim_loader import _merge_result_navigator
from deflector_tuning.data_loading.one_port_matrix import load_one_port_matrix_folder
from deflector_tuning.analysis.marker_pipeline import build_marker_analysis, build_marker_phase_polar_table


@pytest.mark.parametrize("by_run", [True, False])
def test_navigator_rejects_ambiguous_join(by_run):
    table = pd.DataFrame({"run_id": [1 if by_run else None], "tune_position": [0.5]})
    navigator = pd.DataFrame({"run_id": [1, 1] if by_run else [1, 2], "sim_NumTune": [0, 0]})
    with pytest.raises(ValueError, match="Navigator.*unique"):
        _merge_result_navigator(table, navigator)


@pytest.mark.parametrize("layer", ["raw", "prepro"])
@pytest.mark.parametrize("lane,parameter", [("y11", "Y"), ("z11", "Z")])
def test_matrix_preserves_experiment_identity(tmp_path, layer, lane, parameter):
    folder = tmp_path / "data" / layer / f"{layer}_sweep_260723_matrix"
    folder.mkdir(parents=True)
    (folder / f"trace.{parameter.lower()}1p").write_text(f"# GHz {parameter} RI R 50\n2.85 1 2\n")
    table = load_one_port_matrix_folder(folder, lane)
    assert table.data_kind.tolist() == ["experiment"]
    assert table.data_layer.tolist() == [layer]


def test_magnitude_only_pipeline_has_no_phase_products(tmp_path):
    folder = tmp_path / "data/sim/sim_grid_undated_magnitude"
    folder.mkdir(parents=True)
    (folder / "s11.txt").write_text("#Parameters = {r_c=56; w_c=20}\n2.85 -6\n2.87 -3\n")
    dispersion = tmp_path / "dispersion/processed"
    dispersion.mkdir(parents=True)
    (dispersion / "dispersion_summary.csv").write_text("mode_index,freq_90_GHz,freq_120_GHz\n1,2.87,2.85\n")
    tables = build_marker_analysis(sparameter_path=folder, dispersion_path=dispersion.parent, marker_role="sim")
    assert len(tables["marker_pts"]) == 3
    assert tables["marker_pts"].s_db.notna().all()
    for key in ("phase_polar", "phase_adv", "kyhl_admit_pts", "coupler_params", "geom_phase", "rc_line"):
        assert tables[key].empty, key


def test_matrix_navigator_rejects_duplicate_run_id(tmp_path):
    folder = tmp_path / "sim_sweep_260723_matrix"
    folder.mkdir()
    (folder / "run_1.y1p").write_text("# GHz Y RI R 50\n2.85 1 2\n")
    (folder / "result_navigator.csv").write_text('"3D Run ID"\t"r_c"\n1\t56\n1\t57\n')
    with pytest.raises(ValueError, match="Navigator.*unique"):
        load_one_port_matrix_folder(folder, "y11")


def test_polar_table_preserves_unknown_geometry_and_rejects_ambiguous_duplicates():
    points = pd.DataFrame({"source_file": ["trace", "trace"], "sim_custom_length": [1, 2],
                           "marker_name": ["f_2pi3", "f_2pi3"], "s_phase_deg": [10., 80.]})
    table = build_marker_phase_polar_table(points)
    assert table.sim_custom_length.tolist() == [1, 2]
    assert table.f_2pi3_phase_deg.tolist() == [10., 80.]
    with pytest.raises(ValueError, match="Duplicate.*marker"):
        build_marker_phase_polar_table(points.drop(columns="sim_custom_length"))
