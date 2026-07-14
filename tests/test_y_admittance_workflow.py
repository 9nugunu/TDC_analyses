import json
from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.admittance import (
    extract_y11_marker_frequencies,
    load_y11_touchstone_folder,
    sample_y11_markers,
)
from deflector_tuning.runner import run_folder_analysis


def test_load_y11_touchstone_folder_preserves_cst_values_and_result_navigator_metadata(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data" / "sim" / "sim_sweep_260701_y_case"
    folder.mkdir(parents=True)
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\n"1"\t"56.09"\n"2"\t"56.10"\n',
        encoding="utf-8",
    )
    (folder / "run_1.y1p").write_text(
        "# GHz Y RI R 1\n2.85 0.001 0.002\n",
        encoding="utf-8",
    )
    (folder / "run_2.y1p").write_text(
        "# GHz Y RI R 1\n2.85 0.003 0.004\n",
        encoding="utf-8",
    )

    table = load_y11_touchstone_folder(folder)

    assert table[["source_file", "run_id", "sim_r_c"]].to_dict("records") == [
        {"source_file": "run_1.y1p", "run_id": 1, "sim_r_c": pytest.approx(56.09)},
        {"source_file": "run_2.y1p", "run_id": 2, "sim_r_c": pytest.approx(56.10)},
    ]
    assert table[["y_re_siemens", "y_im_siemens"]].to_numpy().tolist() == [
        [pytest.approx(0.001), pytest.approx(0.002)],
        [pytest.approx(0.003), pytest.approx(0.004)],
    ]
    assert table["reference_ohm"].tolist() == [1.0, 1.0]


def test_sample_y11_markers_uses_compact_frequency_and_value_columns(tmp_path: Path) -> None:
    folder = tmp_path / "sim_sweep_260701_y_case"
    folder.mkdir()
    (folder / "run_1.y1p").write_text(
        "# GHz Y RI R 1\n2.84 0.001 0.002\n2.85 0.003 0.004\n",
        encoding="utf-8",
    )
    markers = pd.DataFrame(
        {
            "marker_name": ["f_2pi3"],
            "freq_ghz": [2.85],
            "marker_role": ["sim"],
            "marker_source": ["test"],
        }
    )

    sampled = sample_y11_markers(load_y11_touchstone_folder(folder), markers)

    assert {"y_re_siemens", "y_im_siemens", "freq_target_ghz"}.issubset(sampled.columns)
    assert "y_real_siemens" not in sampled
    assert "target_freq_ghz" not in sampled


def test_extract_y11_marker_frequencies_prefers_explicit_cst_kyhl_marker_table(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_case"
    processed = folder / "processed"
    processed.mkdir(parents=True)
    (processed / "ambiguous_summary.csv").write_text(
        "mode_index,freq_120_GHz,freq_90_GHz\n1,2.143,2.108\n",
        encoding="utf-8",
    )
    (processed / "case_kyhl_markers.csv").write_text(
        "phase_label,cst_freq_GHz\npi/2,2.878\n\"mean(pi/2,2pi/3)\",2.868\n2pi/3,2.857\n",
        encoding="utf-8",
    )

    markers = extract_y11_marker_frequencies(folder, marker_role="sim")

    assert markers[["marker_name", "freq_ghz"]].to_dict("records") == [
        {"marker_name": "f_2pi3", "freq_ghz": pytest.approx(2.857)},
        {"marker_name": "f_mean", "freq_ghz": pytest.approx(2.868)},
        {"marker_name": "f_pi2", "freq_ghz": pytest.approx(2.878)},
    ]


def test_run_folder_analysis_routes_y11_sweep_to_complex_admittance_figure(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    y_folder = data_root / "sim" / "sim_sweep_260701_y_case"
    y_folder.mkdir(parents=True)
    (y_folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\n"1"\t"56.09"\n"2"\t"56.10"\n',
        encoding="utf-8",
    )
    for run_id, real_offset in ((1, 0.001), (2, 0.003)):
        (y_folder / f"run_{run_id}.y1p").write_text(
            "# GHz Y RI R 1\n"
            f"2.85 {real_offset} 0.002\n"
            f"2.86 {real_offset} 0.004\n"
            f"2.87 {real_offset} 0.006\n",
            encoding="utf-8",
        )
    dispersion_folder = data_root / "sim" / "sim_dispersion_260505_case"
    processed = dispersion_folder / "processed"
    processed.mkdir(parents=True)
    (processed / "mode_summary.csv").write_text(
        "mode_index,freq_120_GHz,freq_90_GHz\n1,2.85,2.87\n",
        encoding="utf-8",
    )

    result = run_folder_analysis(
        sparameter_path=y_folder,
        dispersion_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=data_root,
    )

    assert result.analysis_modes == ("y11_admittance",)
    assert set(result.tables) == {"markers", "y11_pts"}
    assert set(result.figures) == {"y11_complex"}
    assert result.figures["y11_complex"]["marker_sweep"].exists()
    marker_points = pd.read_csv(result.tables["y11_pts"])
    assert marker_points.groupby("marker_name").size().to_dict() == {
        "f_2pi3": 2,
        "f_mean": 2,
        "f_pi2": 2,
    }
    assert marker_points["sim_r_c"].tolist() == [56.09, 56.1, 56.09, 56.1, 56.09, 56.1]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["table_contract"] == "y11"
    assert manifest["table_schema_version"] == 2


def test_run_folder_analysis_y11_tables_only_uses_compact_contract(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    y_folder = data_root / "sim" / "sim_sweep_260701_y_case"
    y_folder.mkdir(parents=True)
    (y_folder / "run_1.y1p").write_text(
        "# GHz Y RI R 1\n2.85 0.001 0.002\n2.86 0.003 0.004\n2.87 0.005 0.006\n",
        encoding="utf-8",
    )
    dispersion_folder = data_root / "sim" / "sim_dispersion_260505_case"
    processed = dispersion_folder / "processed"
    processed.mkdir(parents=True)
    (processed / "mode_summary.csv").write_text(
        "mode_index,freq_120_GHz,freq_90_GHz\n1,2.85,2.87\n",
        encoding="utf-8",
    )

    result = run_folder_analysis(
        sparameter_path=y_folder,
        dispersion_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=data_root,
        tables_only=True,
    )

    assert set(result.tables) == {"markers", "y11_pts"}
    assert result.tables["y11_pts"].name == "y11_pts.csv"
    assert result.figures == {}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["table_contract"] == "y11"
    assert manifest["table_schema_version"] == 2
