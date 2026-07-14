from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.one_port_matrix import (
    detect_one_port_matrix_lane,
    load_z11_touchstone_folder,
    sample_z11_markers,
)
from deflector_tuning.runner import run_folder_analysis


def _markers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_name": ["f_2pi3"],
            "freq_ghz": [2.85],
            "marker_role": ["sim"],
            "marker_source": ["test"],
        }
    )


def _z11_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "sim_sweep_260713_z_case"
    folder.mkdir()
    (folder / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"r_c"\n"1"\t"56.09"\n',
        encoding="utf-8",
    )
    (folder / "run_1.z1p").write_text(
        "# GHz Z RI R 50\n2.84 10.0 -1.0\n2.85 12.0 -3.0\n",
        encoding="utf-8",
    )
    return folder


def test_detect_one_port_matrix_lane_rejects_mixed_y_and_z(tmp_path: Path) -> None:
    (tmp_path / "a.y1p").write_text(
        "# GHz Y RI R 1\n2.85 0.1 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "b.z1p").write_text(
        "# GHz Z RI R 50\n2.85 10 20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"both \.y1p and \.z1p"):
        detect_one_port_matrix_lane(tmp_path)


def test_load_and_sample_z11_uses_compact_impedance_columns(tmp_path: Path) -> None:
    loaded = load_z11_touchstone_folder(_z11_folder(tmp_path))
    sampled = sample_z11_markers(loaded, _markers())

    assert {"z_re_ohm", "z_im_ohm", "freq_target_ghz"}.issubset(sampled.columns)
    assert "z_real_ohm" not in sampled
    assert sampled.iloc[0]["z_re_ohm"] == pytest.approx(12.0)
    assert sampled.iloc[0]["z_im_ohm"] == pytest.approx(-3.0)
    assert sampled.iloc[0]["sim_r_c"] == pytest.approx(56.09)


def test_run_folder_analysis_z11_writes_compact_table_and_complex_figure(
    tmp_path: Path,
) -> None:
    z_folder = _z11_folder(tmp_path)
    dispersion_folder = tmp_path / "sim_dispersion_260505_case"
    processed = dispersion_folder / "processed"
    processed.mkdir(parents=True)
    (processed / "mode_summary.csv").write_text(
        "mode_index,freq_120_GHz,freq_90_GHz\n1,2.84,2.86\n",
        encoding="utf-8",
    )

    result = run_folder_analysis(
        sparameter_path=z_folder,
        dispersion_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path,
    )

    assert result.analysis_modes == ("z11_impedance",)
    assert set(result.tables) == {"markers", "z11_pts"}
    assert result.tables["z11_pts"].name == "z11_pts.csv"
    assert result.figures["z11_complex"]["marker_sweep"].exists()
