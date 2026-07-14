from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.table_export import (
    project_table,
    save_standard_tables,
    save_table_contract,
)
from deflector_tuning.table_schema import STANDARD_TABLE_SPECS


def _minimal_standard_tables() -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        (key, pd.DataFrame({"dataset_id": ["sim_sweep_260713_case"]}))
        for key in STANDARD_TABLE_SPECS
    )


def _y11_tables() -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        (
            (
                "markers",
                pd.DataFrame(
                    {
                        "dataset_id": ["sim_sweep_260713_y_case"],
                        "marker_name": ["f_2pi3"],
                        "marker_role": ["sim"],
                    }
                ),
            ),
            (
                "y11_pts",
                pd.DataFrame(
                    {
                        "dataset_id": ["sim_sweep_260713_y_case"],
                        "y_name": ["Y11"],
                        "y_re_siemens": [0.01],
                        "y_im_siemens": [0.02],
                        "freq_target_ghz": [2.85],
                    }
                ),
            ),
        )
    )


def _z11_tables() -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        (
            (
                "markers",
                pd.DataFrame(
                    {
                        "dataset_id": ["sim_sweep_260713_z_case"],
                        "marker_name": ["f_2pi3"],
                    }
                ),
            ),
            (
                "z11_pts",
                pd.DataFrame(
                    {
                        "dataset_id": ["sim_sweep_260713_z_case"],
                        "z_name": ["Z11"],
                        "z_re_ohm": [12.0],
                        "z_im_ohm": [-3.0],
                        "freq_target_ghz": [2.85],
                    }
                ),
            ),
        )
    )


def test_projection_keeps_constant_results_and_moves_constant_metadata_to_context() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["sim_sweep_260713_case"] * 2,
            "marker_role": ["sim"] * 2,
            "op_admit_mag": [1.5, 1.5],
        }
    )

    projected, constants = project_table("kyhl_admit_pts", table)

    assert list(projected) == ["dataset_id", "op_admit_mag"]
    assert constants == {"marker_role": "sim"}


def test_projection_keeps_varying_sim_sweep_columns() -> None:
    table = pd.DataFrame(
        {
            "dataset_id": ["sim_grid_260626_case"] * 2,
            "source_file": ["a.s1p", "b.s1p"],
            "sim_r_c": [55.5, 56.5],
            "s_db": [-20.0, -21.0],
        }
    )

    projected, constants = project_table("marker_pts", table)

    assert "sim_r_c" in projected
    assert "sim_r_c" not in constants


def test_empty_table_omits_context_headers() -> None:
    table = pd.DataFrame(
        columns=["dataset_id", "marker_name", "marker_role", "phase_err_rms_deg"]
    )

    projected, constants = project_table("phase_stats", table)

    assert list(projected) == ["dataset_id", "marker_name", "phase_err_rms_deg"]
    assert constants == {}


def test_successful_save_removes_registered_legacy_files_only(tmp_path: Path) -> None:
    (tmp_path / "marker_points.csv").write_text("old\n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("user\n", encoding="utf-8")

    result = save_standard_tables(_minimal_standard_tables(), tmp_path)

    assert result.paths["marker_pts"] == tmp_path / "marker_pts.csv"
    assert not (tmp_path / "marker_points.csv").exists()
    assert (tmp_path / "notes.csv").read_text(encoding="utf-8") == "user\n"
    assert len(result.paths) == 13


def test_failed_staging_preserves_existing_managed_files(monkeypatch, tmp_path: Path) -> None:
    legacy = tmp_path / "marker_points.csv"
    legacy.write_text("old\n", encoding="utf-8")

    def fail_to_csv(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="disk full"):
        save_standard_tables(_minimal_standard_tables(), tmp_path)

    assert legacy.read_text(encoding="utf-8") == "old\n"


def test_save_rejects_noncanonical_table_key_order(tmp_path: Path) -> None:
    tables = _minimal_standard_tables()
    tables.move_to_end("markers")

    with pytest.raises(ValueError, match="canonical key order"):
        save_standard_tables(tables, tmp_path)


def test_save_y11_tables_projects_context_and_removes_legacy(tmp_path: Path) -> None:
    (tmp_path / "y11_marker_points.csv").write_text("old\n", encoding="utf-8")

    result = save_table_contract(_y11_tables(), tmp_path, "y11")

    assert list(result.paths) == ["markers", "y11_pts"]
    assert (tmp_path / "y11_pts.csv").exists()
    assert not (tmp_path / "y11_marker_points.csv").exists()
    assert result.constants["y11_pts"]["y_name"] == "Y11"


def test_save_z11_tables_rolls_back_complete_contract_on_replace_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "markers.csv"
    z11_path = tmp_path / "z11_pts.csv"
    marker_path.write_text("old markers\n", encoding="utf-8")
    z11_path.write_text("old z11\n", encoding="utf-8")
    real_replace = Path.replace
    calls = 0

    def fail_second_replace(source: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("replace failed")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_table_contract(_z11_tables(), tmp_path, "z11")

    assert marker_path.read_text(encoding="utf-8") == "old markers\n"
    assert z11_path.read_text(encoding="utf-8") == "old z11\n"
    assert not list(tmp_path.glob(".tables-stage-*"))
