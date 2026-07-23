from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deflector_tuning.data_loading.central_loader import DataLoader
from deflector_tuning.data_loading.vna50 import (
    export_vna50_touchstone_dataset,
    impedance_to_reflection,
)


def test_impedance_to_reflection_uses_fixed_reference_impedance() -> None:
    values = np.array([50 + 0j, 0 + 0j, 100 + 50j])
    expected = (values - 50.0) / (values + 50.0)

    result = impedance_to_reflection(values)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("reference", [0.0, -50.0, np.nan, np.inf])
def test_impedance_to_reflection_rejects_invalid_reference(reference: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        impedance_to_reflection([50 + 0j], reference)


def test_impedance_to_reflection_rejects_nonfinite_z() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        impedance_to_reflection([complex(np.nan, 0.0)])


def test_impedance_to_reflection_rejects_singular_denominator() -> None:
    with pytest.raises(ValueError, match="singular"):
        impedance_to_reflection([-50 + 0j])


def test_export_vna50_dataset_round_trips_through_s_loader(tmp_path: Path) -> None:
    source = _make_two_run_z_dataset(tmp_path)
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"

    first = export_vna50_touchstone_dataset(source, output)
    second = export_vna50_touchstone_dataset(source, output)

    assert first == second
    assert first.dataset_path == output
    assert [path.name for path in first.touchstone_files] == ["run_1.s1p", "run_2.s1p"]
    assert first.manifest_path == output / "transform_manifest.json"
    assert (output / "result_navigator.csv").read_bytes() == (
        source / "result_navigator.csv"
    ).read_bytes()

    reloaded = DataLoader().load(output).sort_values(
        ["source_file", "freq_ghz"],
        kind="mergesort",
    )
    source_z = np.array([50 + 0j, 0 + 0j, 100 + 50j, 25 - 10j])
    expected = impedance_to_reflection(source_z)
    actual = reloaded["s_real"].to_numpy() + 1j * reloaded["s_imag"].to_numpy()

    assert set(reloaded["s_name"]) == {"S11"}
    assert reloaded["reference_ohm"].unique().tolist() == [50.0]
    assert reloaded["run_id"].drop_duplicates().tolist() == [1, 2]
    assert reloaded["sim_tuner_insertion_depth"].drop_duplicates().tolist() == [0, 4]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_dataset"] == source.name
    assert manifest["output_dataset"] == output.name
    assert manifest["reference_ohm"] == 50.0
    assert manifest["point_count_per_file"] == 2
    assert len(manifest["files"]) == 2


def test_export_vna50_rejects_mismatched_frequency_grids(tmp_path: Path) -> None:
    source = _make_two_run_z_dataset(tmp_path)
    (source / "run_2.z1p").write_text(
        "# GHz Z RI R 1\n2.6 100 50\n2.8 25 -10\n",
        encoding="utf-8",
    )
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"

    with pytest.raises(ValueError, match="frequency grids"):
        export_vna50_touchstone_dataset(source, output)


@pytest.mark.parametrize(
    ("file_names", "message"),
    [
        (("run_1.z1p", "copy_1.z1p"), "duplicated run ID"),
        (("trace.z1p",), "missing run ID"),
    ],
)
def test_export_vna50_rejects_invalid_file_run_ids(
    tmp_path: Path,
    file_names: tuple[str, ...],
    message: str,
) -> None:
    source = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_zin"
    source.mkdir(parents=True)
    (source / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"depth"\n"1"\t"0"\n',
        encoding="utf-8",
    )
    for file_name in file_names:
        (source / file_name).write_text(
            "# GHz Z RI R 1\n2.6 50 0\n",
            encoding="utf-8",
        )
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"

    with pytest.raises(ValueError, match=message):
        export_vna50_touchstone_dataset(source, output)


def test_export_vna50_rejects_multiport_z_input(tmp_path: Path) -> None:
    source = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_zin"
    source.mkdir(parents=True)
    (source / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"depth"\n"1"\t"0"\n',
        encoding="utf-8",
    )
    (source / "run_1.z2p").write_text(
        "# GHz Z RI R 1\n2.6 50 0 0 0 0 0 50 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"

    with pytest.raises(ValueError, match="one-port Z11"):
        export_vna50_touchstone_dataset(source, output)


def test_export_vna50_rejects_unowned_existing_output(tmp_path: Path) -> None:
    source = _make_two_run_z_dataset(tmp_path)
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"
    output.mkdir(parents=True)
    (output / "unrelated.s1p").write_text(
        "# GHz S RI R 50\n2.6 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting output dataset"):
        export_vna50_touchstone_dataset(source, output)


def test_export_vna50_cli_accepts_source_and_output_paths(tmp_path: Path) -> None:
    source = _make_two_run_z_dataset(tmp_path)
    output = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_vna50"
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_vna50_touchstone.py",
            str(source),
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(output)
    assert (output / "transform_manifest.json").exists()


def _make_two_run_z_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "data" / "sim" / "sim_sweep_260713_tuner_zin"
    source.mkdir(parents=True)
    (source / "result_navigator.csv").write_text(
        '" 3D Run ID"\t"tuner_insertion_depth"\n'
        '"1"\t"0"\n'
        '"2"\t"4"\n'
        '"0"\t""\n',
        encoding="utf-8",
    )
    (source / "run_1.z1p").write_text(
        "# GHz Z RI R 1\n2.6 50 0\n2.7 0 0\n",
        encoding="utf-8",
    )
    (source / "run_2.z1p").write_text(
        "# GHz Z RI R 1\n2.6 100 50\n2.7 25 -10\n",
        encoding="utf-8",
    )
    return source
