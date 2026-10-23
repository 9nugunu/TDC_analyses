from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.field3d import (
    find_field3d_pairs,
    summarize_field3d_file,
)
from deflector_tuning.workflows.field3d import (
    load_cached_field3d_result,
    run_field3d_analysis,
)


def _write_pair(
    folder: Path,
    *,
    case_id: str,
    tip_z_mm: float | None,
) -> None:
    suffix = f"_{case_id}"
    for kind, component, units, amplitude in (
        ("e", "E", "V/m", 2.0),
        ("h", "H", "A/m", 0.5),
    ):
        header = (
            f"x [mm] y [mm] z [mm] "
            f"{component}xRe [{units}] {component}xIm [{units}] "
            f"{component}yRe [{units}] {component}yIm [{units}] "
            f"{component}zRe [{units}] {component}zIm [{units}]"
        )
        rows = [header, "-" * len(header)]
        for z_mm in range(5):
            for y_mm in range(-2, 3):
                for x_mm in range(-2, 3):
                    in_plunger = (
                        tip_z_mm is not None
                        and z_mm >= tip_z_mm
                        and x_mm**2 + y_mm**2 <= 1.1**2
                    )
                    value = 0.0 if in_plunger else amplitude
                    rows.append(
                        f"{x_mm} {y_mm} {z_mm} {value} 0 0 0 0 0"
                    )
        (folder / f"{kind}-field [1]{suffix}.txt").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )


def test_field3d_workflow_saves_compact_outputs_and_reuses_unchanged_manifest(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "data" / "sim" / "sim_profile_260723_field3d"
    input_dir.mkdir(parents=True)
    _write_pair(input_dir, case_id="NoPlunger", tip_z_mm=None)
    _write_pair(input_dir, case_id="NumDepth0.5", tip_z_mm=2.0)
    pairs = find_field3d_pairs(input_dir)
    output_dir = tmp_path / "fig" / "analyses" / input_dir.name

    result = run_field3d_analysis(
        pairs,
        output_dir=output_dir,
        table_dir=output_dir / "tables",
        figure_root=output_dir / "figures",
        sparameter_path=input_dir,
        dispersion_path=tmp_path / "dispersion.txt",
        marker_role="sim",
        plunger_radius_mm=1.1,
        first_tip_z_mm=2.0,
        cell_body_mm=2.0,
        iris_thickness_mm=0.0,
    )

    assert result.analysis_modes == ("field3d",)
    assert list(result.tables) == [
        "field3d_src",
        "field3d_energy",
        "slater_pos",
        "slater_vol",
    ]
    assert set(result.figures["field3d"]) == {
        "field_comp",
        "slater_pos",
        "slater_vol",
    }
    assert all(path.is_file() for path in result.tables.values())
    assert all(path.is_file() for path in result.figures["field3d"].values())
    slater_table = pd.read_csv(result.tables["slater_pos"])
    assert {
        "inserted_case_id",
        "inserted_sample_z_mm",
        "inserted_step_mm",
        "inserted_e_j",
        "inserted_h_j",
        "inserted_k_e_minus_h_j",
        "inserted_k_over_u",
        "inserted_field_basis",
        "k_over_u",
    }.issubset(slater_table.columns)
    volume_table = pd.read_csv(result.tables["slater_vol"])
    assert volume_table.loc[0, "integration_basis"] == (
        "NoPlunger field over nominal-tip-to-zmax cylinder"
    )
    assert volume_table.loc[0, "num_depth"] == 0.5
    assert volume_table.loc[0, "tip_z_mm"] == 2.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["field3d"]["schema_version"] == 6
    assert manifest["field3d"]["radius_mm"] == 1.1
    assert manifest["field3d"]["radius_basis"] == "configured_physical_radius"
    assert manifest["field3d"]["radius_source_case"] == "NumDepth0.5"
    quantities = manifest["field3d"]["slater_quantities"]
    assert quantities["baseline"]["field_basis"] == "NoPlunger"
    assert quantities["baseline"]["normalization"] == (
        "NoPlunger_export_region_energy"
    )
    assert quantities["inserted_state"]["field_basis"] == (
        "per_position_inserted_case"
    )
    assert quantities["inserted_state"]["sample"] == (
        "vacuum_plane_before_detected_tip"
    )
    assert quantities["inserted_state"]["normalization"] == (
        "same_inserted_case_export_region_energy"
    )
    assert quantities["inserted_state"]["frequency_basis"] == (
        "fixed_frequency_complex_monitor"
    )
    assert manifest["field3d"]["local_comparison_quantity"] == (
        "signed_K_over_U"
    )
    assert quantities["cumulative_volume"]["field_basis"] == "NoPlunger"
    assert quantities["cumulative_volume"]["sample"] == (
        "radius_limited_cylinder_from_nominal_tip_to_z_max"
    )
    assert manifest["field3d"]["plunger_axis"]["first_tip_z_mm"] == 2.0
    assert manifest["field3d"]["absolute_frequency_shift_claimed"] is False

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("unchanged field3d run must reuse saved outputs")

    cached = run_field3d_analysis(
        pairs,
        output_dir=output_dir,
        table_dir=output_dir / "tables",
        figure_root=output_dir / "figures",
        sparameter_path=input_dir,
        dispersion_path=tmp_path / "dispersion.txt",
        marker_role="sim",
        summarize_file=fail_if_recomputed,
        plunger_radius_mm=1.1,
        first_tip_z_mm=2.0,
        cell_body_mm=2.0,
        iris_thickness_mm=0.0,
    )

    assert cached == result


@pytest.mark.parametrize(
    ("parameter", "changed_value"),
    [
        ("plunger_radius_mm", 1.5),
        ("first_tip_z_mm", 1.0),
        ("cell_body_mm", 3.0),
        ("iris_thickness_mm", 1.0),
    ],
)
def test_field3d_cache_recomputes_after_each_calculation_parameter_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    parameter: str, changed_value: float,
) -> None:
    import deflector_tuning.workflows.field3d as workflow

    def save_placeholder(table, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test figure")
        return path

    for name in (
        "plot_field3d_components", "plot_slater_positions", "plot_slater_volumes",
    ):
        monkeypatch.setattr(workflow, name, save_placeholder)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_pair(input_dir, case_id="NoPlunger", tip_z_mm=None)
    _write_pair(input_dir, case_id="NumDepth0.5", tip_z_mm=2.0)
    pairs = find_field3d_pairs(input_dir)
    output_dir = tmp_path / "output"
    paths = dict(
        output_dir=output_dir,
        table_dir=output_dir / "tables",
        figure_root=output_dir / "figures",
        sparameter_path=input_dir,
        dispersion_path=tmp_path / "dispersion.txt",
        marker_role="sim",
    )
    parameters = dict(
        plunger_radius_mm=1.1, first_tip_z_mm=2.0,
        cell_body_mm=2.0, iris_thickness_mm=0.0,
    )
    result = run_field3d_analysis(pairs, **paths, **parameters)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summarized = []

    def record_summary(path):
        summarized.append(path)
        return summarize_field3d_file(path)

    parameters[parameter] = changed_value
    run_field3d_analysis(pairs, **paths, **parameters, summarize_file=record_summary)
    assert len(summarized) == 4
    updated = json.loads(result.manifest_path.read_text(encoding="utf-8"))["field3d"]
    actual = (
        updated["radius_mm"] if parameter == "plunger_radius_mm"
        else updated["plunger_axis"][parameter]
    )
    assert actual == changed_value
    summarized.clear()
    run_field3d_analysis(pairs, **paths, **parameters, summarize_file=record_summary)
    assert summarized == []

    assert load_cached_field3d_result(
        result.manifest_path, source_signatures=manifest["field3d"]["sources"],
        **parameters,
    ) == result
    # A caller omitting parameters requests the documented physical defaults.
    assert load_cached_field3d_result(
        result.manifest_path, source_signatures=manifest["field3d"]["sources"],
    ) is None

    # A complete legacy manifest must be recomputed, never deleted or reused.
    legacy = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    legacy["field3d"]["schema_version"] = 5
    result.manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_cached_field3d_result(
        result.manifest_path, source_signatures=legacy["field3d"]["sources"],
        **parameters,
    ) is None
    assert all(path.is_file() for path in result.tables.values())
