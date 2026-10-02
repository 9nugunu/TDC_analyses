from pathlib import Path

import pandas as pd
import pytest

from deflector_tuning.data_loading.field3d import Field3DPair

import deflector_tuning.runner as runner
import deflector_tuning.workflows.marker_analysis as marker_workflow
import deflector_tuning.workflows.marker_figures as marker_figures


def test_run_folder_analysis_uses_short_dispersion_figure_names(tmp_path: Path, monkeypatch) -> None:
    dispersion_folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_case"
    dispersion_folder.mkdir(parents=True)
    (dispersion_folder / "phase_sweep.txt").write_text(
        "\n".join(
            [
                "#",
                '#"phase"\t"Mode 1 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.10",
                "90\t2.90",
                "120\t2.86",
                "180\t2.84",
                "#",
                '#"phase"\t"Mode 2 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.30",
                "90\t3.10",
                "120\t3.05",
                "180\t3.00",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("dispersion input must not use S-parameter marker analysis")

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_nodal_shift", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("dispersion",)
    assert "dispersion" in result.figures
    assert result.figures["dispersion"]["all_modes"].exists()
    assert set(result.figures["dispersion"]) == {
        "all_modes",
        "mode_01",
        "mode_02",
    }
    assert result.figures["dispersion"]["all_modes"].name == "all_modes.png"
    assert result.figures["dispersion"]["mode_01"].name == "mode_01.png"
    assert result.figures["dispersion"]["mode_02"].name == "mode_02.png"
    assert set(result.tables) == {
        "phase_sweep_long",
        "phase_sweep_wide",
        "phase_sweep_summary",
    }
    summary = pd.read_csv(result.tables["phase_sweep_summary"])
    assert summary.loc[0, "freq_120_GHz"] == 2.86
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"dispersion"' in manifest
    assert '"s11"' not in manifest
    assert '"phase_advance"' not in manifest


def test_run_folder_analysis_uses_dispersion_only_lane_for_single_mode_cst_exports(tmp_path: Path, monkeypatch) -> None:
    dispersion_folder = tmp_path / "data" / "sim" / "dispersion_case"
    dispersion_folder.mkdir(parents=True)
    (dispersion_folder / "phase_sweep.txt").write_text(
        "\n".join(
            [
                "#",
                '#"phase"\t"Mode 1 [Real / GHz]"',
                "#-----------------------------",
                "0\t3.10",
                "90\t2.90",
                "120\t2.86",
                "180\t2.84",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("dispersion input must not use S-parameter marker analysis")

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_s11_with_markers", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_phase_advance", fail_sparameter_lane)
    monkeypatch.setattr(marker_figures, "plot_marker_phase_polar_views", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=dispersion_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("dispersion",)
    assert "dispersion" in result.figures
    assert result.figures["dispersion"]["all_modes"].exists()
    assert set(result.figures["dispersion"]) == {"all_modes", "mode_01"}
    assert result.figures["dispersion"]["all_modes"].name == "all_modes.png"
    assert result.figures["dispersion"]["mode_01"].name == "mode_01.png"
    assert set(result.tables) == {
        "phase_sweep_long",
        "phase_sweep_wide",
        "phase_sweep_summary",
    }
    summary = pd.read_csv(result.tables["phase_sweep_summary"])
    assert summary.loc[0, "freq_120_GHz"] == 2.86
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"dispersion"' in manifest
    assert '"s11"' not in manifest
    assert '"phase_advance"' not in manifest


def test_run_folder_analysis_uses_profile_only_lane_for_cst_z_profiles(tmp_path: Path, monkeypatch) -> None:
    profile_folder = tmp_path / "data" / "sim" / "sim_profile_260618_hem11_eh_zcut"
    profile_folder.mkdir(parents=True)
    (profile_folder / "E_fieldDist.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z) [Real]"',
                "#----------------------------------------------",
                "0.0\t1.0",
                "2.92\t3.0",
                "20.414\t0.5",
            ]
        ),
        encoding="utf-8",
    )
    (profile_folder / "EM_fieldPhase.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"e-field (f=2.8565) (1)_Y (Z)_phase [Real]"',
                "#----------------------------------------------",
                "0.0\t0.0",
                "2.92\t60.0",
                "20.414\t120.0",
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z)_phase [Real]"',
                "#----------------------------------------------",
                "0.0\t-20.0",
                "2.92\t40.0",
                "20.414\t100.0",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("profile input must not use S-parameter marker analysis")

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("profile",)
    assert set(result.tables) == {
        "profile_summary",
        "field_energy_ratio_pairs",
        "field_energy_ratio_summary",
    }
    assert set(result.figures) == {"profile"}
    assert set(result.figures["profile"]) == {
        "E_fieldDist",
        "EM_fieldPhase",
        "E_fieldPhase",
        "H_fieldPhase",
    }
    assert result.figures["profile"]["E_fieldDist"].exists()
    assert result.figures["profile"]["E_fieldPhase"].exists()
    assert result.figures["profile"]["H_fieldPhase"].exists()
    summary = pd.read_csv(result.tables["profile_summary"])
    assert summary["source_file"].tolist() == [
        "E_fieldDist.txt",
        "EM_fieldPhase.txt",
        "EM_fieldPhase.txt",
    ]
    assert summary["value_kind"].tolist() == ["real", "phase", "phase"]
    field_ratios = pd.read_csv(result.tables["field_energy_ratio_pairs"])
    assert "adjacent_average_iris_to_cell_energy_ratio" in field_ratios.columns
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"profile"' in manifest
    assert '"s11"' not in manifest


def test_run_folder_analysis_detects_legacy_named_cst_z_profiles(tmp_path: Path, monkeypatch) -> None:
    profile_folder = tmp_path / "data" / "sim" / "sim_EMfield_260618_PhaseDistribution"
    profile_folder.mkdir(parents=True)
    (profile_folder / "H_fieldDist.txt").write_text(
        "\n".join(
            [
                "#Parameters = {d=29.148; t=5.84}",
                '#"Z / mm"\t"h-field (f=2.8565) (1)_X (Z) [Real]"',
                "#----------------------------------------------",
                "0.0\t1.0",
                "2.92\t3.0",
                "20.414\t0.5",
            ]
        ),
        encoding="utf-8",
    )

    def fail_sparameter_lane(*args, **kwargs):
        raise AssertionError("parseable profile input must not use S-parameter marker analysis")

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", fail_sparameter_lane)
    monkeypatch.setattr(runner.DataLoader, "load", fail_sparameter_lane)

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        output_dir=tmp_path / "out",
        marker_role="sim",
        data_root=tmp_path / "data",
    )

    assert result.analysis_modes == ("profile",)
    assert set(result.figures["profile"]) == {"H_fieldDist"}


def test_runner_preserves_profile_and_dispersion_patch_seams(tmp_path: Path, monkeypatch) -> None:
    profile_path = tmp_path / "profile.txt"
    dispersion_path = tmp_path / "dispersion.txt"
    profile_path.write_text("profile", encoding="utf-8")
    dispersion_path.write_text("dispersion", encoding="utf-8")
    profile_calls: list[Path] = []
    dispersion_calls: list[Path] = []

    monkeypatch.setattr(
        runner,
        "load_field_profile_export",
        lambda path: profile_calls.append(Path(path)),
    )
    monkeypatch.setattr(
        runner,
        "load_cst_dispersion_txt",
        lambda path: dispersion_calls.append(Path(path)),
    )

    assert runner.find_cst_profile_inputs(profile_path) == (profile_path,)
    assert runner.find_cst_dispersion_inputs(dispersion_path) == (dispersion_path,)
    assert profile_calls == [profile_path]
    assert dispersion_calls == [dispersion_path]
    assert runner.BASE_ANALYSIS_MODES[0] == "marker_analysis"
    assert "sim_r_c" in runner.GRID_SCAN_REQUIRED_COLUMNS
    assert "f_2pi3" in runner.GRID_SCAN_REQUIRED_MARKERS


def test_run_folder_analysis_routes_3d_field_headers_before_1d_profile_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_folder = (
        tmp_path
        / "data"
        / "sim"
        / "sim_profile_260723_field3d_auto"
    )
    profile_folder.mkdir(parents=True)
    for kind, component, units in (
        ("e", "E", "V/m"),
        ("h", "H", "A/m"),
    ):
        header = (
            f"x [mm] y [mm] z [mm] "
            f"{component}xRe [{units}] {component}xIm [{units}] "
            f"{component}yRe [{units}] {component}yIm [{units}] "
            f"{component}zRe [{units}] {component}zIm [{units}]"
        )
        (profile_folder / f"{kind}-field [1]_NoPlunger.txt").write_text(
            header + "\n" + "-" * len(header) + "\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "fig" / "analyses" / profile_folder.name
    expected = runner.RunResult(
        output_dir=output_dir,
        tables=runner.AnalysisPaths(),
        figures=runner.FigurePaths(),
        analysis_modes=("field3d",),
        manifest_path=output_dir / "manifest.json",
    )
    recorded_pairs = []

    def fake_field3d_run(pairs, **kwargs):
        recorded_pairs.extend(pairs)
        return expected

    monkeypatch.setattr(
        runner,
        "_run_field3d_only_analysis_from_runner",
        fake_field3d_run,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "find_cst_profile_inputs",
        lambda path: (_ for _ in ()).throw(
            AssertionError("3D headers must bypass the 1D profile loader")
        ),
    )

    result = runner.run_folder_analysis(
        sparameter_path=profile_folder,
        dispersion_path=tmp_path / "dispersion",
        output_dir=output_dir,
        marker_role="sim",
    )

    assert result == expected
    assert [pair.case_id for pair in recorded_pairs] == ["NoPlunger"]


@pytest.mark.parametrize(
    ("folder_name", "input_kind"),
    [
        ("sim_profile_260723_profile", "profile"),
        ("legacy_exports", "profile"),
        ("sim_dispersion_260723_modes", "dispersion"),
        ("legacy_exports", "dispersion"),
        ("sim_profile_260723_field3d", "field3d"),
        ("legacy_exports", "field3d"),
    ],
)
def test_run_folder_analysis_rejects_tables_only_for_cst_figure_workflows(
    tmp_path: Path, monkeypatch, folder_name: str, input_kind: str,
) -> None:
    input_folder = tmp_path / "data" / "sim" / folder_name
    input_folder.mkdir(parents=True)
    input_path = input_folder / "input.txt"
    pair = Field3DPair("sample", input_folder / "e.txt", input_folder / "h.txt")
    monkeypatch.setattr(
        runner, "find_cst_dispersion_inputs",
        lambda path: (input_path,) if input_kind == "dispersion" else (),
    )
    monkeypatch.setattr(
        runner, "find_cst_field3d_pairs",
        lambda path: (pair,) if input_kind == "field3d" else (),
    )
    monkeypatch.setattr(
        runner, "find_cst_profile_inputs",
        lambda path: (input_path,) if input_kind == "profile" else (),
    )
    with pytest.raises(ValueError, match="--tables-only is supported only"):
        runner.run_folder_analysis(
            sparameter_path=input_folder,
            output_dir=tmp_path / "out",
            marker_role="sim",
            tables_only=True,
        )
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.mark.parametrize("category", ["profile", "dispersion"])
def test_run_folder_analysis_reports_empty_explicit_cst_category(
    tmp_path: Path, category: str,
) -> None:
    input_folder = tmp_path / "data" / "sim" / f"sim_{category}_260723_empty"
    input_folder.mkdir(parents=True)
    with pytest.raises(ValueError, match=f"Dataset category is {category}"):
        runner.run_folder_analysis(
            sparameter_path=input_folder,
            output_dir=tmp_path / "out",
            marker_role="sim",
        )
