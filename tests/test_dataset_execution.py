from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from deflector_tuning.workflows import dataset_execution as workflow


@pytest.mark.parametrize("marker_role", ["exp", "sim"])
def test_dataset_execution_forwards_scientific_options_and_registers_campaign(
    monkeypatch, tmp_path: Path, marker_role: str
) -> None:
    result = SimpleNamespace(output_dir=tmp_path / "analysis", manifest_path=tmp_path / "manifest.json")
    correction = object()
    defaults = object()
    calls = []
    monkeypatch.setattr(workflow, "resolve_tuning_marker_correction", lambda *args, **kwargs: calls.append(("correction", args, kwargs)) or correction)
    monkeypatch.setattr(workflow, "run_folder_analysis", lambda **kwargs: calls.append(("analysis", kwargs)) or result)
    monkeypatch.setattr(workflow, "register_matching_tuning_campaign", lambda *args, **kwargs: calls.append(("register", args, kwargs)))

    execution = workflow.run_dataset_analysis(
        dataset_id="sim_sweep_260728_zlen",
        sparameter_path=Path("sim/sim_sweep_260728_zlen"),
        dispersion_path=Path("dispersion"),
        output_dir=result.output_dir,
        marker_role=marker_role,
        data_root=tmp_path / "data",
        file_workers=3,
        plot_workers=2,
        tables_only=True,
        project_defaults=defaults,
        geometry_sweep_axis="sim_L_c",
        geometry_sweep_base=29.148,
    )

    assert [call[0] for call in calls] == (["correction"] if marker_role == "exp" else []) + ["analysis", "register"]
    analysis_call = next(call[1] for call in calls if call[0] == "analysis")
    assert analysis_call == dict(
        sparameter_path=Path("sim/sim_sweep_260728_zlen"), dispersion_path=Path("dispersion"),
        output_dir=result.output_dir, marker_role=marker_role, data_root=tmp_path / "data",
        file_workers=3, plot_workers=2, tables_only=True, project_defaults=defaults,
        marker_correction=correction if marker_role == "exp" else None,
        geometry_sweep_axis="sim_L_c", geometry_sweep_base=29.148,
    )
    assert calls[-1] == ("register", ("sim_sweep_260728_zlen",), dict(manifest_path=result.manifest_path, data_root=tmp_path / "data"))
    assert execution.analysis is result
    assert execution.campaign_match is None
    assert execution.plunger_sensitivity is None
    assert execution.tuning_cmp is None
    assert execution.phase_shifts is None


@pytest.mark.parametrize("tables_only", [False, True])
@pytest.mark.parametrize("comparison_enabled", [False, True])
@pytest.mark.parametrize("measurement_kind", ["state", "auxiliary"])
def test_dataset_execution_applies_shared_campaign_postprocessing_policy(
    monkeypatch, tmp_path: Path, tables_only: bool, comparison_enabled: bool,
    measurement_kind: str,
) -> None:
    analysis = SimpleNamespace(output_dir=tmp_path / "analyses" / "state", manifest_path=tmp_path / "manifest.json")
    campaign = object()
    match = SimpleNamespace(campaign=campaign, state_id="s004", measurement_kind=measurement_kind,
                            comparison_enabled=comparison_enabled, phase_offset_sensitivity=object())
    defaults = object()
    sensitivity, comparison, phase = object(), object(), object()
    calls = []
    monkeypatch.setattr(workflow, "resolve_tuning_marker_correction", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "run_folder_analysis", lambda **kwargs: analysis)
    monkeypatch.setattr(workflow, "register_matching_tuning_campaign", lambda *args, **kwargs: match)
    monkeypatch.setattr(workflow, "run_plunger_sensitivity", lambda *args, **kwargs: calls.append(("sensitivity", args, kwargs)) or sensitivity)
    monkeypatch.setattr(workflow, "run_tuning_cmp", lambda *args, **kwargs: calls.append(("comparison", args, kwargs)) or comparison)
    monkeypatch.setattr(workflow, "run_tuning_campaign_phase_shifts", lambda *args, **kwargs: calls.append(("phase", args, kwargs)) or phase)

    execution = workflow.run_dataset_analysis(
        dataset_id="raw_sweep_260722_tune_s004", sparameter_path=Path("raw/raw_sweep_260722_tune_s004"),
        output_dir=analysis.output_dir, marker_role="exp", data_root=tmp_path / "data",
        file_workers=3, plot_workers=2, tables_only=tables_only, project_defaults=defaults,
    )

    expected = [("sensitivity", (match,), dict(current_result=analysis, render_figure=not tables_only, project_defaults=defaults))]
    if comparison_enabled and not tables_only:
        expected.append(("comparison", (match,), dict(current_result=analysis, data_root=tmp_path / "data",
                                                     file_workers=3, plot_workers=2, project_defaults=defaults)))
    if measurement_kind == "state" and not tables_only:
        expected.append(("phase", (campaign,), dict(analysis_root=tmp_path / "analyses", output_dir=analysis.output_dir,
                                                   current_state_id="s004")))
    assert calls == expected
    assert execution.analysis is analysis
    assert execution.campaign_match is match
    assert execution.plunger_sensitivity is sensitivity
    assert execution.tuning_cmp is (comparison if comparison_enabled and not tables_only else None)
    assert execution.phase_shifts is (phase if measurement_kind == "state" and not tables_only else None)
