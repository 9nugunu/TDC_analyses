from dataclasses import replace

import pytest

import deflector_tuning.visualization.plot_config as plot_config
import deflector_tuning.workflows.marker_analysis as marker_workflow


def test_width_policy_uses_project_default_for_partial_mapping_and_explicit_override(monkeypatch):
    defaults = replace(plot_config.DEFAULT_PROJECT_DEFAULTS, design_point_by_axis={"sim_w_c": 21.5})
    monkeypatch.setattr(plot_config, "DEFAULT_PROJECT_DEFAULTS", defaults)
    partial = plot_config.PlotConfig(design_point_by_axis={"sim_r_c": 56.0})
    assert plot_config.resolve_line_scan_width(partial) == 21.5
    assert plot_config.resolve_line_scan_width(partial, fixed_w_c=22.0) == 22.0
    configured = plot_config.PlotConfig(design_point_by_axis={"sim_w_c": 20.0})
    assert plot_config.resolve_line_scan_width(configured) == 20.0
    assert plot_config.resolve_line_scan_width(configured, fixed_w_c=22.0) == 22.0


def test_marker_workflow_uses_shared_width_fallback(monkeypatch, tmp_path):
    defaults = replace(plot_config.DEFAULT_PROJECT_DEFAULTS, design_point_by_axis={"sim_w_c": 21.5})
    monkeypatch.setattr(plot_config, "DEFAULT_PROJECT_DEFAULTS", defaults)

    class AnalysisReached(Exception):
        pass

    def capture_analysis(**kwargs):
        assert kwargs["fixed_w_c"] == 21.5
        raise AnalysisReached

    monkeypatch.setattr(marker_workflow, "build_marker_analysis", capture_analysis)
    with pytest.raises(AnalysisReached):
        marker_workflow.run_marker_analysis(
            sparameter_path=tmp_path / "input", dispersion_path=tmp_path / "reference",
            output_dir=tmp_path / "out", marker_role="sim", dataset_category="grid",
            plot_config=plot_config.PlotConfig(design_point_by_axis={"sim_r_c": 56.0}),
            tables_only=True,
        )
