from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "run_folder_analysis.py"


def _load_runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("root_run_folder_analysis", RUNNER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_root_run_folder_analysis_cli_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run one folder through the standard deflector tuning analysis workflow." in result.stdout
    assert "input_folder" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--marker-role" in result.stdout
    assert "--file-workers" in result.stdout
    assert "--plot-workers" in result.stdout
    assert "--tables-only" in result.stdout
    assert "data/sim/sim_dispersion_260505_single_cell_step1" in result.stdout


def test_root_run_folder_analysis_cli_supports_import_without_running_analysis() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "from deflector_tuning.runner import run_folder_analysis" in source
    assert "if __name__ == \"__main__\":" in source


def test_collect_interactive_args_only_asks_for_input_folder(monkeypatch) -> None:
    module = _load_runner_module()
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "sim/sim_sweep_260527_iris_line"

    monkeypatch.setattr("builtins.input", fake_input)

    args = module.collect_interactive_args()

    assert prompts == ["Input dataset id or folder, e.g. sim_sweep_260527_iris_line: "]
    assert args.input_folder == Path("sim/sim_sweep_260527_iris_line")
    assert args.sparameter_path == Path("sim/sim_sweep_260527_iris_line")
    assert args.output_dir == Path("fig/analyses/sim_sweep_260527_iris_line")
    assert args.marker_role == "sim"
    assert args.dispersion_path is None
    assert args.data_root == Path("data")


def test_parse_args_infers_defaults_from_positional_input_folder() -> None:
    module = _load_runner_module()

    args = module.parse_args(["prepro/prepro_sweep_260415_sparams_fullbrazing"])

    assert args.input_folder == Path("prepro/prepro_sweep_260415_sparams_fullbrazing")
    assert args.sparameter_path == Path("prepro/prepro_sweep_260415_sparams_fullbrazing")
    assert args.output_dir == Path("fig/analyses/prepro_sweep_260415_sparams_fullbrazing")
    assert args.marker_role == "exp"
    assert args.data_root == Path("data")
    assert args.file_workers == 1
    assert args.plot_workers == 1
    assert args.tables_only is False


def test_parse_args_accepts_tables_only() -> None:
    module = _load_runner_module()

    args = module.parse_args(
        [
            "prepro/prepro_sweep_260415_sparams_fullbrazing",
            "--tables-only",
        ]
    )

    assert args.tables_only is True


def test_parse_args_accepts_plot_worker_count_and_normalizes_nonpositive_values() -> None:
    module = _load_runner_module()

    args = module.parse_args(
        [
            "prepro/prepro_sweep_260415_sparams_fullbrazing",
            "--plot-workers",
            "3",
        ]
    )
    assert args.plot_workers == 3

    normalized = module.parse_args(
        [
            "prepro/prepro_sweep_260415_sparams_fullbrazing",
            "--plot-workers",
            "0",
        ]
    )
    assert normalized.plot_workers == 1


def test_parse_args_resolves_dataset_id_by_searching_data_layers(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    (data_root / "sim" / "sim_sweep_260605_iris_2d_solver_export_nonorm").mkdir(parents=True)

    args = module.parse_args([
        "sim_sweep_260605_iris_2d_solver_export_nonorm",
        "--data-root",
        str(data_root),
    ])

    assert args.input_folder == Path("sim_sweep_260605_iris_2d_solver_export_nonorm")
    assert args.sparameter_path == Path("sim/sim_sweep_260605_iris_2d_solver_export_nonorm")
    assert args.output_dir == Path("fig/analyses/sim_sweep_260605_iris_2d_solver_export_nonorm")
    assert args.marker_role == "sim"


def test_parse_args_prefers_prepro_then_raw_then_sim_for_dataset_id(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    dataset_id = "prepro_sweep_260415_sparams_fullbrazing"
    for layer in ("sim", "raw", "prepro"):
        (data_root / layer / dataset_id).mkdir(parents=True)

    args = module.parse_args([dataset_id, "--data-root", str(data_root)])

    assert args.sparameter_path == Path("prepro") / dataset_id
    assert args.marker_role == "exp"


def test_default_output_dir_does_not_duplicate_existing_kind_prefix() -> None:
    module = _load_runner_module()

    assert module.default_output_dir("sim_sweep_260519_scan_dataset", marker_role="sim") == Path("fig/analyses/sim_sweep_260519_scan_dataset")
    assert module.default_output_dir("exp_measured_dataset", marker_role="exp") == Path("fig/analyses/exp_measured_dataset")


def test_parse_args_keeps_advanced_overrides_when_provided() -> None:
    module = _load_runner_module()

    args = module.parse_args(
        [
            "raw/raw_sweep_260604_iris_portE",
            "--output-dir",
            "custom/out",
            "--marker-role",
            "sim",
            "--data-root",
            "custom_data",
            "--file-workers",
            "3",
        ]
    )

    assert args.sparameter_path == Path("raw/raw_sweep_260604_iris_portE")
    assert args.output_dir == Path("custom/out")
    assert args.marker_role == "sim"
    assert args.data_root == Path("custom_data")
    assert args.file_workers == 3


def test_parse_args_loads_scientific_defaults_from_project_config(tmp_path: Path) -> None:
    module = _load_runner_module()
    config_path = tmp_path / "custom_defaults.toml"
    config_path.write_text(
        """
[analysis]
default_dispersion_subpath = "sim/custom_dispersion"

[visualization]
ideal_phase_guide_angles_deg = [15.0, 195.0]

[visualization.design_point_by_axis]
sim_r_c = 57.0
sim_w_c = 20.0
""".strip(),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "sim/sim_sweep_260527_iris_line",
            "--project-config",
            str(config_path),
        ]
    )

    assert args.project_config == config_path
    assert args.project_defaults.default_dispersion_subpath == Path("sim/custom_dispersion")
    assert args.project_defaults.design_point_by_axis == {
        "sim_r_c": 57.0,
        "sim_w_c": 20.0,
    }
    assert args.project_defaults.ideal_phase_guide_angles_deg == (15.0, 195.0)


def test_main_registers_matching_campaign_without_new_cli_arguments(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_runner_module()
    manifest_path = tmp_path / "analysis" / "manifest.json"
    args = SimpleNamespace(
        input_folder=Path("raw_sweep_260701_iris_tune_Torque13p5"),
        sparameter_path=Path("raw") / "raw_sweep_260701_iris_tune_Torque13p5",
        dispersion_path=None,
        output_dir=tmp_path / "analysis",
        marker_role="exp",
        data_root=tmp_path / "data",
        file_workers=1,
        plot_workers=1,
        tables_only=False,
        project_defaults=object(),
    )
    result = SimpleNamespace(
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        analysis_modes=("raw",),
        tables={},
        figures={},
    )
    registered: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    match = SimpleNamespace(
        comparison_enabled=True,
        phase_offset_sensitivity=None,
    )
    monkeypatch.setattr(module, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(module, "run_folder_analysis", lambda **kwargs: result)
    monkeypatch.setattr(
        module,
        "register_matching_tuning_campaign",
        lambda dataset_id, **kwargs: (
            registered.append({"dataset_id": dataset_id, **kwargs}) or match
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "run_tuning_cmp",
        lambda campaign_match, **kwargs: comparisons.append(
            {"campaign_match": campaign_match, **kwargs}
        ),
        raising=False,
    )

    assert module.main([]) == 0

    assert registered == [
        {
            "dataset_id": "raw_sweep_260701_iris_tune_Torque13p5",
            "manifest_path": manifest_path,
            "data_root": tmp_path / "data",
        }
    ]
    assert comparisons == [
        {
            "campaign_match": match,
            "current_result": result,
            "data_root": tmp_path / "data",
            "file_workers": 1,
            "plot_workers": 1,
            "project_defaults": args.project_defaults,
        }
    ]


def test_main_skips_tuning_comparison_for_auxiliary_measurement(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_runner_module()
    manifest_path = tmp_path / "analysis" / "manifest.json"
    args = SimpleNamespace(
        input_folder=Path("raw_sweep_260721_tune_s003_plungersensitivity"),
        sparameter_path=Path("raw") / "raw_sweep_260721_tune_s003_plungersensitivity",
        dispersion_path=None,
        output_dir=tmp_path / "analysis",
        marker_role="exp",
        data_root=tmp_path / "data",
        file_workers=1,
        plot_workers=1,
        tables_only=False,
        project_defaults=object(),
    )
    result = SimpleNamespace(
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        analysis_modes=("raw",),
        tables={},
        figures={},
    )
    comparisons: list[dict[str, object]] = []
    sensitivities: list[dict[str, object]] = []
    match = SimpleNamespace(
        comparison_enabled=False,
        phase_offset_sensitivity=object(),
    )
    monkeypatch.setattr(module, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(module, "run_folder_analysis", lambda **kwargs: result)
    monkeypatch.setattr(
        module,
        "register_matching_tuning_campaign",
        lambda dataset_id, **kwargs: match,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "run_tuning_cmp",
        lambda campaign_match, **kwargs: comparisons.append(
            {"campaign_match": campaign_match, **kwargs}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "run_plunger_sensitivity",
        lambda campaign_match, **kwargs: (
            sensitivities.append({"campaign_match": campaign_match, **kwargs})
            or SimpleNamespace(
                table_path=tmp_path / "phase_vs_plunger_offset.csv",
                figure_path=None,
            )
        ),
        raising=False,
    )

    assert module.main([]) == 0
    assert comparisons == []
    assert sensitivities == [
        {
            "campaign_match": match,
            "current_result": result,
            "render_figure": True,
            "project_defaults": args.project_defaults,
        }
    ]
