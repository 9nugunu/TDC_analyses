from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
