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


def test_root_run_folder_analysis_cli_supports_import_without_running_analysis() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "from deflector_tuning.runner import run_folder_analysis" in source
    assert "if __name__ == \"__main__\":" in source


def test_collect_interactive_args_only_asks_for_input_folder(monkeypatch) -> None:
    module = _load_runner_module()
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "sim/260527_iris_line_sweep"

    monkeypatch.setattr("builtins.input", fake_input)

    args = module.collect_interactive_args()

    assert prompts == ["Input dataset id or folder, e.g. 260527_iris_line_sweep: "]
    assert args.input_folder == Path("sim/260527_iris_line_sweep")
    assert args.sparameter_path == Path("sim/260527_iris_line_sweep")
    assert args.output_dir == Path("fig/analyses/260527_iris_line_sweep")
    assert args.marker_role == "sim"
    assert args.dispersion_path is None
    assert args.data_root == Path("data")


def test_parse_args_infers_defaults_from_positional_input_folder() -> None:
    module = _load_runner_module()

    args = module.parse_args(["prepro/260415_fullbrazing"])

    assert args.input_folder == Path("prepro/260415_fullbrazing")
    assert args.sparameter_path == Path("prepro/260415_fullbrazing")
    assert args.output_dir == Path("fig/analyses/260415_fullbrazing")
    assert args.marker_role == "exp"
    assert args.data_root == Path("data")


def test_parse_args_resolves_dataset_id_by_searching_data_layers(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    (data_root / "sim" / "260605_iris2dsweep_solver_export_no_norm").mkdir(parents=True)

    args = module.parse_args([
        "260605_iris2dsweep_solver_export_no_norm",
        "--data-root",
        str(data_root),
    ])

    assert args.input_folder == Path("260605_iris2dsweep_solver_export_no_norm")
    assert args.sparameter_path == Path("sim/260605_iris2dsweep_solver_export_no_norm")
    assert args.output_dir == Path("fig/analyses/260605_iris2dsweep_solver_export_no_norm")
    assert args.marker_role == "sim"


def test_parse_args_prefers_prepro_then_raw_then_sim_for_dataset_id(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    dataset_id = "260415_fullbrazing"
    for layer in ("sim", "raw", "prepro"):
        (data_root / layer / dataset_id).mkdir(parents=True)

    args = module.parse_args([dataset_id, "--data-root", str(data_root)])

    assert args.sparameter_path == Path("prepro") / dataset_id
    assert args.marker_role == "exp"


def test_parse_args_keeps_advanced_overrides_when_provided() -> None:
    module = _load_runner_module()

    args = module.parse_args(
        [
            "raw/260604_iris_portE",
            "--output-dir",
            "custom/out",
            "--marker-role",
            "sim",
            "--data-root",
            "custom_data",
        ]
    )

    assert args.sparameter_path == Path("raw/260604_iris_portE")
    assert args.output_dir == Path("custom/out")
    assert args.marker_role == "sim"
    assert args.data_root == Path("custom_data")
