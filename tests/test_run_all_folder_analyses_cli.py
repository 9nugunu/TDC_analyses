from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "run_all_folder_analyses.py"


def _write_cst_export(path: Path) -> None:
    path.write_text(
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


def _load_runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("root_run_all_folder_analyses", RUNNER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_root_run_all_folder_analyses_cli_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run all discovered datasets through the standard deflector tuning" in result.stdout
    assert "dataset_ids" in result.stdout
    assert "--output-root" in result.stdout
    assert "--workers" in result.stdout
    assert "os.cpu_count() - 2" in result.stdout


def test_discover_dataset_inputs_uses_prepro_raw_sim_priority(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    for layer in ("sim", "raw", "prepro"):
        (data_root / layer / "shared_dataset").mkdir(parents=True, exist_ok=True)
    (data_root / "sim" / "sim_sweep_260519_only").mkdir(parents=True)
    (data_root / "raw" / "raw_sweep_260604_only").mkdir(parents=True)

    discovered = module.discover_dataset_inputs(data_root=data_root)

    assert discovered == [
        Path("prepro") / "shared_dataset",
        Path("raw") / "raw_sweep_260604_only",
        Path("sim") / "sim_sweep_260519_only",
    ]


def test_resolve_batch_inputs_uses_explicit_dataset_ids(tmp_path: Path) -> None:
    module = _load_runner_module()
    data_root = tmp_path / "data"
    (data_root / "sim" / "sim_sweep_260605_iris_2d_solver_export_nonorm").mkdir(parents=True)

    resolved = module.resolve_batch_inputs(["sim_sweep_260605_iris_2d_solver_export_nonorm"], data_root=data_root)

    assert resolved == [Path("sim") / "sim_sweep_260605_iris_2d_solver_export_nonorm"]


def test_build_batch_tasks_infers_output_and_marker_role(tmp_path: Path) -> None:
    module = _load_runner_module()
    tasks = module.build_batch_tasks(
        [Path("prepro") / "prepro_sweep_260415_good_dataset", Path("sim") / "sim_sweep_260519_scan_dataset"],
        output_root=tmp_path / "analyses",
        marker_role=None,
        dispersion_path=None,
        data_root=tmp_path / "data",
    )

    assert [task.output_dir for task in tasks] == [
        tmp_path / "analyses" / "prepro_sweep_260415_good_dataset",
        tmp_path / "analyses" / "sim_sweep_260519_scan_dataset",
    ]
    assert [task.marker_role for task in tasks] == ["exp", "sim"]


def test_main_passes_custom_project_defaults_to_every_batch_task(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_runner_module()
    config_path = tmp_path / "custom_defaults.toml"
    config_path.write_text(
        """
[analysis]
default_dispersion_subpath = "sim/custom_dispersion"

[visualization]
ideal_phase_guide_angles_deg = [30.0, 210.0]

[visualization.design_point_by_axis]
sim_r_c = 58.0
sim_w_c = 21.0
""".strip(),
        encoding="utf-8",
    )
    captured_tasks: list[object] = []
    monkeypatch.setattr(
        module,
        "resolve_batch_inputs",
        lambda dataset_ids, data_root: [
            Path("sim") / "sim_sweep_260519_scan_dataset"
        ],
    )

    def fake_execute_batch_tasks(tasks, workers):
        captured_tasks.extend(tasks)
        return ([(tasks[0].sparameter_path, tasks[0].output_dir)], [])

    monkeypatch.setattr(module, "execute_batch_tasks", fake_execute_batch_tasks)

    exit_code = module.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "analyses"),
            "--project-config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert len(captured_tasks) == 1
    defaults = captured_tasks[0].project_defaults
    assert defaults.default_dispersion_subpath == Path("sim/custom_dispersion")
    assert defaults.design_point_by_axis == {"sim_r_c": 58.0, "sim_w_c": 21.0}
    assert defaults.ideal_phase_guide_angles_deg == (30.0, 210.0)


def test_prepare_batch_dispersion_input_processes_explicit_txt(tmp_path: Path) -> None:
    module = _load_runner_module()
    source = tmp_path / "data" / "sim" / "sim_dispersion_260505_case" / "dispersion.txt"
    source.parent.mkdir(parents=True)
    _write_cst_export(source)

    prepared = module.prepare_batch_dispersion_input(
        Path("sim") / "sim_dispersion_260505_case" / "dispersion.txt",
        data_root=tmp_path / "data",
    )

    assert prepared == source.parent
    assert (source.parent / "processed" / "dispersion_summary.csv").exists()


def test_prepare_batch_dispersion_input_processes_single_txt_in_folder(tmp_path: Path) -> None:
    module = _load_runner_module()
    folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_case"
    folder.mkdir(parents=True)
    source = folder / "dispersion.txt"
    _write_cst_export(source)

    prepared = module.prepare_batch_dispersion_input(
        Path("sim") / "sim_dispersion_260505_case",
        data_root=tmp_path / "data",
    )

    assert prepared == folder
    assert (folder / "processed" / "dispersion_summary.csv").exists()


def test_prepare_batch_dispersion_input_discovers_dispersion_named_folder(tmp_path: Path) -> None:
    module = _load_runner_module()
    folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_my_new_export"
    folder.mkdir(parents=True)
    source = folder / "cst_dispersion_export.txt"
    _write_cst_export(source)

    prepared = module.prepare_batch_dispersion_input(None, data_root=tmp_path / "data")

    assert prepared == folder
    assert (folder / "processed" / "cst_dispersion_export_summary.csv").exists()


def test_prepare_batch_dispersion_input_rejects_ambiguous_txt_exports(tmp_path: Path) -> None:
    module = _load_runner_module()
    folder = tmp_path / "data" / "sim" / "sim_dispersion_260505_case"
    folder.mkdir(parents=True)
    _write_cst_export(folder / "a_dispersion.txt")
    _write_cst_export(folder / "b_dispersion.txt")

    try:
        module.prepare_batch_dispersion_input(Path("sim") / "sim_dispersion_260505_case", data_root=tmp_path / "data")
    except ValueError as exc:
        assert "Multiple CST txt exports" in str(exc)
    else:
        raise AssertionError("Expected ambiguous txt exports to fail")


def test_default_worker_count_reserves_two_cpus(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setattr(module.os, "cpu_count", lambda: 12)

    assert module.default_worker_count() == 10


def test_default_worker_count_never_drops_below_one(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setattr(module.os, "cpu_count", lambda: 2)

    assert module.default_worker_count() == 1


def test_main_runs_every_discovered_dataset_and_continues_after_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_runner_module()
    recorded_tasks: list[object] = []

    monkeypatch.setattr(
        module,
        "resolve_batch_inputs",
        lambda dataset_ids, data_root: [Path("prepro") / "prepro_sweep_260415_good_dataset", Path("sim") / "sim_sweep_260519_bad_dataset"],
    )
    monkeypatch.setattr(module, "default_worker_count", lambda: 1)

    def fake_execute_batch_tasks(tasks, workers):
        recorded_tasks.extend(tasks)
        assert workers == 1
        return (
            [(Path("prepro") / "prepro_sweep_260415_good_dataset", tmp_path / "analyses" / "prepro_sweep_260415_good_dataset")],
            [(Path("sim") / "sim_sweep_260519_bad_dataset", RuntimeError("bad dataset"))],
        )

    monkeypatch.setattr(module, "execute_batch_tasks", fake_execute_batch_tasks)

    exit_code = module.main(["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "analyses")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert len(recorded_tasks) == 2
    assert recorded_tasks[0].output_dir == tmp_path / "analyses" / "prepro_sweep_260415_good_dataset"
    assert recorded_tasks[0].marker_role == "exp"
    assert recorded_tasks[1].output_dir == tmp_path / "analyses" / "sim_sweep_260519_bad_dataset"
    assert "prepro\\prepro_sweep_260415_good_dataset ->" in captured.out or "prepro/prepro_sweep_260415_good_dataset ->" in captured.out
    assert "failed_datasets:" in captured.out


def test_execute_batch_tasks_logs_progress(monkeypatch, caplog) -> None:
    module = _load_runner_module()
    tasks = [
        module.BatchTask(
            sparameter_path=Path("prepro") / "prepro_sweep_260415_good_dataset",
            dispersion_path=None,
            output_dir=Path("fig") / "analyses" / "prepro_sweep_260415_good_dataset",
            marker_role="exp",
            data_root=Path("data"),
        ),
        module.BatchTask(
            sparameter_path=Path("sim") / "sim_sweep_260519_bad_dataset",
            dispersion_path=None,
            output_dir=Path("fig") / "analyses" / "sim_sweep_260519_bad_dataset",
            marker_role="sim",
            data_root=Path("data"),
        ),
    ]

    def fake_run_batch_task(task):
        if task.sparameter_path == Path("sim") / "sim_sweep_260519_bad_dataset":
            raise RuntimeError("bad dataset")
        return task.sparameter_path, task.output_dir

    monkeypatch.setattr(module, "run_batch_task", fake_run_batch_task)

    with caplog.at_level("INFO", logger=module.logger.name):
        successes, failures = module.execute_batch_tasks(tasks, workers=1)

    assert successes == [(Path("prepro") / "prepro_sweep_260415_good_dataset", Path("fig") / "analyses" / "prepro_sweep_260415_good_dataset")]
    assert len(failures) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("[1/2] completed: prepro" in message for message in messages)
    assert any("[2/2] failed: sim" in message for message in messages)


def test_main_passes_worker_count_to_batch_executor(monkeypatch, tmp_path: Path) -> None:
    module = _load_runner_module()
    recorded_workers: list[int] = []

    monkeypatch.setattr(module, "resolve_batch_inputs", lambda dataset_ids, data_root: [Path("sim") / "sim_sweep_260519_scan_dataset"])

    def fake_execute_batch_tasks(tasks, workers):
        recorded_workers.append(workers)
        return ([(Path("sim") / "sim_sweep_260519_scan_dataset", tmp_path / "analyses" / "sim_sweep_260519_scan_dataset")], [])

    monkeypatch.setattr(module, "execute_batch_tasks", fake_execute_batch_tasks)

    exit_code = module.main(["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "analyses"), "--workers", "4"])

    assert exit_code == 0
    assert recorded_workers == [4]


def test_main_uses_default_worker_count_when_nonpositive(monkeypatch, tmp_path: Path) -> None:
    module = _load_runner_module()
    recorded_workers: list[int] = []

    monkeypatch.setattr(module, "resolve_batch_inputs", lambda dataset_ids, data_root: [Path("sim") / "sim_sweep_260519_scan_dataset"])
    monkeypatch.setattr(module, "default_worker_count", lambda: 6)

    def fake_execute_batch_tasks(tasks, workers):
        recorded_workers.append(workers)
        return ([(Path("sim") / "sim_sweep_260519_scan_dataset", tmp_path / "analyses" / "sim_sweep_260519_scan_dataset")], [])

    monkeypatch.setattr(module, "execute_batch_tasks", fake_execute_batch_tasks)

    exit_code = module.main(["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "analyses"), "--workers", "0"])

    assert exit_code == 0
    assert recorded_workers == [6]
