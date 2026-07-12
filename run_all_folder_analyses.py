"""Command-line entrypoint for batch deflector tuning analysis across datasets."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path

from run_folder_analysis import (
    DATASET_ID_LAYER_PRIORITY,
    DEFAULT_DATA_ROOT,
    apply_inferred_defaults,
    prefixed_dataset_id,
    run_folder_analysis,
)
from deflector_tuning.data_loading.dataset_naming import parse_dataset_id
from deflector_tuning.dispersion import process_cst_dispersion_txt
from deflector_tuning.project_defaults import (
    DEFAULT_PROJECT_CONFIG_PATH,
    DEFAULT_PROJECT_DEFAULTS,
    ProjectDefaults,
    load_project_defaults,
)
from deflector_tuning.progress import progress_iter


DESCRIPTION = "Run all discovered datasets through the standard deflector tuning analysis workflow."
logger = logging.getLogger(__name__)


class BatchTask:
    """One dataset analysis task for the batch runner."""

    def __init__(
        self,
        *,
        sparameter_path: Path,
        dispersion_path: Path | None,
        output_dir: Path,
        marker_role: str,
        data_root: Path,
        project_defaults: ProjectDefaults = DEFAULT_PROJECT_DEFAULTS,
    ) -> None:
        self.sparameter_path = sparameter_path
        self.dispersion_path = dispersion_path
        self.output_dir = output_dir
        self.marker_role = marker_role
        self.data_root = data_root
        self.project_defaults = project_defaults


def default_worker_count() -> int:
    """Reserve two CPUs for the desktop and use the rest for dataset workers."""

    cpu_total = os.cpu_count() or 1
    return max(cpu_total - 2, 1)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the batch folder analysis runner."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        help="Optional dataset ids to restrict the batch. Default: discover all datasets under data/{prepro,raw,sim}.",
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=DEFAULT_PROJECT_CONFIG_PATH,
        help="Scientific defaults TOML. Default: config/project_defaults.toml.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Project data root used for discovering datasets. Default: data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("fig") / "analyses",
        help="Root output folder for dataset results. Default: fig/analyses.",
    )
    parser.add_argument(
        "--marker-role",
        choices=("sim", "exp"),
        default=None,
        help="Advanced override. Default is inferred from each dataset layer.",
    )
    parser.add_argument(
        "--dispersion-path",
        type=Path,
        default=None,
        help="Advanced override passed to every dataset run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of dataset-level worker processes. Default: max(1, os.cpu_count() - 2).",
    )
    return parser


def discover_dataset_inputs(*, data_root: Path) -> list[Path]:
    """Return one layer-qualified dataset path per dataset id using layer priority."""

    discovered: OrderedDict[str, Path] = OrderedDict()
    for layer in DATASET_ID_LAYER_PRIORITY:
        layer_dir = data_root / layer
        if not layer_dir.is_dir():
            continue
        for child in sorted(layer_dir.iterdir(), key=lambda path: path.name.lower()):
            if not child.is_dir() or child.name in discovered:
                continue
            discovered[child.name] = Path(layer) / child.name
    return list(discovered.values())


def resolve_batch_inputs(dataset_ids: list[str], *, data_root: Path) -> list[Path]:
    """Resolve explicit dataset ids or discover all datasets when none are given."""

    if dataset_ids:
        return [
            apply_inferred_defaults(
                argparse.Namespace(
                    input_folder=dataset_id,
                    output_dir=None,
                    marker_role=None,
                    dispersion_path=None,
                    data_root=data_root,
                )
            ).sparameter_path
            for dataset_id in dataset_ids
        ]
    return discover_dataset_inputs(data_root=data_root)


def build_batch_tasks(
    dataset_inputs: list[Path],
    *,
    output_root: Path,
    marker_role: str | None,
    dispersion_path: Path | None,
    data_root: Path,
    project_defaults: ProjectDefaults = DEFAULT_PROJECT_DEFAULTS,
) -> list[BatchTask]:
    """Resolve all per-dataset runtime arguments for the batch runner."""

    tasks: list[BatchTask] = []
    prepared_dispersion_path = prepare_batch_dispersion_input(
        dispersion_path,
        data_root=data_root,
        allow_missing_default=True,
    )
    for dataset_input in dataset_inputs:
        dataset_args = apply_inferred_defaults(
            argparse.Namespace(
                input_folder=dataset_input,
                output_dir=None,
                marker_role=marker_role,
                dispersion_path=dispersion_path,
                data_root=data_root,
            )
        )
        output_dir = output_root / prefixed_dataset_id(dataset_input.name, marker_role=str(dataset_args.marker_role))
        tasks.append(
            BatchTask(
                sparameter_path=Path(dataset_args.sparameter_path),
                dispersion_path=prepared_dispersion_path,
                output_dir=output_dir,
                marker_role=str(dataset_args.marker_role),
                data_root=Path(dataset_args.data_root),
                project_defaults=project_defaults,
            )
        )
    return tasks


def prepare_batch_dispersion_input(
    dispersion_path: Path | None,
    *,
    data_root: Path,
    allow_missing_default: bool = False,
) -> Path | None:
    """Return a dispersion folder with a summary CSV, processing one CST txt export when needed."""

    resolved_path = _resolve_dispersion_path(dispersion_path, data_root=data_root)
    if dispersion_path is None:
        return _prepare_discovered_dispersion_input(data_root=data_root, allow_missing=allow_missing_default)

    if resolved_path.is_file():
        if resolved_path.suffix.lower() != ".txt":
            raise ValueError(f"Dispersion input must be a CST .txt export or processed folder: {resolved_path}")
        logger.info("Processing CST dispersion export: %s", resolved_path)
        process_cst_dispersion_txt(resolved_path, output_dir=resolved_path.parent / "processed")
        return resolved_path.parent

    if not resolved_path.exists():
        raise FileNotFoundError(f"Dispersion path does not exist: {resolved_path}")
    if _has_dispersion_summary(resolved_path):
        return resolved_path

    txt_exports = _dispersion_txt_exports(resolved_path)
    if len(txt_exports) == 1:
        logger.info("Processing CST dispersion export found under %s: %s", resolved_path, txt_exports[0].name)
        process_cst_dispersion_txt(txt_exports[0], output_dir=resolved_path / "processed")
        return resolved_path
    if not txt_exports:
        raise FileNotFoundError(
            f"No dispersion summary CSV or CST txt export found under {resolved_path}. "
            "Run scripts/process_cst_dispersion.py first or pass --dispersion-path to a CST txt export."
        )
    names = ", ".join(path.name for path in txt_exports)
    raise ValueError(
        f"Multiple CST txt exports found under {resolved_path}: {names}. "
        "Pass --dispersion-path to the intended txt export or pre-process one export first."
    )


def _resolve_dispersion_path(dispersion_path: Path | None, *, data_root: Path) -> Path:
    if dispersion_path is None:
        return data_root
    path = Path(dispersion_path)
    if path.is_absolute() or path.parts[:1] == (data_root.name,):
        return path
    return data_root / path


def _has_dispersion_summary(folder: Path) -> bool:
    return any((folder / "processed").glob("*summary.csv")) or any(folder.glob("*summary.csv"))


def _prepare_discovered_dispersion_input(*, data_root: Path, allow_missing: bool) -> Path | None:
    candidates = _discover_dispersion_candidates(data_root)
    if not candidates:
        if allow_missing:
            return None
        raise FileNotFoundError(f"No dispersion-named CST input found under {data_root / 'sim'}")
    prepared: list[Path] = []
    for candidate in candidates:
        prepared_path = prepare_batch_dispersion_input(candidate, data_root=data_root)
        if prepared_path is not None:
            prepared.append(prepared_path)
    if len(prepared) == 1:
        return prepared[0]
    return None


def _discover_dispersion_candidates(data_root: Path) -> list[Path]:
    sim_root = data_root / "sim"
    if not sim_root.is_dir():
        return []
    candidates: list[Path] = []
    for child in sorted(sim_root.iterdir(), key=lambda path: path.name.lower()):
        if not _is_dispersion_dataset(child):
            continue
        if child.is_file() and child.suffix.lower() == ".txt":
            candidates.append(child)
        elif child.is_dir() and (_has_dispersion_summary(child) or _dispersion_txt_exports(child)):
            candidates.append(child)
    return candidates


def _is_dispersion_dataset(path: Path) -> bool:
    try:
        return parse_dataset_id(path.stem if path.is_file() else path.name).category == "dispersion"
    except ValueError:
        return False


def _dispersion_txt_exports(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.txt"))


def run_batch_task(task: BatchTask) -> tuple[Path, Path]:
    """Run one dataset task and return its input/output paths."""

    dispersion_path = (
        prepare_batch_dispersion_input(task.dispersion_path, data_root=task.data_root)
        if task.dispersion_path is not None
        else None
    )
    result = run_folder_analysis(
        sparameter_path=task.sparameter_path,
        dispersion_path=dispersion_path,
        output_dir=task.output_dir,
        marker_role=task.marker_role,
        data_root=task.data_root,
        plot_workers=1,
        project_defaults=task.project_defaults,
    )
    return task.sparameter_path, result.output_dir


def _log_batch_progress(*, completed: int, total: int, dataset_path: Path, success: bool) -> None:
    status = "completed" if success else "failed"
    logger.info("[%d/%d] %s: %s", completed, total, status, dataset_path)


def execute_batch_tasks(tasks: list[BatchTask], *, workers: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Exception]]]:
    """Execute dataset tasks sequentially or with multiprocessing."""

    total = len(tasks)
    successes: list[tuple[Path, Path]] = []
    failures: list[tuple[Path, Exception]] = []

    if workers <= 1:
        for index, task in enumerate(progress_iter(tasks, desc="Running datasets", total=total), start=1):
            logger.info("Running dataset %s -> %s", task.sparameter_path, task.output_dir)
            try:
                successes.append(run_batch_task(task))
                _log_batch_progress(completed=index, total=total, dataset_path=task.sparameter_path, success=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Dataset failed: %s", task.sparameter_path)
                failures.append((task.sparameter_path, exc))
                _log_batch_progress(completed=index, total=total, dataset_path=task.sparameter_path, success=False)
        return successes, failures

    logger.info("Starting multiprocessing batch execution with %d workers", workers)
    future_to_task = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for task in tasks:
            logger.info("Submitting dataset %s -> %s", task.sparameter_path, task.output_dir)
            future_to_task[executor.submit(run_batch_task, task)] = task
        completed = 0
        for future in progress_iter(as_completed(future_to_task), desc="Running datasets", total=total):
            task = future_to_task[future]
            completed += 1
            try:
                successes.append(future.result())
                _log_batch_progress(completed=completed, total=total, dataset_path=task.sparameter_path, success=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Dataset failed: %s", task.sparameter_path)
                failures.append((task.sparameter_path, exc))
                _log_batch_progress(completed=completed, total=total, dataset_path=task.sparameter_path, success=False)
    return successes, failures


def main(argv: list[str] | None = None) -> int:
    """Run one-folder analysis across every resolved dataset input."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    workers = default_worker_count() if int(args.workers) <= 0 else int(args.workers)
    dataset_inputs = resolve_batch_inputs(args.dataset_ids, data_root=Path(args.data_root))
    if not dataset_inputs:
        logger.warning("No datasets found under %s", args.data_root)
        return 0
    tasks = build_batch_tasks(
        dataset_inputs,
        output_root=Path(args.output_root),
        marker_role=args.marker_role,
        dispersion_path=args.dispersion_path,
        data_root=Path(args.data_root),
        project_defaults=load_project_defaults(args.project_config),
    )
    logger.info("Resolved %d dataset(s) for batch analysis with workers=%d", len(tasks), workers)
    successes, failures = execute_batch_tasks(tasks, workers=workers)
    for dataset_path, output_dir in successes:
        print(f"{dataset_path} -> {output_dir}")

    if failures:
        print("failed_datasets:")
        for dataset_path, exc in failures:
            print(f"  {dataset_path}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
