"""Command-line entrypoint for one-folder deflector tuning analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path, PurePath

from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.data_loading.dataset_naming import validate_dataset_id
from deflector_tuning.project_defaults import (
    DEFAULT_PROJECT_CONFIG_PATH,
    DEFAULT_PROJECT_DEFAULTS,
    load_project_defaults,
)
from deflector_tuning.runner import run_folder_analysis
from deflector_tuning.workflows.tuning_campaign import (
    register_matching_tuning_campaign,
)
from deflector_tuning.workflows.tuning_simulation_comparison import (
    run_tuning_cmp,
)
from deflector_tuning.workflows.tuning_phase_shifts import (
    run_tuning_campaign_phase_shifts,
)
from deflector_tuning.workflows.plunger_sensitivity import run_plunger_sensitivity


DESCRIPTION = "Run one folder through the standard deflector tuning analysis workflow."
DEFAULT_DATA_ROOT = Path("data")
DATASET_ID_LAYER_PRIORITY: tuple[str, ...] = ("prepro", "raw", "sim")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the folder analysis runner."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "input_folder",
        nargs="?",
        type=Path,
        help="Input dataset id or folder. Examples: <dataset>, sim/<dataset>, raw/<dataset>, or prepro/<dataset>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Advanced override. Default: fig/analyses/<sim|exp>_<dataset_id>.",
    )
    parser.add_argument(
        "--marker-role",
        choices=("sim", "exp"),
        default=None,
        help="Advanced override. Default is inferred from input folder: data/sim -> sim, data/raw|prepro -> exp.",
    )
    parser.add_argument(
        "--dispersion-path",
        type=Path,
        default=None,
        help=(
            "Advanced override. Default: data/"
            f"{DEFAULT_PROJECT_DEFAULTS.default_dispersion_subpath.as_posix()}."
        ),
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
        help="Project data root used for resolving relative input paths. Default: data.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of per-dataset Touchstone file loading workers. Default: 1.",
    )
    parser.add_argument(
        "--plot-workers",
        type=int,
        default=1,
        help="Number of independent S11 figure rendering workers. Default: 1.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Rebuild the 13 standard CSV tables without rendering figures.",
    )
    return parser


def collect_interactive_args() -> argparse.Namespace:
    """Collect the one user-facing runner setting from a simple prompt."""

    print(DESCRIPTION)
    input_folder = _prompt_required_path("Input dataset id or folder, e.g. sim_sweep_260527_iris_line")
    args = argparse.Namespace(
        input_folder=input_folder,
        output_dir=None,
        marker_role=None,
        dispersion_path=None,
        data_root=DEFAULT_DATA_ROOT,
        file_workers=1,
        plot_workers=1,
        tables_only=False,
    )
    return apply_inferred_defaults(args)


def _prompt_required_path(label: str) -> Path:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return Path(value)
        print("This value is required.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit CLI args, or prompt interactively when none are supplied."""

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return collect_interactive_args()

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_folder is None:
        parser.error("input_folder is required unless running interactive mode")
    return apply_inferred_defaults(args)


def apply_inferred_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Infer runner settings that should not be user-facing by default."""

    input_folder = Path(args.input_folder)
    data_root = Path(args.data_root)
    resolved_input = resolve_input_folder(input_folder, data_root=data_root)
    data_layer = _detect_data_layer_from_input(resolved_input)
    dataset_id = _dataset_id_from_input(resolved_input)
    validate_dataset_id(dataset_id, data_layer)

    args.input_folder = input_folder
    args.sparameter_path = resolved_input
    args.marker_role = args.marker_role or marker_role_for_layer(data_layer)
    args.output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(dataset_id, marker_role=args.marker_role)
    )
    args.dispersion_path = Path(args.dispersion_path) if args.dispersion_path is not None else None
    args.data_root = data_root
    args.file_workers = max(int(getattr(args, "file_workers", 1)), 1)
    args.plot_workers = max(int(getattr(args, "plot_workers", 1)), 1)
    args.tables_only = bool(getattr(args, "tables_only", False))
    args.project_config = Path(getattr(args, "project_config", DEFAULT_PROJECT_CONFIG_PATH))
    args.project_defaults = load_project_defaults(args.project_config)
    return args


def resolve_input_folder(input_folder: Path, *, data_root: Path) -> Path:
    """Resolve either a layer-qualified folder or a bare dataset id."""

    if _has_data_layer(input_folder) or input_folder.is_absolute() or _starts_with_data_root(input_folder, data_root):
        return _strip_data_root_prefix(input_folder, data_root=data_root)

    matches = [layer for layer in DATASET_ID_LAYER_PRIORITY if (data_root / layer / input_folder).is_dir()]
    if not matches:
        searched = ", ".join(str(data_root / layer / input_folder) for layer in DATASET_ID_LAYER_PRIORITY)
        raise FileNotFoundError(f"Could not find dataset id {input_folder!s}; searched: {searched}")
    return Path(matches[0]) / input_folder


def _data_relative_input(input_folder: Path, *, data_root: Path) -> Path:
    if input_folder.is_absolute() or _starts_with_data_root(input_folder, data_root):
        return input_folder
    return data_root / input_folder


def _strip_data_root_prefix(path: Path, *, data_root: Path) -> Path:
    parts = PurePath(path).parts
    if parts and parts[0].lower() == data_root.name.lower() and len(parts) >= 3:
        return Path(*parts[1:])
    return path


def _has_data_layer(path: Path) -> bool:
    return any(part.lower() in {"sim", "raw", "prepro"} for part in PurePath(path).parts)


def _starts_with_data_root(path: Path, data_root: Path) -> bool:
    parts = PurePath(path).parts
    if not parts:
        return False
    return parts[0].lower() == data_root.name.lower()


def _dataset_id_from_input(path: Path) -> str:
    parts = PurePath(path).parts
    layer_index = _data_layer_index(parts)
    if layer_index + 1 >= len(parts):
        raise ValueError(f"Expected dataset folder after sim, raw, or prepro; got {path!s}")
    return parts[layer_index + 1]


def _detect_data_layer_from_input(path: Path) -> DataLayer:
    parts = PurePath(path).parts
    layer_name = parts[_data_layer_index(parts)].lower()
    if layer_name == "sim":
        return DataLayer.SIM
    if layer_name == "raw":
        return DataLayer.RAW
    if layer_name == "prepro":
        return DataLayer.PREPRO
    raise ValueError(f"Expected input under sim, raw, or prepro; got {path!s}")


def _data_layer_index(parts: tuple[str, ...]) -> int:
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts):
        if part in {"sim", "raw", "prepro"}:
            return index
    raise ValueError("Expected input folder like sim/<dataset>, raw/<dataset>, or prepro/<dataset>")


def marker_role_for_layer(data_layer: DataLayer) -> str:
    """Map data-folder identity to marker-frequency role."""

    if data_layer is DataLayer.SIM:
        return "sim"
    return "exp"


def prefixed_dataset_id(dataset_id: str, *, marker_role: str) -> str:
    """Return the output dataset id with a sim/exp role prefix."""

    prefix = f"{marker_role}_"
    if dataset_id.startswith(("sim_", "raw_", "prepro_", "exp_")):
        return dataset_id
    return f"{prefix}{dataset_id}"


def default_output_dir(dataset_id: str, *, marker_role: str) -> Path:
    """Return the canonical default output directory for one-folder analysis."""

    return Path("fig") / "analyses" / prefixed_dataset_id(dataset_id, marker_role=marker_role)


def main(argv: list[str] | None = None) -> int:
    """Parse inputs, run analysis, and print generated paths."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    result = run_folder_analysis(
        sparameter_path=args.sparameter_path,
        dispersion_path=args.dispersion_path,
        output_dir=args.output_dir,
        marker_role=args.marker_role,
        data_root=args.data_root,
        file_workers=args.file_workers,
        plot_workers=args.plot_workers,
        tables_only=args.tables_only,
        project_defaults=args.project_defaults,
    )
    campaign_match = register_matching_tuning_campaign(
        _dataset_id_from_input(Path(args.sparameter_path)),
        manifest_path=result.manifest_path,
        data_root=args.data_root,
    )
    tuning_cmp = None
    if (
        campaign_match is not None
        and campaign_match.phase_offset_sensitivity is not None
    ):
        plunger_sensitivity = run_plunger_sensitivity(
            campaign_match,
            current_result=result,
            render_figure=not args.tables_only,
            project_defaults=args.project_defaults,
        )
    else:
        plunger_sensitivity = None
    if (
        campaign_match is not None
        and campaign_match.comparison_enabled
        and not args.tables_only
    ):
        tuning_cmp = run_tuning_cmp(
            campaign_match,
            current_result=result,
            data_root=args.data_root,
            file_workers=args.file_workers,
            plot_workers=args.plot_workers,
            project_defaults=args.project_defaults,
        )
    phase_shifts = None
    if (
        campaign_match is not None
        and getattr(campaign_match, "measurement_kind", None) == "state"
        and not args.tables_only
    ):
        phase_shifts = run_tuning_campaign_phase_shifts(
            campaign_match.campaign,
            analysis_root=Path(result.output_dir).parent,
            output_dir=result.output_dir,
            current_state_id=getattr(campaign_match, "state_id", None),
        )

    print(f"input_folder: {args.input_folder}")
    print(f"marker_role: {args.marker_role}")
    print(f"output_dir: {result.output_dir}")
    print(f"manifest: {result.manifest_path}")
    print("analysis_modes: " + ", ".join(result.analysis_modes))
    print("tables:")
    for name, path in result.tables.items():
        print(f"  {name}: {path}")
    print("figures:")
    for group, paths in result.figures.items():
        print(f"  {group}:")
        for name, path in paths.items():
            print(f"    {name}: {path}")
    if tuning_cmp is not None:
        print("tuning_cmp:")
        print("  families: " + ", ".join(tuning_cmp.families))
        for name, path in tuning_cmp.figures.items():
            print(f"  {name}: {path}")
    if plunger_sensitivity is not None:
        print("plunger_sensitivity:")
        print(f"  table: {plunger_sensitivity.table_path}")
        if plunger_sensitivity.figure_path is not None:
            print(f"  figure: {plunger_sensitivity.figure_path}")
    if phase_shifts is not None:
        print("tuning_phase_shifts:")
        print(f"  table: {phase_shifts.table_path}")
        print(f"  combined: {phase_shifts.combined_figure_path}")
        print(f"  iris: {phase_shifts.iris_figure_path}")
        print(f"  cell: {phase_shifts.cell_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
