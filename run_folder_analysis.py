"""Command-line entrypoint for one-folder deflector tuning analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path, PurePath

from deflector_tuning.data_loading.source_layer import DataLayer
from deflector_tuning.runner import run_folder_analysis


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
        help="Advanced override. Default: fig/analyses/<dataset_id>.",
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
        help="Advanced override. Defaults to data/sim/260505_single_cell_dispersion_step1.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Project data root used for resolving relative input paths. Default: data.",
    )
    return parser


def collect_interactive_args() -> argparse.Namespace:
    """Collect the one user-facing runner setting from a simple prompt."""

    print(DESCRIPTION)
    input_folder = _prompt_required_path("Input dataset id or folder, e.g. 260527_iris_line_sweep")
    args = argparse.Namespace(
        input_folder=input_folder,
        output_dir=None,
        marker_role=None,
        dispersion_path=None,
        data_root=DEFAULT_DATA_ROOT,
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

    args.input_folder = input_folder
    args.sparameter_path = resolved_input
    args.output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir(dataset_id)
    args.marker_role = args.marker_role or marker_role_for_layer(data_layer)
    args.dispersion_path = Path(args.dispersion_path) if args.dispersion_path is not None else None
    args.data_root = data_root
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


def default_output_dir(dataset_id: str) -> Path:
    """Return the canonical default output directory for one-folder analysis."""

    return Path("fig") / "analyses" / dataset_id


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
