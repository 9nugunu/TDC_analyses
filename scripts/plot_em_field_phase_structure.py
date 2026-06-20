"""Plot EM field phase traces with an inferred TDC half-section overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.em_field_structure_plots import (
    load_field_phase_export,
    plot_field_phase_with_tdc_structure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="CST field phase txt export")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "fig/analyses/<dataset>/figures/em_field_phase/field_phase_tdc_structure.png."
        ),
    )
    parser.add_argument(
        "--regular-cell-count",
        type=int,
        default=9,
        help="Number of regular cells between the two coupler cells.",
    )
    args = parser.parse_args()

    output = args.output or default_output_path(args.input_path)
    export = load_field_phase_export(args.input_path)
    path = plot_field_phase_with_tdc_structure(
        export,
        output,
        regular_cell_count=args.regular_cell_count,
    )
    print(path)


def default_output_path(input_path: Path) -> Path:
    """Return the canonical figure path for a data-layer field export."""

    parts = input_path.parts
    lower_parts = [part.lower() for part in parts]
    for index, part in enumerate(lower_parts):
        if part in {"sim", "raw", "prepro"} and index + 1 < len(parts):
            dataset_id = parts[index + 1]
            return Path("fig") / "analyses" / dataset_id / "figures" / "em_field_phase" / "field_phase_tdc_structure.png"
    return Path("fig") / "analyses" / input_path.stem / "figures" / "em_field_phase" / "field_phase_tdc_structure.png"


if __name__ == "__main__":
    main()
