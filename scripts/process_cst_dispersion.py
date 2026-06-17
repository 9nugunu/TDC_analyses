"""Process one CST dispersion txt export and optionally write a curve figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.dispersion import load_cst_dispersion_txt, process_cst_dispersion_txt
from deflector_tuning.visualization.dispersion_plots import plot_dispersion_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_txt", type=Path, help="CST dispersion text export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV output directory. Defaults to data/prepro/<dataset> for data-layer inputs.",
    )
    parser.add_argument("--figure-path", type=Path, default=None, help="Optional PNG output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = process_cst_dispersion_txt(args.input_txt, output_dir=args.output_dir)
    print(f"long_csv: {outputs.long_csv}")
    print(f"wide_csv: {outputs.wide_csv}")
    print(f"summary_csv: {outputs.summary_csv}")
    if args.figure_path is not None:
        table = load_cst_dispersion_txt(args.input_txt)
        figure_path = plot_dispersion_curves(table, args.figure_path)
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
