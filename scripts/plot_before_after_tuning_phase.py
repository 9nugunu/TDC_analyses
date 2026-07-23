"""Compare marker-point S-parameter phases before and after an iris tuning step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.tuning_phase_comparison_plots import (
    plot_before_after_sparameter_phase_position_scan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_marker_csv", type=Path, help="Pre-tuning marker_pts.csv")
    parser.add_argument("after_marker_csv", type=Path, help="Post-tuning marker_pts.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the PNG and comparison CSV")
    parser.add_argument("--before-label", default="Before tuning", help="Legend label for the pre-tuning trace")
    parser.add_argument("--after-label", default="After tuning", help="Legend label for the post-tuning trace")
    args = parser.parse_args()

    path = plot_before_after_sparameter_phase_position_scan(
        pd.read_csv(args.before_marker_csv),
        pd.read_csv(args.after_marker_csv),
        args.output_dir,
        before_label=args.before_label,
        after_label=args.after_label,
    )
    print(path)


if __name__ == "__main__":
    main()
