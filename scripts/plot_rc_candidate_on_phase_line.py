"""Add one candidate-radius guide to an existing S-parameter phase line scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.grid_scan_phase_line_plots import (
    plot_phase_rc_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("line_scan_csv", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--candidate-r-c-mm", required=True, type=float)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--before-r-c-mm", type=float)
    parser.add_argument("--before-label")
    parser.add_argument("--target-r-c-mm", type=float)
    parser.add_argument("--target-label")
    parser.add_argument("--fixed-w-c-mm", type=float, default=19.3224)
    args = parser.parse_args()

    rc_fit = pd.DataFrame(
        [
            {"state": "baseline", "r_c_mm": args.before_r_c_mm},
            {"state": "current", "r_c_mm": args.candidate_r_c_mm},
            {"state": "design", "r_c_mm": args.target_r_c_mm},
        ]
    )
    if rc_fit["r_c_mm"].isna().any():
        parser.error("--before-r-c-mm and --target-r-c-mm are required")
    path = plot_phase_rc_map(
        pd.read_csv(args.line_scan_csv),
        rc_fit,
        args.output_path,
        fixed_w_c=args.fixed_w_c_mm,
    )
    print(path)


if __name__ == "__main__":
    main()
