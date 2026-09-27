"""Render concise tuning-state guides on a simulated ``r_c`` phase scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.grid_scan_phase_line_plots import plot_phase_rc_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("line_scan_csv", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--before-r-c-mm", required=True, type=float)
    parser.add_argument("--current-r-c-mm", required=True, type=float)
    parser.add_argument("--s003-r-c-mm", required=True, type=float)
    parser.add_argument("--design-r-c-mm", required=True, type=float)
    parser.add_argument("--f-mean-zero-r-c-mm", required=True, type=float)
    parser.add_argument("--fixed-w-c-mm", type=float, default=19.3224)
    args = parser.parse_args()

    path = plot_phase_rc_states(
        pd.read_csv(args.line_scan_csv),
        args.output_path,
        before_r_c_mm=args.before_r_c_mm,
        current_r_c_mm=args.current_r_c_mm,
        s003_r_c_mm=args.s003_r_c_mm,
        design_r_c_mm=args.design_r_c_mm,
        f_mean_zero_r_c_mm=args.f_mean_zero_r_c_mm,
        fixed_w_c=args.fixed_w_c_mm,
    )
    print(path)


if __name__ == "__main__":
    main()
