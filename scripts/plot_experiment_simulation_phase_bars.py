"""Compare measured and simulated S11 phases at two candidate radius states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.analysis.phase_radius_equivalence import (
    build_experiment_simulation_phase_comparison,
)
from deflector_tuning.visualization.phase_radius_equivalence_plots import (
    plot_phase_cmp_bars,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_phase_line_csv", type=Path)
    parser.add_argument("raw_phase_observation_csv", type=Path)
    parser.add_argument("--before-r-c-mm", required=True, type=float)
    parser.add_argument("--current-r-c-mm", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    comparison = build_experiment_simulation_phase_comparison(
        pd.read_csv(args.sim_phase_line_csv),
        pd.read_csv(args.raw_phase_observation_csv),
        before_r_c_mm=args.before_r_c_mm,
        current_r_c_mm=args.current_r_c_mm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "phase_cmp.csv", index=False)
    path = plot_phase_cmp_bars(
        comparison,
        args.output_dir / "phase_cmp_bars.png",
    )
    print(path)


if __name__ == "__main__":
    main()
