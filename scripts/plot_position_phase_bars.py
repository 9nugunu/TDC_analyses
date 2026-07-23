"""Plot wrapped S-parameter phase at selected tuning positions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deflector_tuning.visualization.position_phase_bar_plots import (
    build_position_phase_target_table,
    plot_position_phase_advance_bars,
    plot_position_phase_bars,
)


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the position-phase grouped-bar export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-points", required=True, type=Path)
    parser.add_argument("--positions", required=True, nargs="+", type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--advance-output", type=Path)
    args = parser.parse_args(argv)

    marker_points = pd.read_csv(args.marker_points)
    comparison = build_position_phase_target_table(
        marker_points,
        positions=args.positions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output.with_suffix(".csv"), index=False)
    path = plot_position_phase_bars(
        comparison,
        args.output,
        positions=args.positions,
    )
    advance_output = args.advance_output or args.output.with_name(
        f"{args.output.stem}_phase_advance{args.output.suffix}"
    )
    plot_position_phase_advance_bars(
        comparison,
        advance_output,
        positions=args.positions,
    )
    print(path)
    print(advance_output)
    return path


if __name__ == "__main__":
    main()
