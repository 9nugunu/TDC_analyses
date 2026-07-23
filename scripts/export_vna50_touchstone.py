"""Export one-port CST Z11 files as VNA-reference S1P files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deflector_tuning.data_loading.vna50 import export_vna50_touchstone_dataset


def main() -> int:
    """Run the path-configurable VNA-reference export."""

    parser = argparse.ArgumentParser(
        description="Convert a one-port Z Touchstone dataset to reference-impedance S1P.",
    )
    parser.add_argument("source", type=Path, help="Source data/sim Z1P dataset")
    parser.add_argument("output", type=Path, help="Derived data/sim S1P dataset")
    parser.add_argument(
        "--reference-ohm",
        type=float,
        default=50.0,
        help="Real VNA reference impedance in ohms (default: 50)",
    )
    args = parser.parse_args()
    result = export_vna50_touchstone_dataset(
        args.source,
        args.output,
        reference_ohm=args.reference_ohm,
    )
    print(result.dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
