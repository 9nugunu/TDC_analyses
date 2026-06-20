"""Create a copy of a Touchstone folder with two-port files port-swapped."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deflector_tuning.data_loading.touchstone_ports import write_folder_with_s2p_ports_swapped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = write_folder_with_s2p_ports_swapped(args.source_folder, args.output_folder)
    print(f"wrote {len(written)} files to {args.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
