"""Combine verified CST-Run-Analyze batch exports into one navigator-indexed sim dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deflector_tuning.data_loading.cst_batch_delivery import assemble_delivery

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Delivery config, e.g. config/delivery_261009_R07_extended_recomputed.json")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite files in an existing delivery folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = assemble_delivery(REPO_ROOT, config, config_path.relative_to(REPO_ROOT).as_posix(),
                               overwrite=args.overwrite)
    for row in result["rows"]:
        LOGGER.info("run %s  %s=%s  <- %s/%s (%s)", row["run_id"], config["varying_parameter"],
                    row[config["varying_parameter"]], row["source_batch"], row["source_run_label"],
                    row["port_transform"])
    LOGGER.info("wrote %d files to %s", result["file_count"], result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
