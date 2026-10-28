"""Thin CLI; run from the project root after installing the package."""

import argparse
import logging
from pathlib import Path
import sys

from fraud_detection.data_foundation import run_foundation
from fraud_detection.data_foundation.contract import SplitPolicy
from fraud_detection.data_foundation.pipeline import find_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit original PaySim CSV and freeze temporal splits.")
    parser.add_argument("--csv", type=Path, help="Explicit original CSV; otherwise discover exactly one file.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_foundation"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--split-reason", help="Required when changing ratios; decided before evaluating models.")
    parser.add_argument("--fixture", action="store_true", help="Mark all outputs as test fixture, not real PaySim.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        manifest = run_foundation(args.csv or find_csv(args.raw_dir), args.output_dir,
            chunksize=args.chunksize, policy=SplitPolicy(args.train_ratio, args.validation_ratio),
            fixture=args.fixture, split_reason=args.split_reason)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        logging.error("%s", error)
        return 1
    print(f"Rows audited: {manifest['raw']['rows']:,}; raw SHA-256: {manifest['raw']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

