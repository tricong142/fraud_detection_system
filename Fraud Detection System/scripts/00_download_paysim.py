"""Download original PaySim through Kaggle CLI; never handle credentials in this repo."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from fraud_detection.data_foundation.contract import DATASET_SLUG, SOURCE_URL
from fraud_detection.data_foundation.pipeline import find_csv, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("Download directory is not empty. Preserve existing raw data; use another directory.")
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "kaggle.cli", "datasets", "download",
               "-d", DATASET_SLUG, "-p", str(destination), "--unzip"]
    result = subprocess.run(command, check=False)
    if result.returncode:
        print("Kaggle download failed. Configure credentials outside the repository if required; "
              "do not paste tokens into logs/chat. No receipt was written.", file=sys.stderr)
        print("With Kaggle 1.7.4.5, anonymous download can fail with KeyError: 'username'. "
              "Use real Kaggle credentials; never insert placeholder credentials.", file=sys.stderr)
        return result.returncode
    csv_path = find_csv(destination)
    write_json(destination / "download_receipt.json", {
        "slug": DATASET_SLUG,
        "source_url": SOURCE_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "kaggle_cli_datasets_download",
        "raw_filename": csv_path.name,
        "raw_sha256": sha256_file(csv_path),
        "version": None,
    })
    print(f"Downloaded original CSV: {csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

