"""Application service orchestrating the offline foundation use case."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import pyarrow

from .audit import collect_audit, write_report
from .contract import CONTRACT_VERSION, SOURCE_URL, SplitPolicy
from .storage import build_accounts, export_parquet, stage_csv


LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def find_csv(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one CSV in {raw_dir}; found {len(files)}. Pass --csv explicitly.")
    return files[0]


def run_foundation(csv_path: Path, output_dir: Path, *, chunksize: int = 100_000,
                   policy: SplitPolicy | None = None, fixture: bool = False,
                   split_reason: str | None = None) -> dict:
    csv_path = Path(csv_path).resolve()
    output_dir = Path(output_dir).resolve()
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; use a new directory to preserve frozen splits: {output_dir}")
    policy = policy or SplitPolicy()
    if policy != SplitPolicy() and not split_reason:
        raise ValueError("Changing temporal ratios requires --split-reason recorded before model evaluation.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    raw_checksum = sha256_file(csv_path)
    working = Path(tempfile.mkdtemp(prefix=".foundation-", dir=output_dir.parent))
    try:
        with tempfile.TemporaryDirectory(prefix=".audit-state-", dir=output_dir.parent) as temporary:
            connection = sqlite3.connect(Path(temporary) / "audit.sqlite")
            try:
                connection.execute("PRAGMA cache_size = -32768")
                connection.execute("PRAGMA temp_store = FILE")
                basic = stage_csv(csv_path, connection, chunksize)
                LOGGER.info("Building account history audit (disk-backed)")
                build_accounts(connection)
                LOGGER.info("Computing global duplicates, temporal and graph summaries")
                audit, by_step = collect_audit(connection, working, basic)
                partitions = policy.partition(by_step["step"].tolist())
                splits = {}
                (working / "splits").mkdir()
                for name, steps in partitions.items():
                    selected = by_step[by_step["step"].isin(steps)]
                    fraud = int(selected["fraud"].sum())
                    count = int(selected["transactions"].sum())
                    if fraud == 0 or fraud == count:
                        raise ValueError(f"{name} has fraud={fraud}, nonfraud={count - fraud}; "
                                         "each split needs both labels. Review by-step distribution "
                                         "and record a split decision; do not silently move test boundaries.")
                    path = working / "splits" / f"{name}.parquet"
                    counts = export_parquet(connection,
                        "SELECT * FROM transactions WHERE step BETWEEN ? AND ? ORDER BY row_id",
                        path, chunksize, transaction_schema=True, params=(steps[0], steps[-1]))
                    if counts["rows"] != count or counts["fraud"] != fraud:
                        raise RuntimeError("Split export did not match audit counts.")
                    splits[name] = {"path": f"splits/{name}.parquet", "steps": steps,
                                    "step_min": steps[0], "step_max": steps[-1],
                                    **counts, "sha256": sha256_file(path)}
                export_parquet(connection, "SELECT * FROM accounts ORDER BY account_id",
                               working / "account_activity.parquet", chunksize)
            finally:
                connection.close()
        if sha256_file(csv_path) != raw_checksum:
            raise RuntimeError("Raw CSV changed during the run; refusing inconsistent outputs.")
        if sum(info["rows"] for info in splits.values()) != audit["rows"]:
            raise RuntimeError("Temporal splits do not cover the full dataset.")
        split_definition = {
            "contract_version": CONTRACT_VERSION,
            "raw_sha256": raw_checksum,
            "method": "consecutive_observed_steps_floor_cumulative_ratio",
            "train_ratio": policy.train_ratio,
            "validation_ratio": policy.validation_ratio,
            "test_ratio": round(1 - policy.train_ratio - policy.validation_ratio, 10),
            "decision": split_reason or "Initial 70/15/15 observed-step policy; no automatic label-based adjustment.",
            "history_rule": "step < prediction_step",
            "shared_by_experiments": ["E1", "E2", "E3"],
            "splits": splits,
        }
        write_json(working / "split_definition.json", split_definition)
        write_json(working / "audit_summary.json", audit)
        write_report(working, audit, splits, fixture=fixture)
        shutil.copyfile(Path(__file__).with_name("feature_policy.md"), working / "feature_policy.md")
        artifacts = {
            str(path.relative_to(working)).replace("\\", "/"): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(working.rglob("*")) if path.is_file()
        }
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_kind": "synthetic_test_fixture" if fixture else "user_supplied_raw_csv",
            "source": {"declared_url": SOURCE_URL, "slug": "ealaxi/paysim1",
                       "version": None, "provenance_verified": False,
                       "note": "Declared source only. SHA-256 identifies local bytes, not source authenticity."},
            "raw": {"filename": csv_path.name, "sha256": raw_checksum,
                    "size_bytes": csv_path.stat().st_size, "rows": audit["rows"], "columns": audit["columns"]},
            "runtime": {"pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                        "sqlite": sqlite3.sqlite_version},
            "artifacts": artifacts,
        }
        receipt_path = csv_path.parent / "download_receipt.json"
        if receipt_path.is_file() and not fixture:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("slug") == "ealaxi/paysim1"
                    and receipt.get("raw_sha256") == raw_checksum):
                manifest["data_kind"] = "kaggle_cli_download"
                manifest["source"]["download_receipt"] = receipt
                manifest["source"]["note"] = (
                    "Local receipt records successful Kaggle CLI download and matching raw bytes. "
                    "Upstream version and independent authenticity were not verified."
                )
                shutil.copyfile(receipt_path, working / "download_receipt.json")
                manifest["artifacts"]["download_receipt.json"] = {
                    "sha256": sha256_file(working / "download_receipt.json"),
                    "size_bytes": (working / "download_receipt.json").stat().st_size,
                }
        write_json(working / "dataset_manifest.json", manifest)
        working.rename(output_dir)
        LOGGER.info("Published data foundation: %s", output_dir)
        return manifest
    finally:
        if working.exists():
            shutil.rmtree(working)

