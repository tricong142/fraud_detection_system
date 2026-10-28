"""Disk-backed staging repository: exact aggregates without loading the CSV at once."""

import logging
from pathlib import Path
import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contract import COLUMNS, DTYPES, validate_chunk


LOGGER = logging.getLogger(__name__)


def stage_csv(path: Path, connection: sqlite3.Connection, chunksize: int) -> dict:
    header = tuple(pd.read_csv(path, nrows=0).columns)
    if header != COLUMNS:
        raise ValueError(f"Expected original PaySim CSV columns {COLUMNS}; got {header}.")
    sql_types = {"string": "TEXT", "float64": "REAL", "int64": "INTEGER", "int8": "INTEGER"}
    definitions = ", ".join(f'"{name}" {sql_types[dtype]}' for name, dtype in DTYPES.items())
    connection.execute(f"CREATE TABLE transactions (row_id INTEGER PRIMARY KEY, {definitions})")
    placeholders = ",".join("?" for _ in range(len(COLUMNS) + 1))
    missing = {column: 0 for column in COLUMNS}
    rows = 0
    previous_step = None
    with pd.read_csv(path, chunksize=chunksize, dtype="string", keep_default_na=False) as reader:
        for chunk in reader:
            if chunk.empty:
                continue
            # Validate before narrowing labels: an int8 cast can silently wrap 256 into 0.
            for column, dtype in DTYPES.items():
                if dtype != "string":
                    values = pd.to_numeric(chunk[column].replace("", pd.NA), errors="raise")
                    if dtype.startswith("int"):
                        if values.isna().any() or (dtype == "int8" and not values.isin([0, 1]).all()):
                            raise ValueError(f"Invalid integer/label values in {column}.")
                        if dtype == "int64" and ((values % 1 != 0).any() or (values < 1).any()):
                            raise ValueError("step must be a positive integer.")
                    chunk[column] = values.astype(dtype)
            validate_chunk(chunk)
            if not chunk["step"].is_monotonic_increasing or (
                previous_step is not None and int(chunk["step"].iloc[0]) < previous_step
            ):
                raise ValueError("CSV must be ordered by step; same-step row order is not event time.")
            previous_step = int(chunk["step"].iloc[-1])
            for column, count in chunk.isna().sum().items():
                missing[column] += int(count)
            records = chunk.astype(object).where(pd.notna(chunk), None)
            connection.executemany(
                f"INSERT INTO transactions VALUES ({placeholders})",
                ((rows + offset, *record) for offset, record in enumerate(records.itertuples(index=False, name=None))),
            )
            connection.commit()
            rows += len(chunk)
            LOGGER.info("Staged %s rows", f"{rows:,}")
    if rows == 0:
        raise ValueError("CSV contains no transactions.")
    connection.execute("CREATE INDEX transactions_step ON transactions(step)")
    connection.commit()
    return {"rows": rows, "columns": len(COLUMNS), "dtypes": DTYPES, "missing": missing}


def build_accounts(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            outgoing_count INTEGER NOT NULL,
            incoming_count INTEGER NOT NULL,
            first_seen_step INTEGER NOT NULL,
            first_origin_step INTEGER,
            first_destination_step INTEGER
        ) WITHOUT ROWID
    """)
    connection.execute("""
        INSERT INTO accounts
        SELECT account_id, SUM(outgoing), SUM(incoming), MIN(step),
               MIN(CASE WHEN outgoing = 1 THEN step END),
               MIN(CASE WHEN incoming = 1 THEN step END)
        FROM (
            SELECT nameOrig AS account_id, step, 1 AS outgoing, 0 AS incoming FROM transactions
            UNION ALL
            SELECT nameDest AS account_id, step, 0 AS outgoing, 1 AS incoming FROM transactions
        ) GROUP BY account_id
    """)
    connection.commit()


def export_parquet(connection: sqlite3.Connection, query: str, path: Path,
                   chunksize: int, *, transaction_schema: bool = False,
                   params: tuple = ()) -> dict:
    rows = 0
    fraud = 0
    writer = None
    try:
        for frame in pd.read_sql_query(query, connection, params=params, chunksize=chunksize):
            if frame.empty:
                continue
            if transaction_schema:
                frame = frame.astype(DTYPES)
            else:
                frame["first_origin_step"] = frame["first_origin_step"].astype("Int64")
                frame["first_destination_step"] = frame["first_destination_step"].astype("Int64")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            if transaction_schema:
                fraud += int(frame["isFraud"].sum())
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "fraud": fraud, "nonfraud": rows - fraud} if transaction_schema else {"rows": rows}

