"""PaySim raw schema and the shared point-in-time contract."""

from dataclasses import dataclass
import math

import pandas as pd


SOURCE_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"
DATASET_SLUG = "ealaxi/paysim1"
CONTRACT_VERSION = "1.0.0"
DTYPES = {
    "step": "int64",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}
COLUMNS = tuple(DTYPES)
BALANCE_COLUMNS = (
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
)
TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


@dataclass(frozen=True)
class SplitPolicy:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15

    def __post_init__(self):
        if not (0 < self.train_ratio < 1 and 0 < self.validation_ratio < 1
                and self.train_ratio + self.validation_ratio < 1):
            raise ValueError("Split ratios must be positive and leave a nonempty test ratio.")

    def partition(self, steps: list[int]) -> dict[str, list[int]]:
        ordered = sorted(set(steps))
        train_end = math.floor(len(ordered) * self.train_ratio)
        val_end = math.floor(len(ordered) * (self.train_ratio + self.validation_ratio))
        if not (0 < train_end < val_end < len(ordered)):
            raise ValueError("Not enough distinct steps for three nonempty temporal splits.")
        return {
            "train": ordered[:train_end],
            "validation": ordered[train_end:val_end],
            "test": ordered[val_end:],
        }


def validate_chunk(frame: pd.DataFrame) -> None:
    """Reject unusable raw inputs; balance nulls are audited but never model features."""
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"Expected original PaySim columns in order: {COLUMNS}")
    required = [column for column in COLUMNS if column not in BALANCE_COLUMNS]
    if frame[required].isna().any().any():
        counts = frame[required].isna().sum()
        raise ValueError(f"Missing required fields: {counts[counts > 0].to_dict()}")
    if (frame["step"] < 1).any():
        raise ValueError("step must be a positive integer.")
    if not frame["type"].isin(TRANSACTION_TYPES).all():
        raise ValueError("Unknown PaySim transaction type.")
    for column in ("nameOrig", "nameDest"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Empty account ID in {column}.")
    for column in ("isFraud", "isFlaggedFraud"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")
    for column, dtype in DTYPES.items():
        if dtype == "float64":
            values = frame[column].dropna()
            if values.abs().eq(float("inf")).any():
                raise ValueError(f"Non-finite value in {column}.")
    if (frame["amount"] < 0).any():
        raise ValueError("amount must be nonnegative.")


def baseline_features(frame: pd.DataFrame, *, include_step: bool = False) -> pd.DataFrame:
    """Allowlist prevents accidentally feeding labels, balances or raw IDs to E1."""
    columns = ["amount", "type"] + (["step"] if include_step else [])
    return frame.loc[:, columns].copy()


def history_before(frame: pd.DataFrame, prediction_step: int) -> pd.DataFrame:
    """Every transaction at t sees the same frozen state from steps strictly before t."""
    return frame.loc[frame["step"] < prediction_step].copy()

