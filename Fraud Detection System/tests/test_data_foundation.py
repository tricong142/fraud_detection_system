"""Regression checks on synthetic data, never presented as measurements of PaySim."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fraud_detection.data_foundation import run_foundation
from fraud_detection.data_foundation.contract import (
    COLUMNS, SplitPolicy, baseline_features, history_before,
)
from fraud_detection.data_foundation.pipeline import find_csv, sha256_file


def sample_frame() -> pd.DataFrame:
    rows = []
    for step in range(1, 21):
        rows.extend([
            [step, "PAYMENT", 10.0, "Cshared", 100.0, 90.0, "Mshared", 0.0, 10.0, 0, 0],
            [step, "TRANSFER", 50.0, f"C{step}", 50.0, 0.0, "Cshared", 0.0, 50.0, 1, 0],
        ])
    rows.insert(2, rows[0].copy())  # Duplicate spans a chunk boundary with chunksize=2.
    return pd.DataFrame(rows, columns=COLUMNS)


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / "synthetic.csv"
        self.frame = sample_frame()
        self.frame.to_csv(self.csv, index=False)

    def run_pipeline(self, name="result", **kwargs):
        return run_foundation(self.csv, self.root / name, chunksize=2, fixture=True, **kwargs)

    def test_end_to_end_and_pandas_reference(self):
        manifest = self.run_pipeline()
        output = self.root / "result"
        audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["rows"], len(self.frame))
        self.assertEqual(audit["duplicates_excluding_first"], int(self.frame.duplicated().sum()))
        self.assertEqual(audit["fraud"], int(self.frame.isFraud.sum()))
        self.assertEqual(audit["unique_origin_accounts"], self.frame.nameOrig.nunique())
        self.assertEqual(audit["unique_destination_accounts"], self.frame.nameDest.nunique())
        accounts = set(self.frame.nameOrig) | set(self.frame.nameDest)
        self.assertEqual(audit["unique_accounts_any_role"], len(accounts))
        self.assertEqual(audit["unique_directed_edges"], len(self.frame[["nameOrig", "nameDest"]].drop_duplicates()))
        actual = pd.read_csv(output / "by_step_type.csv")
        expected = self.frame.groupby(["step", "type"]).isFraud.agg(transactions="count", fraud="sum").reset_index()
        pd.testing.assert_frame_equal(actual[expected.columns], expected)
        definition = json.loads((output / "split_definition.json").read_text())
        self.assertEqual(definition["splits"]["train"]["steps"], list(range(1, 15)))
        self.assertEqual(definition["splits"]["validation"]["steps"], [15, 16, 17])
        self.assertEqual(definition["splits"]["test"]["steps"], [18, 19, 20])
        frames = [pd.read_parquet(output / details["path"]) for details in definition["splits"].values()]
        joined = pd.concat(frames, ignore_index=True)
        self.assertEqual(joined.row_id.tolist(), list(range(len(self.frame))))
        pd.testing.assert_frame_equal(joined[list(COLUMNS)], self.frame.astype(joined[list(COLUMNS)].dtypes.to_dict()))
        self.assertLess(frames[0].step.max(), frames[1].step.min())
        self.assertLess(frames[1].step.max(), frames[2].step.min())
        for name, metadata in manifest["artifacts"].items():
            self.assertEqual(sha256_file(output / name), metadata["sha256"])
        self.assertEqual(manifest["raw"]["sha256"], sha256_file(self.csv))
        self.assertEqual(manifest["data_kind"], "synthetic_test_fixture")
        self.assertIn("NOT PAYSIM", (output / "data_audit.md").read_text(encoding="utf-8"))

    def test_account_roles_and_same_step_first_seen(self):
        self.run_pipeline()
        output = self.root / "result"
        accounts = pd.read_parquet(output / "account_activity.parquet").set_index("account_id")
        self.assertEqual(accounts.loc["Cshared", "outgoing_count"], 21)
        self.assertEqual(accounts.loc["Cshared", "incoming_count"], 20)
        by_step = pd.read_csv(output / "by_step.csv").set_index("step")
        self.assertEqual(by_step.loc[1, "new_accounts"], 3)
        self.assertEqual(by_step.loc[1, "new_origin_accounts"], 2)
        self.assertEqual(by_step.loc[1, "new_destination_accounts"], 2)
        self.assertEqual(by_step.loc[2, "prior_seen_accounts"], 3)
        self.assertEqual(int(by_step.new_accounts.sum()), len(accounts))

    def test_strict_history_and_feature_allowlist(self):
        history = history_before(self.frame, 10)
        self.assertTrue((history.step < 10).all())
        self.assertNotIn(10, history.step.tolist())
        self.assertEqual(len(history_before(self.frame, 1)), 0)
        same_step = self.frame[self.frame.step == 10]
        for _ in same_step.itertuples():
            pd.testing.assert_frame_equal(history_before(self.frame, 10), history)
        self.assertEqual(list(baseline_features(self.frame).columns), ["amount", "type"])
        self.assertEqual(list(baseline_features(self.frame, include_step=True).columns), ["amount", "type", "step"])

    def test_reject_label_overflow_before_int8_cast(self):
        for label in (2, 256, -1, 1.5):
            with self.subTest(label=label):
                frame = self.frame.copy()
                frame["isFraud"] = frame["isFraud"].astype("float64")
                frame.loc[0, "isFraud"] = label
                frame.to_csv(self.csv, index=False)
                with self.assertRaises(ValueError):
                    self.run_pipeline()
                self.assertFalse((self.root / "result").exists())

    def test_missing_balance_is_audited_and_preserved(self):
        self.frame.loc[0, "oldbalanceOrg"] = float("nan")
        self.frame.to_csv(self.csv, index=False)
        self.run_pipeline()
        audit = json.loads((self.root / "result" / "audit_summary.json").read_text())
        self.assertEqual(audit["missing"]["oldbalanceOrg"], 1)
        train = pd.read_parquet(self.root / "result" / "splits" / "train.parquet")
        self.assertTrue(pd.isna(train.loc[0, "oldbalanceOrg"]))

    def test_reject_single_class_split_and_do_not_publish(self):
        self.frame.loc[self.frame.step >= 18, "isFraud"] = 0
        self.frame.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "test has fraud=0"):
            self.run_pipeline()
        self.assertFalse((self.root / "result").exists())
        self.assertFalse(list(self.root.glob(".foundation-*")))

    def test_reject_unsorted_across_chunks(self):
        self.frame.iloc[::-1].to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "ordered by step"):
            self.run_pipeline()

    def test_frozen_output_not_overwritten(self):
        self.run_pipeline()
        checksum = sha256_file(self.root / "result" / "dataset_manifest.json")
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.assertEqual(sha256_file(self.root / "result" / "dataset_manifest.json"), checksum)

    def test_bad_schema_and_ambiguous_csv(self):
        self.frame.assign(engineered_feature=1).to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "original PaySim"):
            self.run_pipeline()
        (self.root / "second.csv").touch()
        with self.assertRaisesRegex(ValueError, "exactly one CSV"):
            find_csv(self.root)

    def test_empty_dataset_and_invalid_inputs(self):
        self.frame.iloc[:0].to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "no transactions"):
            self.run_pipeline()
        for column, value in (("amount", -1), ("amount", float("inf")),
                              ("step", 1.5), ("nameOrig", ""), ("type", "UNKNOWN")):
            with self.subTest(column=column, value=value):
                frame = self.frame.copy()
                if column == "step":
                    frame[column] = frame[column].astype("float64")
                frame.loc[0, column] = value
                frame.to_csv(self.csv, index=False)
                with self.assertRaises(ValueError):
                    self.run_pipeline()

    def test_split_ratios_gaps_and_decision(self):
        for ratios in ((0, .15), (.9, .2), (float("nan"), .15)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                SplitPolicy(*ratios)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            SplitPolicy().partition([1, 2, 3])
        steps = list(range(1, 40, 2))
        self.assertEqual(SplitPolicy().partition(steps)["test"], [35, 37, 39])
        with self.assertRaisesRegex(ValueError, "split-reason"):
            self.run_pipeline(policy=SplitPolicy(.6, .2))


if __name__ == "__main__":
    unittest.main()

