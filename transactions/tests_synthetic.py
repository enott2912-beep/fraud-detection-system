import unittest

import numpy as np
import pandas as pd

from ml.synthetic import build_dataset as synthetic
from ml.synthetic import config


class SyntheticDatasetTests(unittest.TestCase):

    def setUp(self):
        self.reference_date = pd.Timestamp("2026-06-26 00:00:00", tz="UTC")
        self.accounts_df = pd.DataFrame(
            {
                "account_id": [1, 2, 3],
                "account_number": ["ACC1", "ACC2", "ACC3"],
                "balance": [1000.0, 2000.0, 3000.0],
                "created_offset_days": [30, 20, 10],
            }
        )
        self.accounts_df.attrs["reference_date"] = self.reference_date

    def test_assign_transaction_timestamps_and_account_age(self):
        transactions_df = pd.DataFrame(
            {
                "sender_account_id": [1, 2, 2, 2, 2],
                "receiver_account_id": [2, 1, 3, 1, 3],
                "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
                "is_fraud": [False, False, True, True, True],
                "fraud_pattern": [None, None, "velocity_fraud", "velocity_fraud", "velocity_fraud"],
                "burst_id": [np.nan, np.nan, 1, 1, 1],
                "transaction_hour": [5, 18, 9, 9, 9],
            }
        )
        rng = np.random.default_rng(123)

        assigned = synthetic.assign_transaction_timestamps(
            transactions_df, self.accounts_df, rng, self.reference_date
        )

        self.assertIn("created_at", assigned.columns)
        self.assertNotIn("sender_created_at", assigned.columns)

        sender_created_at = synthetic.offsets_to_datetime(
            self.accounts_df.set_index("account_id")["created_offset_days"],
            self.reference_date,
        )
        assigned_with_sender = assigned.merge(
            sender_created_at.rename("sender_created_at"),
            left_on="sender_account_id",
            right_index=True,
            how="left",
        )
        self.assertTrue(
            (assigned_with_sender["created_at"] > assigned_with_sender["sender_created_at"]).all()
        )

        normal = assigned[assigned["burst_id"].isna()]
        self.assertTrue((normal["created_at"].dt.hour == normal["transaction_hour"]).all())

        burst = assigned[assigned["burst_id"].notna()]
        self.assertLessEqual(
            burst["created_at"].max() - burst["created_at"].min(),
            pd.Timedelta(days=2, hours=1),
        )

        account_age_days = synthetic.compute_account_age_days(
            assigned, self.accounts_df
        )
        self.assertEqual(len(account_age_days), len(assigned))
        self.assertTrue((account_age_days >= 0).all())

    def test_compute_tx_last_hour(self):
        transactions_df = pd.DataFrame(
            {
                "sender_account_id": [1, 1, 1, 2, 1],
                "created_at": pd.to_datetime(
                    [
                        "2026-06-25 10:00:00+00:00",
                        "2026-06-25 10:30:00+00:00",
                        "2026-06-25 10:59:00+00:00",
                        "2026-06-25 11:00:00+00:00",
                        "2026-06-25 12:00:00+00:00",
                    ]
                ),
                "amount": [1, 1, 1, 1, 1],
            }
        ).sample(frac=1, random_state=7).reset_index(drop=True)

        tx_last_hour = synthetic.compute_tx_last_hour(transactions_df)

        ordered = transactions_df.assign(tx_last_hour=tx_last_hour).sort_values(
            "created_at"
        )
        sender1 = ordered[ordered["sender_account_id"] == 1]["tx_last_hour"].tolist()
        self.assertEqual(sender1[:3], [0, 1, 2])
        self.assertEqual(ordered[ordered["sender_account_id"] == 2]["tx_last_hour"].iloc[0], 0)

    def test_build_dataset_smoke(self):
        original_values = {
            "N_ACCOUNTS": config.N_ACCOUNTS,
            "N_TRANSACTIONS": config.N_TRANSACTIONS,
            "FRAUD_RATIO": config.FRAUD_RATIO,
        }
        try:
            config.N_ACCOUNTS = 32
            config.N_TRANSACTIONS = 120
            config.FRAUD_RATIO = 0.1

            accounts_df, transactions_df = synthetic.build_dataset()

            self.assertEqual(len(accounts_df), 32)
            self.assertEqual(len(transactions_df), 120)
            self.assertTrue(transactions_df["created_at"].notna().all())
            sender_created_at = synthetic.offsets_to_datetime(
                accounts_df.set_index("account_id")["created_offset_days"],
                accounts_df.attrs["reference_date"],
            )
            merged = transactions_df.merge(
                sender_created_at.rename("sender_created_at"),
                left_on="sender_account_id",
                right_index=True,
                how="left",
            )
            self.assertTrue((merged["created_at"] > merged["sender_created_at"]).all())
            self.assertTrue((transactions_df["account_age_days"] >= 0).all())
            self.assertTrue((transactions_df["tx_last_hour"] >= 0).all())
        finally:
            for key, value in original_values.items():
                setattr(config, key, value)


if __name__ == "__main__":
    unittest.main()
