"""
Assembling the final dataset.
"""
import numpy as np
import pandas as pd
from faker import Faker

from . import config
from .distributions import generate_balance, generate_account_created_offset
from .generators import generate_all_transactions


def generate_accounts(n, rng):

    fake = Faker()
    Faker.seed(config.SEED)

    account_ids = np.arange(1, n + 1)
    created_offset_days = generate_account_created_offset(
        n, rng, config.ACCOUNT_CREATED_DAYS_RANGE
    )
    balances = generate_balance(n, rng)
    account_numbers = [fake.unique.iban() for _ in range(n)]

    return pd.DataFrame({
        "account_id": account_ids,
        "account_number": account_numbers,
        "balance": balances,
        "created_offset_days": created_offset_days,
    })


def offsets_to_datetime(offset_days, reference_date):

    return reference_date - pd.to_timedelta(offset_days, unit="D")


def assign_transaction_timestamps(transactions_df, accounts_df, rng, reference_date):

    transactions_df = transactions_df.copy()

    accounts_with_created = accounts_df.loc[
        :, ["account_id", "created_offset_days"]
    ].copy()
    accounts_with_created["sender_created_at"] = offsets_to_datetime(
        accounts_with_created["created_offset_days"], reference_date
    )

    transactions_df = transactions_df.merge(
        accounts_with_created[["account_id", "sender_created_at"]],
        left_on="sender_account_id",
        right_on="account_id",
        how="left",
        validate="many_to_one",
    )
    transactions_df = transactions_df.drop(columns=["account_id"])

    reference_day = reference_date.normalize() - pd.Timedelta(days=1)
    transaction_datetimes = pd.Series(
        index=transactions_df.index, dtype="datetime64[ns, UTC]"
    )

    burst_mask = transactions_df["burst_id"].notna()
    normal_mask = ~burst_mask

    if normal_mask.any():
        normal_df = transactions_df.loc[normal_mask].copy()
        sender_days = normal_df["sender_created_at"].dt.normalize()
        max_offsets = (reference_day - sender_days).dt.days
        max_offsets = max_offsets.clip(lower=0)
        day_offsets = pd.Series(
            rng.integers(0, max_offsets.to_numpy() + 1),
            index=normal_df.index,
        )
        base_dates = sender_days + pd.to_timedelta(day_offsets, unit="D")

        minutes = rng.integers(0, 60, size=len(normal_df))
        seconds = rng.integers(0, 60, size=len(normal_df))
        timestamps = pd.Series(
            (
                base_dates
                + pd.to_timedelta(normal_df["transaction_hour"], unit="h")
                + pd.to_timedelta(minutes, unit="m")
                + pd.to_timedelta(seconds, unit="s")
            ),
            index=normal_df.index,
        )

        needs_bump = timestamps <= normal_df["sender_created_at"]
        if needs_bump.any():
            timestamps.loc[needs_bump] = (
                normal_df.loc[needs_bump, "sender_created_at"]
                + pd.to_timedelta(1, unit="s")
            )

        transaction_datetimes.loc[normal_df.index] = timestamps

    if burst_mask.any():
        burst_groups = transactions_df.loc[burst_mask].groupby("burst_id", sort=False)
        for _, burst_df in burst_groups:
            burst_df = burst_df.copy()
            sender_created_at = burst_df["sender_created_at"].iloc[0]
            burst_day_start = sender_created_at.normalize()
            max_offset = int((reference_day - burst_day_start).days)
            day_offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            base_day = burst_day_start + pd.to_timedelta(day_offset, unit="D")
            base_hour = int(burst_df["transaction_hour"].iloc[0])
            base_timestamp = base_day + pd.to_timedelta(base_hour, unit="h")

            minute_offsets = rng.integers(0, 60, size=len(burst_df))
            second_offsets = rng.integers(0, 60, size=len(burst_df))
            burst_timestamps = pd.Series(
                (
                    base_timestamp
                    + pd.to_timedelta(minute_offsets, unit="m")
                    + pd.to_timedelta(second_offsets, unit="s")
                ),
                index=burst_df.index,
            )

            needs_bump = burst_timestamps <= burst_df["sender_created_at"]
            if needs_bump.any():
                burst_timestamps.loc[needs_bump] = (
                    burst_df.loc[needs_bump, "sender_created_at"]
                    + pd.to_timedelta(1, unit="s")
                )

            transaction_datetimes.loc[burst_df.index] = burst_timestamps

    transactions_df["created_at"] = transaction_datetimes
    return transactions_df.drop(columns=["sender_created_at"])


def compute_account_age_days(transactions_df, accounts_df):

    reference_date = accounts_df.attrs.get(
        "reference_date",
        pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1),
    )
    sender_created_at = accounts_df.loc[:, ["account_id", "created_offset_days"]].copy()
    sender_created_at["sender_created_at"] = offsets_to_datetime(
        sender_created_at["created_offset_days"], reference_date
    )
    merged = transactions_df.merge(
        sender_created_at[["account_id", "sender_created_at"]],
        left_on="sender_account_id",
        right_on="account_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["account_id"])
    return (merged["created_at"] - merged["sender_created_at"]).dt.days


def compute_tx_last_hour(transactions_df):

    ordered = transactions_df.reset_index(names="_orig_index").sort_values(
        ["sender_account_id", "created_at", "_orig_index"]
    )

    rolling_counts = (
        ordered.groupby("sender_account_id", sort=False)
        .rolling("1h", on="created_at", closed="left")["amount"]
        .count()
        .reset_index(level=0, drop=True)
    )

    ordered["tx_last_hour"] = rolling_counts.fillna(0).to_numpy(dtype="int64")
    return ordered.sort_values("_orig_index").set_index("_orig_index")[
        "tx_last_hour"
    ].sort_index()


def build_dataset():

    rng = np.random.default_rng(config.SEED)
    reference_date = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)

    accounts_df = generate_accounts(config.N_ACCOUNTS, rng)
    accounts_df.attrs["reference_date"] = reference_date

    transactions_df = generate_all_transactions(
        n_total=config.N_TRANSACTIONS,
        fraud_ratio=config.FRAUD_RATIO,
        accounts_df=accounts_df,
        rng=rng,
        enabled_patterns=config.ENABLED_FRAUD_PATTERNS,
    )

    transactions_df = assign_transaction_timestamps(
        transactions_df, accounts_df, rng, reference_date
    )

    transactions_df["account_age_days"] = compute_account_age_days(
        transactions_df, accounts_df
    )
    transactions_df["tx_last_hour"] = compute_tx_last_hour(transactions_df)

    return accounts_df, transactions_df


if __name__ == "__main__":
    accounts_df, transactions_df = build_dataset()
    print(accounts_df.head())
    print(transactions_df.head())
    print(f"Total transactions: {len(transactions_df)}")
    print(f"Fraud: {transactions_df['is_fraud'].sum()} "
          f"({transactions_df['is_fraud'].mean():.2%})")
