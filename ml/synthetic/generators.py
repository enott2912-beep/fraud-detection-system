"""
Fraud pattern generators for synthetic data.
"""
import numpy as np
import pandas as pd

from .distributions import (
    generate_normal_amount,
    generate_fraud_amount,
    generate_transaction_hour,
)
from .config import SEED


def split_fraud_counts(n_fraud_total, n_patterns):

    base = n_fraud_total // n_patterns
    counts = [base] * n_patterns
    remainder = n_fraud_total - base * n_patterns
    counts[-1] += remainder
    return counts


def generate_normal_transactions(n, accounts_df, rng):
    account_ids = accounts_df["account_id"].to_numpy()

    senders = rng.choice(account_ids, size=n)
    receivers = rng.choice(account_ids, size=n)

    same_mask = senders == receivers
    while same_mask.any():
        receivers[same_mask] = rng.choice(account_ids, size=same_mask.sum())
        same_mask = senders == receivers

    amounts = generate_normal_amount(n, rng)

    return pd.DataFrame({
        "sender_account_id": senders,
        "receiver_account_id": receivers,
        "amount": amounts,
        "is_fraud": False,
        "fraud_pattern": None,
    })


def new_account_fraud(n, accounts_df, rng):
    account_ids = accounts_df["account_id"].to_numpy()
    ages = accounts_df["created_offset_days"].to_numpy().astype(float)

   
    K = 30.0
    raw_weights = np.exp(-ages / K)
    weights = raw_weights / raw_weights.sum()

    senders = rng.choice(account_ids, size=n, p=weights)

    receivers = rng.choice(account_ids, size=n)
    same_mask = senders == receivers
    while same_mask.any():
        receivers[same_mask] = rng.choice(account_ids, size=same_mask.sum())
        same_mask = senders == receivers

    amounts = generate_fraud_amount(n, rng)

    return pd.DataFrame({
        "sender_account_id": senders,
        "receiver_account_id": receivers,
        "amount": amounts,
        "is_fraud": True,
        "fraud_pattern": "new_account_fraud",
    })


def velocity_fraud(n, accounts_df, rng, burst_size_range=(5, 15)):
    account_ids = accounts_df["account_id"].to_numpy()

    rows = []
    burst_id = 0
    generated = 0

    while generated < n:
        burst_size = rng.integers(burst_size_range[0], burst_size_range[1] + 1)
        burst_size = min(burst_size, n - generated)

        sender = rng.choice(account_ids)

        receivers = rng.choice(account_ids, size=burst_size)
        same_mask = receivers == sender
        while same_mask.any():
            receivers[same_mask] = rng.choice(account_ids, size=same_mask.sum())
            same_mask = receivers == sender

        amounts = generate_fraud_amount(burst_size, rng)

        burst_df = pd.DataFrame({
            "sender_account_id": np.full(burst_size, sender),
            "receiver_account_id": receivers,
            "amount": amounts,
            "is_fraud": True,
            "fraud_pattern": "velocity_fraud",
            "burst_id": burst_id,
        })
        rows.append(burst_df)

        generated += burst_size
        burst_id += 1

    return pd.concat(rows, ignore_index=True)


FRAUD_PATTERN_REGISTRY = {
    "new_account_fraud": new_account_fraud,
    "velocity_fraud": velocity_fraud,
}


def generate_all_transactions(n_total, fraud_ratio, accounts_df, rng,
                                enabled_patterns):
    n_fraud_total = int(n_total * fraud_ratio)
    n_normal = n_total - n_fraud_total

    pattern_counts = split_fraud_counts(n_fraud_total, len(enabled_patterns))

    frames = [generate_normal_transactions(n_normal, accounts_df, rng)]

    for pattern_name, pattern_n in zip(enabled_patterns, pattern_counts):
        generator_fn = FRAUD_PATTERN_REGISTRY[pattern_name]
        frames.append(generator_fn(pattern_n, accounts_df, rng))

    df = pd.concat(frames, ignore_index=True)

    df = df.sample(frac=1, random_state=SEED)

    df["transaction_hour"] = generate_transaction_hour(
        len(df), rng, fraud_mask=df["is_fraud"]
    )

    return df.reset_index(drop=True)