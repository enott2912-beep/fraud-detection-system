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

    senders = rng.choice(account_ids, size=n)

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


def mule_fraud(n, accounts_df, rng, burst_size_range=(8, 20)):
    account_ids = accounts_df["account_id"].to_numpy()
    rows = []
    burst_id = 0
    generated = 0
    while generated < n:
        burst_size = rng.integers(burst_size_range[0], burst_size_range[1] + 1)
        burst_size = min(burst_size, n - generated)
        receiver = rng.choice(account_ids)
        senders = rng.choice(account_ids, size=burst_size, replace=False)
        same_mask = senders == receiver
        while same_mask.any():
            senders[same_mask] = rng.choice(account_ids, size=same_mask.sum())
            same_mask = senders == receiver
        amounts = generate_normal_amount(burst_size, rng)
        burst_df = pd.DataFrame({
            "sender_account_id": senders,
            "receiver_account_id": np.full(burst_size, receiver),
            "amount": amounts,
            "is_fraud": True,
            "fraud_pattern": "mule_fraud",
            "burst_id": burst_id,
        })
        rows.append(burst_df)
        generated += burst_size
        burst_id += 1
    return pd.concat(rows, ignore_index=True)


def structuring_fraud(n, accounts_df, rng, burst_size_range=(3, 6)):
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
        amounts = rng.uniform(40000, 95000, size=burst_size)
        amounts = np.round(amounts, 2)
        burst_df = pd.DataFrame({
            "sender_account_id": np.full(burst_size, sender),
            "receiver_account_id": receivers,
            "amount": amounts,
            "is_fraud": True,
            "fraud_pattern": "structuring_fraud",
            "burst_id": burst_id,
        })
        rows.append(burst_df)
        generated += burst_size
        burst_id += 1
    return pd.concat(rows, ignore_index=True)


def balance_drain_fraud(n, accounts_df, rng):
    account_ids = accounts_df["account_id"].to_numpy()
    balances = accounts_df["balance"].to_numpy()
    balance_map = dict(zip(account_ids, balances))
    senders = rng.choice(account_ids, size=n)
    receivers = rng.choice(account_ids, size=n)
    same_mask = senders == receivers
    while same_mask.any():
        receivers[same_mask] = rng.choice(account_ids, size=same_mask.sum())
        same_mask = senders == receivers
    sender_balances = np.array([balance_map[s] for s in senders])
    drain_ratios = rng.uniform(0.85, 0.99, size=n)
    amounts = np.round(sender_balances * drain_ratios, 2)
    return pd.DataFrame({
        "sender_account_id": senders,
        "receiver_account_id": receivers,
        "amount": amounts,
        "is_fraud": True,
        "fraud_pattern": "balance_drain_fraud",
    })


def dormant_reactivation_fraud(n, accounts_df, rng, min_age_days=30):
    account_ids = accounts_df["account_id"].to_numpy()
    ages = accounts_df["created_offset_days"].to_numpy().astype(float)
    old_mask = ages >= min_age_days
    old_ids = account_ids[old_mask]
    if len(old_ids) == 0:
        raise ValueError(f"No accounts with age >= {min_age_days} days")
    replace = n > len(old_ids)
    senders = rng.choice(old_ids, size=n, replace=replace)
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
        "fraud_pattern": "dormant_reactivation_fraud",
    })


FRAUD_PATTERN_REGISTRY = {
    "new_account_fraud": new_account_fraud,
    "velocity_fraud": velocity_fraud,
    "mule_fraud": mule_fraud,
    "structuring_fraud": structuring_fraud,
    "balance_drain_fraud": balance_drain_fraud,
    "dormant_reactivation_fraud": dormant_reactivation_fraud,
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

    if "burst_id" in df.columns:
        bid_offset = 0
        for pattern_name in ["velocity_fraud", "mule_fraud", "structuring_fraud"]:
            mask = df["fraud_pattern"] == pattern_name
            if mask.any():
                df.loc[mask, "burst_id"] = df.loc[mask, "burst_id"] + bid_offset
                bid_offset = int(df.loc[mask, "burst_id"].max()) + 1

    df = df.sample(frac=1, random_state=SEED)

    patterns_with_night_hours = ["new_account_fraud", "velocity_fraud"]
    night_mask = df["fraud_pattern"].isin(patterns_with_night_hours)
    df["transaction_hour"] = generate_transaction_hour(
        len(df), rng, fraud_mask=night_mask
    )

    return df.reset_index(drop=True)