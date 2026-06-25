"""
Statistical distributions for synthetic features.
"""
import numpy as np


def generate_balance(n, rng):
    balances = rng.lognormal(mean=8.2, sigma=1.0, size=n)
    return np.round(balances, 2)


def generate_normal_amount(n, rng):
    amounts = rng.lognormal(mean=8.7, sigma=0.5, size=n)
    return np.round(amounts, 2)


def generate_fraud_amount(n, rng):
    amounts = rng.lognormal(mean=11.0, sigma=0.6, size=n)
    return np.round(amounts, 2)


def generate_account_created_offset(n, rng, days_range):
    return rng.integers(low=days_range[0], high=days_range[1], size=n)


def generate_transaction_hour(n, rng, fraud_mask):
    fraud_mask = np.asarray(fraud_mask, dtype=bool)

    hours = np.arange(24)

    normal_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.03, 0.04, 0.06, 0.07, 0.07, 0.07,
        0.07, 0.07, 0.07, 0.07, 0.07, 0.06,
        0.05, 0.04, 0.03, 0.02, 0.02, 0.02,
    ])
    normal_weights = normal_weights / normal_weights.sum()

    fraud_weights = np.array([
        0.11, 0.11, 0.10, 0.10, 0.09, 0.08,
        0.04, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.04, 0.06,
    ])
    fraud_weights = fraud_weights / fraud_weights.sum()

    transaction_hours = rng.choice(hours, size=n, p=normal_weights)
    fraud_count = int(fraud_mask.sum())
    if fraud_count:
        transaction_hours[fraud_mask] = rng.choice(
            hours,
            size=fraud_count,
            p=fraud_weights,
        )

    return transaction_hours
