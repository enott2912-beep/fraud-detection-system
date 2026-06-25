"""
Central configuration of the synthetic data generator.
"""

SEED = 42

N_ACCOUNTS = 2000
N_TRANSACTIONS = 20000

FRAUD_RATIO = 0.03

ENABLED_FRAUD_PATTERNS = [
    "new_account_fraud",
    "velocity_fraud",
]

ACCOUNT_CREATED_DAYS_RANGE = (1, 730)

TRANSACTION_DATE_RANGE_DAYS_RANGE = (1, 365)