from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "synthetic_v5.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model_v5.pkl"

FEATURE_COLS = [
    "amount", "account_age_days", "tx_last_hour", "transaction_hour",
    "receiver_tx_count_24h", "sender_daily_amount_sum",
    "amount_to_balance_ratio", "days_since_last_tx",
]

BEST_THRESHOLD = 0.59

ESCALATION_THRESHOLD = 0.29
