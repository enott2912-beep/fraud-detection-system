"""
Export the synthetic dataset to CSV, together with a snapshot of the
generation config used to produce it.

Separate from persist_to_db.py on purpose: this path keeps is_fraud /
fraud_pattern (ground truth for ML), the DB path strips them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .build_dataset import build_dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def export_dataset(version: str = "v1"):
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    accounts_df, transactions_df = build_dataset()

    csv_path = DATASETS_DIR / f"synthetic_{version}.csv"
    transactions_df.to_csv(csv_path, index=False)

    config_snapshot = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.SEED,
        "n_accounts": config.N_ACCOUNTS,
        "n_transactions": config.N_TRANSACTIONS,
        "fraud_ratio": config.FRAUD_RATIO,
        "enabled_fraud_patterns": config.ENABLED_FRAUD_PATTERNS,
        "account_created_days_range": list(config.ACCOUNT_CREATED_DAYS_RANGE),
        "transaction_created_days_range": list(config.TRANSACTION_DATE_DAYS_RANGE),
        "row_count": len(transactions_df),
        "fraud_count": int(transactions_df["is_fraud"].sum()),
        "fraud_ratio_actual": float(transactions_df["is_fraud"].mean()),
    }
    config_path = DATASETS_DIR / f"synthetic_{version}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    return csv_path, config_path


def main():
    csv_path, config_path = export_dataset()
    print(f"Saved dataset to {csv_path}")
    print(f"Saved config snapshot to {config_path}")


if __name__ == "__main__":
    main()