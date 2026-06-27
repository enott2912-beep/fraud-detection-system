"""
Persist synthetic dataset into the local Django database.
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal

import pandas as pd
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.db import transaction  # noqa: E402

from accounts.models import Account  # noqa: E402
from transactions.models import Transaction  # noqa: E402

from .build_dataset import build_dataset  # noqa: E402


DEMO_USER_COUNT = 10


def ensure_users():
    users = list(User.objects.order_by("id"))
    if users:
        return users

    users = []
    for index in range(1, DEMO_USER_COUNT + 1):
        users.append(
            User.objects.create_user(
                username=f"demo_user_{index}",
                password="demo-password",
            )
        )
    return users


def clear_synthetic_tables():
    Account.objects.filter(is_synthetic=True).delete()


def persist_dataset():
    accounts_df, transactions_df = build_dataset()
    reference_date = accounts_df.attrs["reference_date"]

    with transaction.atomic():
        users = ensure_users()
        clear_synthetic_tables()

        accounts_to_create = []
        for index, row in accounts_df.iterrows():
            owner = users[index % len(users)]
            accounts_to_create.append(
                Account(
                    owner=owner,
                    account_number=row["account_number"],
                    balance=Decimal(str(row["balance"])),
                    is_synthetic=True,
                    created_at=(
                        reference_date
                        - pd.to_timedelta(row["created_offset_days"], unit="D")
                    ).to_pydatetime(),
                )
            )

        created_accounts = Account.objects.bulk_create(
            accounts_to_create, batch_size=1000
        )
        for account_obj, row in zip(created_accounts, accounts_df.itertuples()):
            account_obj.created_at = (
                reference_date
                - pd.to_timedelta(row.created_offset_days, unit="D")
            ).to_pydatetime()
        Account.objects.bulk_update(created_accounts, ["created_at"], batch_size=1000)
        account_id_map = {
            row.account_id: account_obj
            for row, account_obj in zip(accounts_df.itertuples(), created_accounts)
        }

        transactions_to_create = []
        for _, row in transactions_df.iterrows():
            transactions_to_create.append(
                Transaction(
                    sender_account=account_id_map[int(row["sender_account_id"])],
                    receiver_account=account_id_map[int(row["receiver_account_id"])],
                    amount=Decimal(str(row["amount"])),
                    created_at=pd.Timestamp(row["created_at"]).to_pydatetime(),
                )
            )

        created_txns = Transaction.objects.bulk_create(
            transactions_to_create, batch_size=1000
        )
        for txn_obj, row in zip(created_txns, transactions_df.itertuples()):
            txn_obj.created_at = row.created_at.to_pydatetime()
        Transaction.objects.bulk_update(created_txns, ["created_at"], batch_size=1000)

    return {
        "accounts": len(accounts_to_create),
        "transactions": len(transactions_to_create),
        "users": len(users),
    }


def main():
    result = persist_dataset()
    print(
        f"Seeded {result['accounts']} accounts, "
        f"{result['transactions']} transactions, "
        f"using {result['users']} users."
    )


if __name__ == "__main__":
    main()
