"""
Persist synthetic dataset into the local Django database.
"""
from __future__ import annotations

import os
from decimal import Decimal

import pandas as pd
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.db import transaction  # noqa: E402

from accounts.models import Account  # noqa: E402
from transactions.models import Transaction  # noqa: E402
from fraud_checks.models import FraudCheck  # noqa: E402

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
    FraudCheck.objects.all().delete()
    Transaction.objects.all().delete()
    Account.objects.all().delete()


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
                    created_at=(
                        reference_date
                        - pd.to_timedelta(row["created_offset_days"], unit="D")
                    ).to_pydatetime(),
                    is_blocked=False,
                )
            )

        Account.objects.bulk_create(accounts_to_create, batch_size=1000)
        created_accounts = {
            account.account_number: account
            for account in Account.objects.filter(
                account_number__in=accounts_df["account_number"].tolist()
            )
        }
        account_id_map = {
            row["account_id"]: created_accounts[row["account_number"]]
            for _, row in accounts_df.iterrows()
        }

        transactions_to_create = []
        for _, row in transactions_df.iterrows():
            status = (
                Transaction.Status.BLOCKED
                if bool(row["is_fraud"])
                else Transaction.Status.APPROVED
            )
            transactions_to_create.append(
                Transaction(
                    sender_account=account_id_map[int(row["sender_account_id"])],
                    receiver_account=account_id_map[int(row["receiver_account_id"])],
                    amount=Decimal(str(row["amount"])),
                    status=status,
                    created_at=pd.Timestamp(row["created_at"]).to_pydatetime(),
                )
            )

        Transaction.objects.bulk_create(transactions_to_create, batch_size=1000)

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
