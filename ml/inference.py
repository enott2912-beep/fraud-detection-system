from datetime import timedelta

import joblib
import pandas as pd
from django.db.models import Sum

from ml.constants import FEATURE_COLS, MODEL_PATH

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def compute_live_features(transaction):
    from transactions.models import Transaction

    sender = transaction.sender_account
    receiver = transaction.receiver_account
    tx_time = transaction.created_at
    amount = float(transaction.amount)
    sender_balance = float(sender.balance)

    account_age_days = max((tx_time - sender.created_at).days, 0)

    one_hour_ago = tx_time - timedelta(hours=1)
    tx_last_hour = Transaction.objects.filter(
        sender_account=sender,
        created_at__gte=one_hour_ago,
        created_at__lt=tx_time,
    ).count()

    transaction_hour = tx_time.hour

    twenty_four_hours_ago = tx_time - timedelta(hours=24)
    receiver_tx_count_24h = Transaction.objects.filter(
        receiver_account=receiver,
        created_at__gte=twenty_four_hours_ago,
        created_at__lt=tx_time,
    ).count()

    sender_daily_amount_sum = float(
        Transaction.objects.filter(
            sender_account=sender,
            created_at__gte=twenty_four_hours_ago,
            created_at__lt=tx_time,
        ).exclude(status=Transaction.Status.BLOCKED)
         .aggregate(total=Sum("amount"))["total"] or 0
    )

    amount_to_balance_ratio = (
        round(amount / sender_balance, 4) if sender_balance > 0 else 0.0
    )

    prev_tx = (
        Transaction.objects.filter(
            sender_account=sender,
            created_at__lt=tx_time,
        )
        .order_by("-created_at")
        .first()
    )
    if prev_tx:
        days_since_last_tx = max(
            (tx_time - prev_tx.created_at).total_seconds() / 86400, 0
        )
    else:
        days_since_last_tx = float(account_age_days)

    return {
        "amount": amount,
        "account_age_days": account_age_days,
        "tx_last_hour": tx_last_hour,
        "transaction_hour": transaction_hour,
        "receiver_tx_count_24h": receiver_tx_count_24h,
        "sender_daily_amount_sum": sender_daily_amount_sum,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "days_since_last_tx": days_since_last_tx,
    }


def predict_ml_proba(transaction):
    features = compute_live_features(transaction)
    model = load_model()
    X = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    return float(model.predict_proba(X)[0, 1])
