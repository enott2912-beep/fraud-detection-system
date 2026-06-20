from decimal import Decimal
from django.db import transaction as db_transaction
from django.db.models import Sum
from django.utils import timezone
from datetime import timedelta
from transactions.models import Transaction
from accounts.models import Account
from fraud_checks.models import FraudCheck


def calculate_risk(transaction):
    risk_score = 0
    reasons = []

    amount = Decimal(str(transaction.amount))
    sender_account = transaction.sender_account
    account_age_days = (timezone.now() - sender_account.created_at).days
    
    # 1. Frequency check
    tx_query = Transaction.objects.filter(
        sender_account=sender_account,
        created_at__gte=timezone.now() - timedelta(hours=1)
    )
    if transaction.id:
        tx_query = tx_query.exclude(id=transaction.id)
    recent_transaction_count = tx_query.count()

    # 2. Blocked account check
    if sender_account.is_blocked:
        risk_score += 100
        reasons.append("account_blocked")

    # 3. Balance sufficiency check
    sender_balance = Decimal(str(sender_account.balance))
    if sender_balance < amount:
        risk_score += 80
        reasons.append("insufficient_balance")

    # 4. Daily limits check
    daily_query = Transaction.objects.filter(
        sender_account=sender_account,
        created_at__gte=timezone.now() - timedelta(days=1)
    ).exclude(status=Transaction.Status.BLOCKED)
    if transaction.id:
        daily_query = daily_query.exclude(id=transaction.id)
    daily_spent_val = daily_query.aggregate(total=Sum('amount'))['total'] or 0
    daily_spent = Decimal(str(daily_spent_val))

    DAILY_LIMIT = Decimal('200000.00')
    if daily_spent + amount > DAILY_LIMIT:
        risk_score += 60
        reasons.append("limit_exceeded")

    # 5. Amount check
    if amount > 100000:
        risk_score += 40
        reasons.append("high_amount")

    # 6. Age check
    if account_age_days < 7:
        risk_score += 20
        reasons.append("new_account")
    
    # 7. Frequency check evaluation
    if recent_transaction_count > 10:
        risk_score += 30
        reasons.append("high_frequency")
    
    # Decision evaluation
    if risk_score >= 60:
        decision = "BLOCKED"
    elif risk_score >= 30:
        decision = "REVIEW"
    else:
        decision = "APPROVED"       

    return {
        "risk_score": risk_score,
        "decision": decision,
        "reasons": reasons,
    }


def run_fraud_check(transaction):
    with db_transaction.atomic():
        # Retrieve sender and receiver accounts with a row lock to prevent race conditions
        sender = Account.objects.select_for_update().get(id=transaction.sender_account.id)
        receiver = Account.objects.select_for_update().get(id=transaction.receiver_account.id)

        # Update the references inside transaction to use the locked ones
        transaction.sender_account = sender
        transaction.receiver_account = receiver

        # Calculate risk score and decision
        result = calculate_risk(transaction)

        # Create FraudCheck record
        fraud_check = FraudCheck.objects.create(
            transaction=transaction,
            risk_score=result["risk_score"],
            decision=result["decision"],
            reasons=result["reasons"],
        )

        status_map = {
            "APPROVED": Transaction.Status.APPROVED,
            "BLOCKED": Transaction.Status.BLOCKED,
            "REVIEW": Transaction.Status.PENDING,
        }

        new_status = status_map.get(result["decision"])

        # Update and save the transaction status
        if new_status and new_status != transaction.status:
            transaction.status = new_status
            transaction.save(update_fields=["status"])

        # Execute balance transfer if APPROVED
        if result["decision"] == "APPROVED":
            amount = Decimal(str(transaction.amount))
            sender.balance = Decimal(str(sender.balance)) - amount
            receiver.balance = Decimal(str(receiver.balance)) + amount
            sender.save(update_fields=["balance"])
            receiver.save(update_fields=["balance"])

        # Lock account if it reaches 3+ BLOCKED transactions
        elif result["decision"] == "BLOCKED":
            blocked_count = Transaction.objects.filter(
                sender_account=sender,
                status=Transaction.Status.BLOCKED
            ).count()
            if blocked_count >= 3:
                sender.is_blocked = True
                sender.save(update_fields=["is_blocked"])

    return fraud_check