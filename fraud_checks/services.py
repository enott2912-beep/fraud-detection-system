from transactions.models import Transaction
from accounts.models import Account
from fraud_checks.models import FraudCheck
from django.utils import timezone
from datetime import timedelta

def calculate_risk(transaction):
    risk_score = 0
    reasons = []

    amount = transaction.amount
    sender_account = transaction.sender_account
    account_age_days = (timezone.now() - sender_account.created_at).days
    recent_transaction_count = Transaction.objects.filter(sender_account=sender_account).filter(
    created_at__gte=timezone.now() - timedelta(hours=1)).count()

    if amount > 100000:
        risk_score += 40
        reasons.append("high_amount")

    if account_age_days < 7:
        risk_score += 20
        reasons.append("new_account")
    
    if recent_transaction_count > 10:
        risk_score += 30
        reasons.append("high_frequency")
    
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

    result = calculate_risk(transaction)

    fraud_check = FraudCheck.objects.create(
        transaction=transaction,
        risk_score=result["risk_score"],
        decision=result["decision"],
        reasons=result["reasons"],
    )

    return fraud_check