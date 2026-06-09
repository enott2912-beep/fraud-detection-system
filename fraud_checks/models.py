from django.db import models

from django.db import models
from transactions.models import Transaction


class FraudCheck(models.Model):

    class Decision(models.TextChoices):
        APPROVED = "APPROVED", "Approved"
        REVIEW = "REVIEW", "Review"
        BLOCKED = "BLOCKED", "Blocked"

    transaction = models.OneToOneField(
        Transaction,
        on_delete=models.CASCADE,
        related_name="fraud_check"
    )

    risk_score = models.PositiveSmallIntegerField()

    decision = models.CharField(
        max_length=20,
        choices=Decision.choices
    )

    reason = models.TextField()

    checked_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"Fraud check for transaction {self.transaction.id}"
