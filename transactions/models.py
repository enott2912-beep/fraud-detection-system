from django.db import models
from accounts.models import Account


class Transaction(models.Model):

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        APPROVED = "APPROVED", "Approved"
        BLOCKED = "BLOCKED", "Blocked"

    sender_account = models.ForeignKey(
        Account,
        on_delete=models.CASCADE,
        related_name="outgoing_transactions"
    )

    receiver_account = models.ForeignKey(
        Account,
        on_delete=models.CASCADE,
        related_name="incoming_transactions"
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:

        ordering = ['-created_at']

        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['created_at'])
        ]

    def __str__(self):
        return f"Transaction #{self.id}"
