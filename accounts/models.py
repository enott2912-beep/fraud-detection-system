from django.db import models
from django.contrib.auth.models import User


class Account(models.Model):
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="accounts"
    )

    account_number = models.CharField(
        max_length=32,
        unique=True
    )

    balance = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    is_blocked = models.BooleanField(
        default=False
    )

    is_synthetic = models.BooleanField(
        default=False
    )

    def __str__(self):
        return self.account_number
