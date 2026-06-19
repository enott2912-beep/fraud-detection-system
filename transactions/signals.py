from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Transaction


@receiver(post_save, sender=Transaction)
def trigger_fraud_check(sender, instance, created, **kwargs):
    if not created:
        return

    from fraud_checks.services import run_fraud_check

    run_fraud_check(instance)