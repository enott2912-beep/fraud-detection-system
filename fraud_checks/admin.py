from django.contrib import admin
from .models import FraudCheck


@admin.register(FraudCheck)
class FraudCheckAdmin(admin.ModelAdmin):
    list_display = (
        "transaction",
        "risk_score",
        "checked_at",
        "decision",
    )

    list_filter = (
        "checked_at",
    )

    search_fields = (
        'transaction',
    )

    ordering = ("-checked_at",)