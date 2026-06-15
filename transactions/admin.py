from django.contrib import admin
from .models import Transaction


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "sender_account",
        "receiver_account",
        "amount",
        "status",
        "created_at",
    )

    list_filter = (
        "status",
        "created_at",
    )

    ordering = ("-created_at",)
