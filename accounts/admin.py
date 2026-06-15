from django.contrib import admin
from .models import Account


@admin.register(Account)
class AccountAdmin(admin.ModelAdmin):
    list_display = (
        "account_number",
        "owner",
        "balance",
    )

    search_fields = (
        "account_number",
        "owner__username",
    )

    ordering = ("-created_at",)
