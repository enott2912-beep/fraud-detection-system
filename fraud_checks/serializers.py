from rest_framework import serializers
from .models import FraudCheck


class FraudCheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = FraudCheck
        fields = ['id', 'transaction', 'risk_score', 'decision', 'reasons', 'checked_at']
