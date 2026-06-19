from rest_framework import serializers
from .models import Transaction
from fraud_checks.models import FraudCheck


class NestedFraudCheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = FraudCheck
        fields = ['risk_score', 'decision', 'reasons']


class TransactionSerializer(serializers.ModelSerializer):
    fraud_check = NestedFraudCheckSerializer(read_only=True, allow_null=True)

    class Meta:
        model = Transaction
        fields = [
            'id',
            'sender_account',
            'receiver_account',
            'amount',
            'status',
            'created_at',
            'fraud_check',
        ]
        read_only_fields = ['status', 'created_at']

    def validate_amount(self, value):
        if value <= 0:
            raise serializers.ValidationError("Amount must be greater than 0.")
        return value

    def validate(self, attrs):
        sender = attrs.get('sender_account')
        receiver = attrs.get('receiver_account')
        
        if sender and receiver and sender == receiver:
            raise serializers.ValidationError("Sender and receiver accounts must be different.")
        
        return attrs
