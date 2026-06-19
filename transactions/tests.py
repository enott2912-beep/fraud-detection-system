from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status
from accounts.models import Account
from transactions.models import Transaction
from fraud_checks.models import FraudCheck


class TransactionAPITests(APITestCase):

    def setUp(self):
        # Create users
        self.user1 = User.objects.create_user(username="user1", password="password")
        self.user2 = User.objects.create_user(username="user2", password="password")

        # Create accounts
        self.account1 = Account.objects.create(
            owner=self.user1,
            account_number="ACC123",
            balance=1000.00
        )
        self.account2 = Account.objects.create(
            owner=self.user2,
            account_number="ACC456",
            balance=500.00
        )

    def test_create_transaction_success(self):
        """Test successful creation of a transaction."""
        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "100.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Check transaction in database
        transaction = Transaction.objects.get(id=response.data["id"])
        self.assertEqual(transaction.amount, 100.00)
        self.assertEqual(transaction.sender_account, self.account1)
        self.assertEqual(transaction.receiver_account, self.account2)

        # Check that post_save signal created a FraudCheck
        self.assertTrue(FraudCheck.objects.filter(transaction=transaction).exists())
        fraud_check = transaction.fraud_check
        self.assertIsNotNone(fraud_check)

        # Check response contains serialized nested fraud_check
        self.assertIn("fraud_check", response.data)
        self.assertEqual(response.data["fraud_check"]["risk_score"], fraud_check.risk_score)
        self.assertEqual(response.data["fraud_check"]["decision"], fraud_check.decision)
        self.assertEqual(response.data["fraud_check"]["reasons"], fraud_check.reasons)

        # Check transaction status was updated according to decision
        self.assertEqual(transaction.status, Transaction.Status.APPROVED)
        self.assertEqual(response.data["status"], "APPROVED")

    def test_create_transaction_blocked(self):
        """Test transaction resulting in BLOCKED status (risk >= 60)."""
        # Sender account is new (< 7 days) -> +20 risk score
        # Amount > 100,000 -> +40 risk score
        # Total risk = 60 -> BLOCKED
        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "150000.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        transaction = Transaction.objects.get(id=response.data["id"])
        self.assertEqual(transaction.status, Transaction.Status.BLOCKED)
        self.assertEqual(response.data["status"], "BLOCKED")

        self.assertTrue(FraudCheck.objects.filter(transaction=transaction).exists())
        self.assertEqual(transaction.fraud_check.decision, "BLOCKED")
        self.assertEqual(transaction.fraud_check.risk_score, 60)

    def test_create_transaction_review(self):
        """Test transaction resulting in REVIEW decision -> PENDING status (30 <= risk < 60)."""
        from django.utils import timezone
        from datetime import timedelta

        # Set sender account age to 10 days (so no new_account risk score of +20)
        Account.objects.filter(id=self.account1.id).update(created_at=timezone.now() - timedelta(days=10))

        # Amount > 100,000 -> +40 risk score
        # Total risk = 40 -> REVIEW -> Transaction status is PENDING
        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "150000.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        transaction = Transaction.objects.get(id=response.data["id"])
        self.assertEqual(transaction.status, Transaction.Status.PENDING)
        self.assertEqual(response.data["status"], "PENDING")

        self.assertTrue(FraudCheck.objects.filter(transaction=transaction).exists())
        self.assertEqual(transaction.fraud_check.decision, "REVIEW")
        self.assertEqual(transaction.fraud_check.risk_score, 40)

    def test_create_transaction_invalid_amount(self):
        """Test validation error when amount is <= 0."""
        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "0.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("amount", response.data)

        # Negative amount
        data["amount"] = "-50.00"
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("amount", response.data)

    def test_create_transaction_same_accounts(self):
        """Test validation error when sender and receiver accounts are the same."""
        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account1.id,
            "amount": "100.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("non_field_errors", response.data)
        self.assertEqual(
            response.data["non_field_errors"][0],
            "Sender and receiver accounts must be different."
        )

    def test_transaction_endpoints_restricted(self):
        """Test that put, patch, delete are not allowed on transactions."""
        # Create a transaction
        tx = Transaction.objects.create(
            sender_account=self.account1,
            receiver_account=self.account2,
            amount=50.00
        )
        url = f"/api/transactions/{tx.id}/"

        # PUT not allowed
        response = self.client.put(url, {"amount": "60.00"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        # PATCH not allowed
        response = self.client.patch(url, {"amount": "60.00"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        # DELETE not allowed
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        # GET is allowed
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_accounts_endpoints_read_only(self):
        """Test that accounts endpoint is read-only."""
        url_list = "/api/accounts/"
        url_detail = f"/api/accounts/{self.account1.id}/"

        # GET is allowed
        response = self.client.get(url_list)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response = self.client.get(url_detail)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # POST is not allowed
        response = self.client.post(url_list, {"account_number": "ACC999"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_fraud_checks_endpoints_read_only(self):
        """Test that fraud-checks endpoint is read-only."""
        # Trigger fraud check creation by creating a transaction
        tx = Transaction.objects.create(
            sender_account=self.account1,
            receiver_account=self.account2,
            amount=50.00
        )
        fc = tx.fraud_check
        url_list = "/api/fraud-checks/"
        url_detail = f"/api/fraud-checks/{fc.id}/"

        # GET is allowed
        response = self.client.get(url_list)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response = self.client.get(url_detail)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # POST is not allowed
        response = self.client.post(url_list, {"risk_score": 10}, format="json")
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

