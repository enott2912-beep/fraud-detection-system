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
        self.account1.balance = 200000.00
        self.account1.save()

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

        self.account1.balance = 200000.00
        self.account1.save()

        # Set sender account age to 10 days (so no new_account risk score of +20)
        Account.objects.filter(id=self.account1.id).update(created_at=timezone.now() - timedelta(days=10))
        self.account1.refresh_from_db()

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

    def test_insufficient_balance(self):
        """Test transaction blocked when sender has insufficient balance."""
        self.account1.balance = 50.00
        self.account1.save()

        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "100.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Transaction should be BLOCKED due to insufficient balance
        self.assertEqual(response.data["status"], "BLOCKED")
        self.assertIn("insufficient_balance", response.data["fraud_check"]["reasons"])

        # Check balances were not modified
        self.account1.refresh_from_db()
        self.account2.refresh_from_db()
        self.assertEqual(self.account1.balance, 50.00)
        self.assertEqual(self.account2.balance, 500.00)

    def test_balance_transfer_success(self):
        """Test that a successful APPROVED transaction transfers balances correctly."""
        self.account1.balance = 1000.00
        self.account1.save()

        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "200.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(response.data["status"], "APPROVED")

        # Verify balances updated in database
        self.account1.refresh_from_db()
        self.account2.refresh_from_db()
        self.assertEqual(self.account1.balance, 800.00)
        self.assertEqual(self.account2.balance, 700.00)

    def test_daily_spent_limit_exceeded(self):
        """Test daily spent limit check (exceeding DAILY_LIMIT = 200000.00)."""
        from django.utils import timezone
        from datetime import timedelta

        # Make account old to avoid new_account risk (+20)
        Account.objects.filter(id=self.account1.id).update(created_at=timezone.now() - timedelta(days=10))
        self.account1.refresh_from_db()

        # Increase balance to allow large transactions
        self.account1.balance = 300000.00
        self.account1.save()

        # Create previous transactions summing to 180,000.00
        # They must be in status APPROVED to be included in daily spent sum
        Transaction.objects.create(
            sender_account=self.account1,
            receiver_account=self.account2,
            amount=180000.00,
            status=Transaction.Status.APPROVED
        )

        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "30000.00"  # 180000 + 30000 = 210000 > 200000 limit
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Should be BLOCKED due to daily spent limit exceeded
        self.assertEqual(response.data["status"], "BLOCKED")
        self.assertIn("limit_exceeded", response.data["fraud_check"]["reasons"])

        # Balance should not be deducted for blocked transaction
        self.account1.refresh_from_db()
        self.assertEqual(self.account1.balance, 300000.00)

    def test_account_auto_blocking_after_repeated_blocks(self):
        """Test that account is automatically blocked after 3 BLOCKED transactions."""
        self.account1.balance = 10.00
        self.account1.save()

        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "50.00"  # gets BLOCKED due to insufficient balance
        }

        # 1st blocked transaction
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["status"], "BLOCKED")
        self.account1.refresh_from_db()
        self.assertFalse(self.account1.is_blocked)

        # 2nd blocked transaction
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["status"], "BLOCKED")
        self.account1.refresh_from_db()
        self.assertFalse(self.account1.is_blocked)

        # 3rd blocked transaction
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["status"], "BLOCKED")
        self.account1.refresh_from_db()
        self.assertTrue(self.account1.is_blocked)  # now blocked!

    def test_blocked_account_rejected_immediately(self):
        """Test that a transaction from an already blocked account is rejected immediately."""
        self.account1.is_blocked = True
        self.account1.save()

        url = "/api/transactions/"
        data = {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "5.00"
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(response.data["status"], "BLOCKED")
        self.assertIn("account_blocked", response.data["fraud_check"]["reasons"])
        self.assertEqual(response.data["fraud_check"]["risk_score"], 120)  # 100 blocked + 20 new_account

    def test_analytics_endpoint(self):
        """Test the custom analytics endpoint aggregates data correctly."""
        from django.utils import timezone
        from datetime import timedelta
        
        # Clear objects created in setUp if any to have exact predictable counts
        Transaction.objects.all().delete()
        FraudCheck.objects.all().delete()
        
        # Set account1 to old to trigger REVIEW (+40 risk) on high amount, and give it high balance
        Account.objects.filter(id=self.account1.id).update(created_at=timezone.now() - timedelta(days=10))
        self.account1.refresh_from_db()
        self.account1.balance = 500000.00
        self.account1.save()

        # Set account2 to old, give it high balance
        Account.objects.filter(id=self.account2.id).update(created_at=timezone.now() - timedelta(days=10))
        self.account2.refresh_from_db()
        self.account2.balance = 500000.00
        self.account2.save()

        url_transactions = "/api/transactions/"

        # 1. Transaction 1: account1 -> account2, amount 100.00 (APPROVED, risk_score = 0)
        response = self.client.post(url_transactions, {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "100.00"
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # 2. Transaction 2: account1 -> account2, amount 150000.00 (REVIEW, risk_score = 40)
        response = self.client.post(url_transactions, {
            "sender_account": self.account1.id,
            "receiver_account": self.account2.id,
            "amount": "150000.00"
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # 3. Transaction 3: account2 -> account1, amount 150000.00 on new account age
        # Let's temporarily make account2 a new account
        Account.objects.filter(id=self.account2.id).update(created_at=timezone.now())
        self.account2.refresh_from_db()
        # Amount 150000.00 + new account -> risk_score = 60 (BLOCKED)
        response = self.client.post(url_transactions, {
            "sender_account": self.account2.id,
            "receiver_account": self.account1.id,
            "amount": "150000.00"
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Total checks:
        # Check 1: risk 0, APPROVED
        # Check 2: risk 40, REVIEW
        # Check 3: risk 60, BLOCKED
        # Average risk score: (0 + 40 + 60) / 3 = 33.33
        # Blocked count: 1
        # Bad transactions count:
        # account1 has 1 (Transaction 2 is REVIEW)
        # account2 has 1 (Transaction 3 is BLOCKED)

        url_analytics = "/api/fraud-checks/analytics/"
        response = self.client.get(url_analytics)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(response.data["blocked_transactions_count"], 1)
        self.assertEqual(response.data["average_risk_score"], 33.33)
        self.assertEqual(len(response.data["top_flagged_accounts"]), 2)

        # Verify top flagged accounts list
        flagged_ids = [acc["id"] for acc in response.data["top_flagged_accounts"]]
        self.assertIn(self.account1.id, flagged_ids)
        self.assertIn(self.account2.id, flagged_ids)



