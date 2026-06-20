from django.db.models import Avg, Count, Q
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from accounts.models import Account
from .models import FraudCheck
from .serializers import FraudCheckSerializer


class FraudCheckViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = FraudCheck.objects.all()
    serializer_class = FraudCheckSerializer

    @action(detail=False, methods=['get'], url_path='analytics')
    def analytics(self, request):
        blocked_count = FraudCheck.objects.filter(decision="BLOCKED").count()
        avg_risk_score = FraudCheck.objects.aggregate(avg=Avg('risk_score'))['avg'] or 0.0
        
        top_accounts_qs = Account.objects.annotate(
            bad_tx_count=Count(
                'outgoing_transactions',
                filter=Q(outgoing_transactions__fraud_check__decision__in=["BLOCKED", "REVIEW"])
            )
        ).filter(bad_tx_count__gt=0).order_by('-bad_tx_count')[:5]
        
        top_accounts_data = [
            {
                "id": acc.id,
                "account_number": acc.account_number,
                "owner": acc.owner.username,
                "blocked_or_review_count": acc.bad_tx_count
            }
            for acc in top_accounts_qs
        ]
        
        return Response({
            "blocked_transactions_count": blocked_count,
            "average_risk_score": round(avg_risk_score, 2),
            "top_flagged_accounts": top_accounts_data
        })


