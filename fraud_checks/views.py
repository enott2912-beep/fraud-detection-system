from rest_framework import viewsets
from .models import FraudCheck
from .serializers import FraudCheckSerializer


class FraudCheckViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = FraudCheck.objects.all()
    serializer_class = FraudCheckSerializer

