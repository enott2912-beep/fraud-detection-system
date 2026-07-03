from django.apps import AppConfig, apps


class FraudChecksConfig(AppConfig):
    name = "fraud_checks"

    def ready(self):
        from ml.inference import load_model

        self.ml_model = load_model()
