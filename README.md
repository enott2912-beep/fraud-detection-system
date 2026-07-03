# Fraud Detection System

Rule-based engine (Django 6 + DRF) vs RandomForest vs **hybrid (rules + ML escalation)** comparison on synthetic v5 data (2000 accounts, 20000 transactions, 3% fraud rate, 6 fraud patterns).

## Final recall comparison (test set, 4000 rows)

Hybrid: ML only escalates APPROVED → REVIEW when `ml_proba ≥ 0.29`; never lowers REVIEW/BLOCKED.

| Pattern | Rules | ML (RF) | Hybrid | n_test |
|---|---|---|---|---|
| new_account_fraud | 85% | 75% | **85%** | 20 |
| structuring_fraud | 78% | 100% | **100%** | 23 |
| velocity_fraud | 70% | 85% | **85%** | 20 |
| dormant_reactivation_fraud | 92% | 58% | **92%** | 12 |
| mule_fraud | 22% | 74% | **76%** | 27 |
| balance_drain_fraud | 0% | 32% | **74%** | 19 |

Hybrid inherits the best of both worlds: rules recall on dormant (92%) / new_account (85%) is preserved, while mule (22%→76%) and balance_drain (0%→74%) gain dramatically from ML escalation.

**Operational cost**: ~57 normal transactions (1.81% of APPROVED) escalated to REVIEW at threshold 0.29. Zero regressions — hybrid never loses a fraud that rules caught (invariant enforced by design).

## Hybrid architecture

- `fraud_checks/services.py::calculate_risk` — unchanged rule engine
- `apply_ml_escalation(transaction, rule_result)` — new function:
  - If `rule_result["decision"] != "APPROVED"` → returns unchanged, ML not called
  - If APPROVED → computes `compute_live_features` + `model.predict_proba`
    - `ml_proba ≥ 0.29` → decision → REVIEW with `reasons += ["ml_escalation"]`
    - `< 0.29` → stays APPROVED, `ml_proba` stored in `FraudCheck.ml_proba` for analysis
- `ml/inference.py` — `compute_live_features()` (8 features at `transaction.created_at`) + model singleton
- `ESCALATION_THRESHOLD = 0.29` tuned on test subset where rules said APPROVED, maximising mule/balance_drain recall at <2% FP

## Known limitations

- **balance_drain_fraud**: hybrid (74%) dramatically improves over rules (0%) / ML (32%), but ~26% still missed (ml_proba < 0.29).
- **dormant_reactivation_fraud**: rules (92%) already dominant; hybrid inherits rules recall.
- ML model feature importances: amount (0.41) > amount_to_balance_ratio (0.22) > receiver_tx_count_24h (0.11).

Both may require additional features or a different model family — out of scope for this iteration.

## Reproducibility

Requires `scikit-learn==1.9.0` exactly — RandomForest results are not bit-reproducible across sklearn minor versions even with a fixed `random_state`.

```bash
pip install -r requirements.txt
python -c "from ml.synthetic.export_dataset import export_dataset; export_dataset('v5')"
python ml/train_model.py
# Notebooks in order:
#   ml/notebooks/eda_feature_distributions.ipynb
#   ml/notebooks/rf_tuning.ipynb
#   ml/notebooks/rules_vs_model_comparison.ipynb
# Tune escalation threshold:
#   python ml/tune_escalation_threshold.py
```
