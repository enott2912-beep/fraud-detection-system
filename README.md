# Fraud Detection System

Rule-based engine (Django 6 + DRF) vs RandomForest comparison on synthetic v5 data (2000 accounts, 20000 transactions, 3% fraud rate, 6 fraud patterns).

## Final recall comparison (test set, 4000 rows, threshold=0.59)

| Pattern | Rules | ML (RF) | n_test |
|---|---|---|---|
| new_account_fraud | 85% | 75% | 20 |
| structuring_fraud | 78% | 100% | 23 |
| velocity_fraud | 70% | 85% | 20 |
| dormant_reactivation_fraud | 92% | 58% | 12 |
| mule_fraud | 22% | 74% | 27 |
| balance_drain_fraud | 0% | 32% | 19 |

ML catches 31 frauds that rules miss; rules catch 9 that ML misses; 24 missed by both.

## Known limitation

`balance_drain_fraud` is a blind spot for both systems. The feature `amount_to_balance_ratio` overlaps heavily with normal transactions. Resolving this needs additional features or a different model family — out of scope for this iteration.

## Reproducibility

```bash
python -c "from ml.synthetic.export_dataset import export_dataset; export_dataset('v5')"
python ml/train_model.py
# Notebooks in order:
#   ml/notebooks/eda_feature_distributions.ipynb
#   ml/notebooks/rf_tuning.ipynb
#   ml/notebooks/rules_vs_model_comparison.ipynb
```
