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

Disagreement: both caught 57, ML better 31, rules better 9, both missed 24 (sum=121 ✓).

## Known limitations

- **balance_drain_fraud**: blind spot for both systems — rules (0%) and ML (32% on sklearn 1.9.0). Recall varies 11–32% between sklearn 1.8.0 and 1.9.0 with identical code/data/`random_state=42`. Root cause: `amount_to_balance_ratio` overlaps with normal transactions (see `eda_feature_distributions.ipynb`).
- **dormant_reactivation_fraud**: ML (58%) significantly behind rules (92%) — rules catch it accidentally via the `new_account` heuristic (<7 days) which overlaps with dormant reactivation profiles.

Both require additional features or a different model family — out of scope for this iteration.

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
```
