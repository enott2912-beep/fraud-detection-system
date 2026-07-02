"""
Train an ML model (RandomForestClassifier) on the synthetic dataset.
Hyperparameters and threshold are fixed from rf_tuning.ipynb tuning on val set.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "synthetic_v5.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "fraud_model_v5.pkl"

FEATURE_COLS = [
    "amount", "account_age_days", "tx_last_hour", "transaction_hour",
    "receiver_tx_count_24h", "sender_daily_amount_sum",
    "amount_to_balance_ratio", "days_since_last_tx",
]
LABEL_COL = "is_fraud"

BEST_THRESHOLD = 0.59  # Known limitation: balance_drain_fraud recall ~32% at this threshold.
                         # Root cause: amount_to_balance_ratio overlaps with normal population
                         # (see eda_feature_distributions.ipynb). Not resolved — out of scope.
                         # NOTE: RandomForest results vary between sklearn minor versions even
                         # with fixed random_state. Pinned to sklearn==1.9.0 in requirements.txt.

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
}


def load_data():
    df = pd.read_csv(DATASET_PATH, parse_dates=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def time_based_split(df, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, df_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= BEST_THRESHOLD).astype(int)

    print(f"=== Classification report (threshold={BEST_THRESHOLD}) ===")
    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))

    print("=== Confusion matrix ===")
    print("       pred_normal  pred_fraud")
    cm = confusion_matrix(y_test, y_pred)
    print(f"normal      {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"fraud       {cm[1][0]:>6}      {cm[1][1]:>6}")

    print()
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    print()
    print("=== Per-pattern recall (test) ===")
    test_local = df_test.copy()
    test_local["pred"] = y_pred
    h = f"{'Pattern':30s}  {'n (test)':>8s}  {'pred_fraud':>10s}  {'recall (test)':>12s}"
    print(h)
    print("-" * len(h))
    for pattern in sorted(df_test["fraud_pattern"].dropna().unique()):
        mask = test_local["fraud_pattern"] == pattern
        p = test_local[mask]
        rec = recall_score(p[LABEL_COL], p["pred"]) if len(p) > 0 else 0.0
        print(f"{pattern:30s}  {len(p):6d}  {int(p['pred'].sum()):10d}  {rec:6.3f}")

    print()
    print("=== Feature importance ===")
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(importances)


def main():
    df = load_data()
    train_df, test_df = time_based_split(df)

    print(f"Train: {len(train_df)} lines ({train_df[LABEL_COL].mean():.2%} fraud)")
    print(f"Test:  {len(test_df)} lines ({test_df[LABEL_COL].mean():.2%} fraud)")
    print()

    X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[LABEL_COL]

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, test_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print()
    print(f"Model saved in {MODEL_PATH}")


if __name__ == "__main__":
    main()
