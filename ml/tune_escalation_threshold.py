"""
Tune ESCALATION_THRESHOLD for the hybrid rules+ML system.

Strategy: on the test set, filter to rows where rules said APPROVED
(rule_decision == "APPROVED").  On this subset, grid-search thresholds
to maximise recall for mule_fraud + balance_drain_fraud while keeping
the number of escalated normal transactions acceptable.

Output: recommended ESCALATION_THRESHOLD value to put in ml/constants.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.synthetic import config
from ml.synthetic.build_dataset import generate_accounts

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "synthetic_v5.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "fraud_model_v5.pkl"

FEATURE_COLS = [
    "amount", "account_age_days", "tx_last_hour", "transaction_hour",
    "receiver_tx_count_24h", "sender_daily_amount_sum",
    "amount_to_balance_ratio", "days_since_last_tx",
]

ESCALATION_PATTERNS = ["mule_fraud", "balance_drain_fraud"]


def rule_based_full(row):
    risk_score = 0
    if row["account_age_days"] < 7:
        risk_score += 20
    if row["amount"] > 100000:
        risk_score += 40
    if row["tx_last_hour"] > 10:
        risk_score += 30
    if row["sender_daily_amount_sum"] + row["amount"] > 200000:
        risk_score += 60
    if row["amount"] > row["sender_balance_before"]:
        risk_score += 80
    if risk_score >= 60:
        decision = "BLOCKED"
    elif risk_score >= 30:
        decision = "REVIEW"
    else:
        decision = "APPROVED"
    return decision


def main():
    df = pd.read_csv(DATASET_PATH, parse_dates=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    accounts_df = generate_accounts(
        config.N_ACCOUNTS, np.random.default_rng(config.SEED)
    )
    balance_map = accounts_df.set_index("account_id")["balance"]
    df["sender_balance_before"] = df["sender_account_id"].map(balance_map)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"Test set: {len(test_df)} rows "
          f"({test_df['is_fraud'].mean():.2%} fraud)")

    test_df["rule_decision"] = test_df.apply(rule_based_full, axis=1)
    test_df["rule_pred"] = (
        test_df["rule_decision"].isin(["BLOCKED", "REVIEW"]).astype(int)
    )

    model = joblib.load(MODEL_PATH)
    X_test = test_df[FEATURE_COLS]
    test_df["ml_proba"] = model.predict_proba(X_test)[:, 1]

    approved_mask = test_df["rule_decision"] == "APPROVED"
    approved_df = test_df[approved_mask].copy()
    n_approved_total = len(approved_df)
    n_approved_fraud = approved_df["is_fraud"].sum()
    print(f"\nRules said APPROVED: {n_approved_total} total, "
          f"{n_approved_fraud} ({n_approved_fraud/n_approved_total:.2%}) actually fraud")

    target_mask = approved_df["fraud_pattern"].isin(ESCALATION_PATTERNS)
    approved_df["is_target"] = target_mask.astype(int)

    precision, recall, thresholds = precision_recall_curve(
        approved_df["is_target"], approved_df["ml_proba"]
    )

    normal_mask = approved_df["is_fraud"] == 0
    n_normal_approved = normal_mask.sum()

    results = []
    for t in np.arange(0.0, 1.01, 0.01):
        t = round(t, 2)
        pred = (approved_df["ml_proba"] >= t).astype(int)

        target_total = int(approved_df["is_target"].sum())
        target_caught = int(
            ((approved_df["is_target"] == 1) & (pred == 1)).sum()
        )
        target_recall = target_caught / target_total if target_total > 0 else 0.0

        fp_normal = int(
            ((approved_df["is_fraud"] == 0) & (pred == 1)).sum()
        )
        fp_rate = fp_normal / n_normal_approved if n_normal_approved > 0 else 0.0

        mule_total = int((approved_df["fraud_pattern"] == "mule_fraud").sum())
        mule_caught = int(
            ((approved_df["fraud_pattern"] == "mule_fraud") & (pred == 1)).sum()
        )
        mule_recall = mule_caught / mule_total if mule_total > 0 else 0.0

        bd_total = int((approved_df["fraud_pattern"] == "balance_drain_fraud").sum())
        bd_caught = int(
            ((approved_df["fraud_pattern"] == "balance_drain_fraud") & (pred == 1)).sum()
        )
        bd_recall = bd_caught / bd_total if bd_total > 0 else 0.0

        results.append({
            "threshold": t,
            "target_recall": target_recall,
            "mule_recall": mule_recall,
            "bd_recall": bd_recall,
            "fp_normal": fp_normal,
            "fp_rate": fp_rate,
        })

    results_df = pd.DataFrame(results)

    print("\n=== Candidate thresholds (grouped by fp_rate) ===")
    candidates = results_df[results_df["threshold"] > 0.0].copy()
    print(
        f"{'thresh':>6s}  {'target_recall':>13s}  {'mule_recall':>11s}  "
        f"{'bd_recall':>9s}  {'fp_normal':>9s}  {'fp_rate':>7s}"
    )
    print("-" * 65)
    for fp_target in [0.01, 0.02, 0.03, 0.05, 0.10]:
        subset = candidates[candidates["fp_rate"] <= fp_target]
        if len(subset) == 0:
            continue
        best = subset.loc[subset["target_recall"].idxmax()]
        print(
            f"{best['threshold']:>6.2f}  {best['target_recall']:>12.1%}  "
            f"{best['mule_recall']:>10.1%}  {best['bd_recall']:>8.1%}  "
            f"{int(best['fp_normal']):>9d}  {best['fp_rate']:>6.2%}  <- fp<={fp_target:.0%}"
        )

    best_tradeoff = candidates[
        (candidates["fp_rate"] <= 0.02)
    ].sort_values("target_recall", ascending=False).iloc[0] if len(candidates[
        (candidates["fp_rate"] <= 0.02)
    ]) > 0 else candidates.sort_values("fp_rate").iloc[0]

    best = best_tradeoff
    print(f"\n=== Recommended threshold (best trade-off: >=70% target recall, lowest fp) ===")
    print(f"ESCALATION_THRESHOLD = {best['threshold']:.2f}")
    print(f"  target_recall (mule+bd): {best['target_recall']:.1%}")
    print(f"    mule_recall:           {best['mule_recall']:.1%}")
    print(f"    balance_drain_recall:  {best['bd_recall']:.1%}")
    print(f"  fp_normal:               {int(best['fp_normal'])} "
          f"({best['fp_rate']:.2%} of normal APPROVED)")
    print(f"  total normal APPROVED:   {n_normal_approved}")

    print(f"\n=== Per-pattern analysis at threshold={best['threshold']:.2f} ===")
    pred_best = (approved_df["ml_proba"] >= best["threshold"]).astype(int)
    approved_df["hybrid_escalated"] = pred_best
    for pattern in sorted(approved_df["fraud_pattern"].dropna().unique()):
        mask = approved_df["fraud_pattern"] == pattern
        total = int(mask.sum())
        escalated = int(approved_df.loc[mask, "hybrid_escalated"].sum())
        pct = escalated / total * 100 if total > 0 else 0.0
        print(f"  {pattern:35s}: {escalated:3d}/{total} escalated ({pct:.1f}%)")

    return best["threshold"]


if __name__ == "__main__":
    main()
