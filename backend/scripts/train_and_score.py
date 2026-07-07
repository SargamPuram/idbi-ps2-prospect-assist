"""
Training + batch-scoring pipeline for PS2 Prospect Assist AI.

1. Loads data/customers.csv + data/monthly_transactions.csv
2. Engineers behavioral features (scoring.models.engineer_features) — purely from raw
   transactions/static fields, no ground-truth flags.
3. Trains a calibrated XGBoost classifier for the INTENT score (target: actually_took_loan,
   using only intent-related features) and reports held-out AUC.
4. Trains a small logistic-regression "ML adjustment" used to nudge the rules-based
   CAPACITY score, per the spec's "Rules engine + ML adjustment" design.
5. Computes Capacity (rules+ML) and Propensity (rule-based pattern detection) scores.
6. Computes the composite lead score (0.40 Intent + 0.35 Capacity + 0.25 Propensity) and
   RAG classification (Hot >75 / Warm 50-75 / Cold <50).
7. Computes a product recommendation for every customer (same function the live
   POST /recommend/{id} endpoint calls) and writes everything to data/scored_customers.csv.
8. Saves models/intent_model.joblib and data/model_performance.json for the /analytics API.

Run from the backend/ directory:
    python scripts/train_and_score.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.models import (
    INTENT_FEATURE_COLUMNS, build_talking_points, compute_capacity_score,
    compute_composite, compute_propensity_score, engineer_features,
    intent_score_from_proba, prepare_intent_matrix, rag_status, recommend_product,
)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BEST_CONTACT_TIME = {
    "IT Professional": "Weekday evenings, 7-9 PM",
    "Government Employee": "Lunch hour, 1-2 PM",
    "Doctor": "Weekend mornings, 10-12 PM",
    "Business Owner": "Weekday mornings, 10-12 PM",
    "Teacher": "After school hours, 4-6 PM",
    "Engineer": "Weekday evenings, 6-8 PM",
    "CA": "Weekday mornings, 9-11 AM",
    "Freelancer": "Weekday afternoons, 2-5 PM",
    "Gig Worker": "Evenings, 6-9 PM",
}


def main():
    t0 = time.time()
    print("[train_and_score] Loading data...")
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    monthly = pd.read_csv(os.path.join(DATA_DIR, "monthly_transactions.csv"))
    print(f"  customers: {customers.shape}, monthly: {monthly.shape}")

    print("[train_and_score] Engineering behavioral features...")
    features = engineer_features(customers, monthly)
    assert (features["customer_id"].values == customers["customer_id"].values).all(), "row order mismatch"

    y = customers["actually_took_loan"].astype(int).to_numpy()

    # ---------------------------------------------------------------- INTENT model (XGBoost, calibrated)
    print("[train_and_score] Training Intent classifier (XGBoost, gradient boosted)...")
    X_intent = prepare_intent_matrix(features)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_intent, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )

    base_clf = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=5, reg_lambda=2.0,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc", random_state=42, n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    test_proba = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    ap = average_precision_score(y_test, test_proba)
    preds_at_50 = (test_proba >= 0.5).astype(int)
    precision = precision_score(y_test, preds_at_50, zero_division=0)
    recall = recall_score(y_test, preds_at_50, zero_division=0)
    f1 = f1_score(y_test, preds_at_50, zero_division=0)
    acc = accuracy_score(y_test, preds_at_50)
    cm = confusion_matrix(y_test, preds_at_50).tolist()
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    roc_points = [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in
                  zip(fpr[::max(1, len(fpr) // 60)], tpr[::max(1, len(tpr) // 60)])]

    print(f"  Intent model held-out AUC: {auc:.4f} | PR-AUC: {ap:.4f} | F1@0.5: {f1:.4f}")

    # refit on full data for production scoring (still same architecture/calibration)
    final_model = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    final_model.fit(X_intent, y)
    full_proba = final_model.predict_proba(X_intent)[:, 1]
    intent_score, intent_lo, intent_hi = intent_score_from_proba(full_proba)

    joblib.dump(final_model, os.path.join(MODELS_DIR, "intent_model.joblib"))

    feature_importance = None
    try:
        raw_importances = base_clf.fit(X_intent, y).feature_importances_
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(INTENT_FEATURE_COLUMNS, raw_importances)],
            key=lambda d: -d["importance"],
        )
    except Exception as e:
        print(f"  (feature importance skipped: {e})")

    # ---------------------------------------------------------------- CAPACITY: rules + small ML adjustment
    print("[train_and_score] Training Capacity ML-adjustment model (logistic regression)...")
    capacity_ml_features = features[["foir", "disposable_income", "estimated_monthly_income", "income_regularity_score"]].copy()
    capacity_ml_features["disposable_income"] = capacity_ml_features["disposable_income"] / 100000.0
    capacity_ml_features["estimated_monthly_income"] = capacity_ml_features["estimated_monthly_income"] / 100000.0
    cap_logreg = LogisticRegression(max_iter=500, class_weight="balanced")
    cap_logreg.fit(capacity_ml_features, y)
    cap_ml_proba = cap_logreg.predict_proba(capacity_ml_features)[:, 1]
    cap_ml_adjustment = cap_ml_proba * 100
    joblib.dump(cap_logreg, os.path.join(MODELS_DIR, "capacity_adjustment_model.joblib"))

    capacity_score = compute_capacity_score(features, customers, ml_adjustment=cap_ml_adjustment)

    # ---------------------------------------------------------------- PROPENSITY: rule-based
    print("[train_and_score] Computing Propensity scores (rule-based life-event detection)...")
    propensity_score, detected_events_list = compute_propensity_score(features)

    # ---------------------------------------------------------------- COMPOSITE + RAG
    composite = compute_composite(intent_score, capacity_score, propensity_score)
    rag = rag_status(composite)

    # ---------------------------------------------------------------- Product recommendation (per customer)
    print("[train_and_score] Computing product recommendations for all customers...")
    recs = []
    talking_points_list = []
    for i in range(len(customers)):
        row_feats = features.iloc[i]
        row_cust = customers.iloc[i]
        rec = recommend_product(row_feats, row_cust)
        recs.append(rec)
        talking_points_list.append(build_talking_points(row_feats, row_cust, detected_events_list[i], rec))

    recommended_product = [r["primary"]["loan_type"] for r in recs]
    recommended_amount = [r["primary"]["suggested_amount"] for r in recs]
    recommended_tenure = [r["primary"]["suggested_tenure_months"] for r in recs]
    recommended_emi = [r["primary"]["estimated_emi"] for r in recs]
    recommendation_confidence = [r["primary"]["confidence"] for r in recs]

    best_contact_time = customers["occupation"].map(BEST_CONTACT_TIME).fillna("Weekday afternoons, 2-4 PM")

    rng = np.random.default_rng(7)
    status_roll = rng.random(len(customers))
    status = np.where(
        rag == "Hot", np.where(status_roll < 0.35, "Scheduled", np.where(status_roll < 0.55, "Contacted", "Pending")),
        np.where(rag == "Warm", np.where(status_roll < 0.25, "Contacted", "Pending"), "Pending"),
    )

    print("[train_and_score] Assembling scored_customers.csv...")
    scored = customers[[
        "customer_id", "name", "age", "gender", "city", "occupation", "employer_name",
        "employment_type", "account_type", "account_tenure_years", "avg_monthly_balance",
        "app_login_frequency", "loan_page_visits", "loan_calculator_usage", "time_on_loan_pages",
        "products_viewed", "application_started_not_completed", "last_visit_days_ago",
        "last_visit_timestamp", "credit_score", "existing_loan_count", "total_emi_burden",
        "credit_utilization", "payment_history_score", "actually_took_loan", "loan_type_taken",
        "loan_amount_taken", "days_to_conversion",
    ]].copy()

    scored["intent_score"] = intent_score
    scored["capacity_score"] = capacity_score
    scored["propensity_score"] = propensity_score
    scored["lead_score"] = composite
    scored["lead_status"] = rag
    scored["estimated_monthly_income"] = np.round(features["estimated_monthly_income"], 2)
    scored["income_method"] = features["income_method"]
    scored["foir"] = features["foir"]
    scored["disposable_income"] = features["disposable_income"]
    scored["recommended_product"] = recommended_product
    scored["recommended_amount"] = recommended_amount
    scored["recommended_tenure_months"] = recommended_tenure
    scored["recommended_emi"] = recommended_emi
    scored["recommendation_confidence"] = recommendation_confidence
    scored["best_contact_time"] = best_contact_time
    scored["contact_status"] = status
    scored["detected_events_count"] = [len(e) for e in detected_events_list]
    scored["detected_events_json"] = [json.dumps(e) for e in detected_events_list]

    scored_path = os.path.join(DATA_DIR, "scored_customers.csv")
    scored.to_csv(scored_path, index=False)
    print(f"  Wrote {len(scored):,} scored customers -> {scored_path}")

    print("[train_and_score] Distribution summary:")
    print(scored["lead_status"].value_counts())
    print(f"  Mean lead_score: {scored['lead_score'].mean():.1f}")
    print(f"  Predicted conversion rate proxy (Hot conv-weighted): see /dashboard endpoint")

    performance = {
        "intent_model": {
            "algorithm": "XGBoost (gradient boosted trees) + Platt sigmoid calibration",
            "target": "actually_took_loan",
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "auc_roc": round(float(auc), 4),
            "pr_auc": round(float(ap), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["Actual Negative", "Actual Positive"],
            "roc_curve": roc_points,
            "feature_importance": feature_importance,
        },
        "capacity_model": {
            "algorithm": "Rules engine (FOIR, disposable income, credit score, payment history, utilization) "
                         "+ 15% weighted logistic-regression ML adjustment",
        },
        "propensity_model": {
            "algorithm": "Rule-based life-event pattern detection (7 signals) on 6-month transaction time series",
        },
        "intent_score_calibration": {"lo": round(intent_lo, 6), "hi": round(intent_hi, 6)},
        "composite_formula": "0.40 * Intent + 0.35 * Capacity + 0.25 * Propensity",
        "rag_thresholds": {"hot": "> 75", "warm": "50 - 75", "cold": "< 50"},
        "generated_customers": int(len(customers)),
        "actual_conversion_rate_baseline": round(float(y.mean()), 4),
    }
    with open(os.path.join(DATA_DIR, "model_performance.json"), "w") as f:
        json.dump(performance, f, indent=2)

    print(f"[train_and_score] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
