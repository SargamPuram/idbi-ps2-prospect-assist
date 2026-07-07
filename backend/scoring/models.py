"""
Shared scoring logic for PS2 Prospect Assist AI.

This module is imported by BOTH:
  - scripts/train_and_score.py  (offline: trains the Intent classifier, batch-scores
    all 20,000 customers, writes data/scored_customers.csv + models/intent_model.joblib)
  - app/main.py                 (live serving: recomputes recommendations / re-derives
    behavioral insights on demand for a single customer via the exact same functions)

so a batch-scored customer and a live-recomputed one are always consistent.

Design note on leakage: `engineer_features()` derives every signal purely from the
raw monthly transaction time series and static customer fields — it never reads the
"_true_*" ground-truth columns injected by scripts/generate_data.py. Life events are
genuinely *detected* from the numbers (recent vs. earlier period comparisons), not
looked up from a hidden flag, mirroring what would be available for a real customer.
"""

import numpy as np
import pandas as pd

from scoring.constants import (
    COMPOSITE_WEIGHTS, EMI_CATEGORIES, INVESTMENT_CATEGORIES, LOAN_AMOUNT_CEILING,
    LOAN_INCOME_MULTIPLE, LOAN_INTEREST_RATES, LOAN_TENURE_MONTHS, LOAN_TYPES,
    LUXURY_CATEGORIES, NEEDS_CATEGORIES, N_MONTHS, RAG_HOT_THRESHOLD, RAG_WARM_THRESHOLD,
    UPI_CATEGORIES, WANTS_CATEGORIES,
)

INTENT_FEATURE_COLUMNS = [
    "loan_page_visits", "loan_calculator_usage", "time_on_loan_pages",
    "application_started_not_completed", "products_viewed_diversity", "last_visit_days_ago",
]


# --------------------------------------------------------------------------- feature engineering

def _pivot(monthly: pd.DataFrame, col: str, customer_order: pd.Index) -> np.ndarray:
    """Pivot a monthly long column into an (N, 6) array ordered to match customer_order."""
    p = monthly.pivot(index="customer_id", columns="month", values=col)
    p = p.reindex(customer_order)
    p = p.reindex(columns=range(1, N_MONTHS + 1))
    return p.to_numpy(dtype=float)


def engineer_features(customers: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-customer behavioral feature table used by all three scores.
    Fully vectorized: every customer has exactly N_MONTHS rows (a regular grid), so we
    pivot each metric into an (N, 6) matrix instead of a slow per-group python loop.
    """
    customers = customers.set_index("customer_id", drop=False)
    order = customers.index

    salary = _pivot(monthly, "salary_credit", order)
    other_income = _pivot(monthly, "other_income_credit", order)
    cat = {c: _pivot(monthly, f"upi_{c}", order) for c in UPI_CATEGORIES}

    feats = pd.DataFrame(index=order)

    # ---- Income estimation (behavioral, income-source-aware) ----
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        salary_nz = np.where(salary > 0, salary, np.nan)
        salary_mean = np.nanmean(salary_nz, axis=1)
        salary_mean = np.nan_to_num(salary_mean, nan=0.0)

        other_mean = other_income.mean(axis=1)
        other_std = other_income.std(axis=1)
        other_cv = np.where(other_mean > 1, other_std / np.maximum(other_mean, 1), 1.0)

        sorted_other = np.sort(other_income, axis=1)
        trimmed_mean = sorted_other[:, 1:-1].mean(axis=1)  # drop lowest & highest month

    employment_type = customers["employment_type"].to_numpy()
    estimated_income = np.where(
        employment_type == "Salaried", salary_mean,
        np.where(employment_type == "Self-employed", other_mean, trimmed_mean),
    )
    estimated_income = np.maximum(estimated_income, 5000)
    income_method = np.where(
        employment_type == "Salaried", "Salary-based",
        np.where(employment_type == "Self-employed", "Transaction-based", "Pattern-based"),
    )
    regularity_score = np.clip(1 - np.clip(other_cv, 0, 1), 0, 1)
    regularity_score = np.where(employment_type == "Salaried", 0.97, regularity_score)

    feats["estimated_monthly_income"] = estimated_income
    feats["income_method"] = income_method
    feats["income_regularity_score"] = np.round(regularity_score, 3)

    # ---- Expense ratios ----
    cat_mean = {c: cat[c].mean(axis=1) for c in UPI_CATEGORIES}
    needs_total = sum(cat_mean[c] for c in NEEDS_CATEGORIES)
    wants_total = sum(cat_mean[c] for c in WANTS_CATEGORIES)
    luxury_total = sum(cat_mean[c] for c in LUXURY_CATEGORIES)
    emi_total_txn = sum(cat_mean[c] for c in EMI_CATEGORIES)
    invest_total = sum(cat_mean[c] for c in INVESTMENT_CATEGORIES)
    outflow_total = needs_total + wants_total + luxury_total + emi_total_txn + invest_total
    savings_total = np.clip(estimated_income - outflow_total, 0, None)

    denom = np.maximum(estimated_income, 1)
    feats["needs_ratio"] = np.round(needs_total / denom, 4)
    feats["wants_ratio"] = np.round(wants_total / denom, 4)
    feats["luxury_ratio"] = np.round(luxury_total / denom, 4)
    feats["emi_ratio_txn"] = np.round(emi_total_txn / denom, 4)
    feats["investment_ratio"] = np.round(invest_total / denom, 4)
    feats["savings_ratio"] = np.round(savings_total / denom, 4)
    for c in UPI_CATEGORIES:
        feats[f"avg_{c}"] = np.round(cat_mean[c], 2)

    # ---- Capacity primitives ----
    total_emi_burden = customers["total_emi_burden"].to_numpy()
    foir = total_emi_burden / denom
    disposable_income = np.clip(estimated_income - needs_total - total_emi_burden, 0, None)
    feats["foir"] = np.round(np.clip(foir, 0, 3), 4)
    feats["disposable_income"] = np.round(disposable_income, 2)
    feats["capacity_for_new_emi"] = np.round(disposable_income * 0.5, 2)

    # ---- Life-event detection purely from the transaction time series ----
    first2 = slice(0, 2)
    last2 = slice(4, 6)
    first3 = slice(0, 3)
    last3 = slice(3, 6)

    salary_first2 = salary[:, first2].mean(axis=1)
    salary_last2 = salary[:, last2].mean(axis=1)
    salary_hike_pct = np.where(salary_first2 > 1000, (salary_last2 - salary_first2) / np.maximum(salary_first2, 1) * 100, 0.0)
    salary_hike_detected = (employment_type == "Salaried") & (salary_hike_pct > 15)

    rent_first2 = cat["rent"][:, first2].mean(axis=1)
    rent_last2 = cat["rent"][:, last2].mean(axis=1)
    new_rent_detected = (rent_first2 < 500) & (rent_last2 > 3000)

    edu_first3 = cat["education"][:, first3].mean(axis=1)
    edu_last3 = cat["education"][:, last3].mean(axis=1)
    education_spend_increase = (edu_last3 > 1.4 * np.maximum(edu_first3, 1)) & (edu_last3 > 800)

    med_first3_mean = cat["medical"][:, first3].mean(axis=1)
    med_last3_max = cat["medical"][:, last3].max(axis=1)
    medical_expenses_spike = (med_last3_max > 2.5 * np.maximum(med_first3_mean, 1)) & (med_last3_max > 1500)

    marriage_signal = cat["shopping"] + cat["entertainment"]
    marriage_month_max = marriage_signal.max(axis=1)
    marriage_month_sum = marriage_signal.sum(axis=1)
    marriage_other_mean = np.maximum((marriage_month_sum - marriage_month_max) / (N_MONTHS - 1), 1)
    marriage_indicators = (marriage_month_max > 3.5 * marriage_other_mean) & (marriage_month_max > 40000)

    other_income_max = other_income.max(axis=1)
    other_income_sum = other_income.sum(axis=1)
    other_income_rest_mean = np.maximum((other_income_sum - other_income_max) / (N_MONTHS - 1), 1)
    investment_maturity = (other_income_max > 4 * other_income_rest_mean) & (other_income_max > 50000)

    ins_first3 = cat["insurance_premium"][:, first3].mean(axis=1)
    ins_last2 = cat["insurance_premium"][:, last2].mean(axis=1)
    vehicle_insurance_lapse = (ins_first3 > 300) & (ins_last2 < 50)

    feats["salary_hike_detected"] = salary_hike_detected
    feats["salary_hike_pct"] = np.round(salary_hike_pct, 1)
    feats["new_rent_detected"] = new_rent_detected
    feats["education_spend_increase"] = education_spend_increase
    feats["medical_expenses_spike"] = medical_expenses_spike
    feats["marriage_indicators"] = marriage_indicators
    feats["investment_maturity"] = investment_maturity
    feats["vehicle_insurance_lapse"] = vehicle_insurance_lapse

    # ---- carry through raw intent inputs & static fields needed downstream ----
    for col in INTENT_FEATURE_COLUMNS:
        feats[col] = customers[col].to_numpy()
    feats["customer_id"] = order
    feats.reset_index(drop=True, inplace=True)
    return feats


# --------------------------------------------------------------------------- Intent score

def prepare_intent_matrix(features: pd.DataFrame) -> pd.DataFrame:
    X = features[INTENT_FEATURE_COLUMNS].copy()
    X["application_started_not_completed"] = X["application_started_not_completed"].astype(int)
    return X


def rescale_proba_to_score(proba: np.ndarray, lo: float = None, hi: float = None):
    """
    Convert a rare-event probability (e.g. from a classifier trained on a ~5% positive
    target) into a 0-100 score that actually spreads across the range.

    Raw probabilities from an imbalanced-target model cluster tightly around the base
    rate, so a direct proba*100 mapping would squash almost every customer near single
    digits. Instead we robust-min-max rescale against the 1st/99th percentile of the
    scored population — a standard lead-scoring technique — so scores spread meaningfully
    across 0-100 while remaining strictly monotonic in the underlying probability (ranking
    is unaffected). `lo`/`hi` can be supplied (from a previous run) to score new customers
    on the exact same scale; otherwise they are derived from `proba` itself.
    """
    if lo is None or hi is None:
        lo, hi = float(np.percentile(proba, 1)), float(np.percentile(proba, 99))
    span = max(hi - lo, 1e-9)
    score = np.clip((proba - lo) / span * 100, 0, 100)
    return np.round(score, 1), lo, hi


def intent_score_from_proba(proba: np.ndarray, lo: float = None, hi: float = None):
    """Convert calibrated Intent-model conversion probabilities into a 0-100 Intent score."""
    return rescale_proba_to_score(proba, lo, hi)


# --------------------------------------------------------------------------- Capacity score

def compute_capacity_score(features: pd.DataFrame, customers: pd.DataFrame, ml_adjustment: np.ndarray = None) -> np.ndarray:
    foir = features["foir"].to_numpy()
    disposable_ratio = np.clip(
        features["disposable_income"].to_numpy() / np.maximum(features["estimated_monthly_income"].to_numpy(), 1), 0, 1
    )
    credit_score = customers["credit_score"].to_numpy()
    payment_history = customers["payment_history_score"].to_numpy()
    utilization = customers["credit_utilization"].to_numpy()

    score_foir = 100 * (1 - np.clip(foir, 0, 1))
    score_disp = 100 * disposable_ratio
    score_credit = (credit_score - 300) / (900 - 300) * 100
    score_payment = payment_history
    score_util = 100 * (1 - np.clip(utilization / 100, 0, 1))

    capacity_raw = (
        0.30 * score_foir + 0.25 * score_disp + 0.20 * score_credit
        + 0.15 * score_payment + 0.10 * score_util
    )
    capacity_raw = np.clip(capacity_raw, 0, 100)

    if ml_adjustment is not None:
        # ml_adjustment is a rare-event probability (0-1 or already-rescaled 0-100 score);
        # rescale it the same way as Intent so it doesn't just drag every score toward 0.
        adj = ml_adjustment
        if adj.max() <= 1.0 + 1e-9:
            adj = adj * 100
        adj_rescaled, _, _ = rescale_proba_to_score(adj / 100.0)
        capacity_final = np.clip(0.85 * capacity_raw + 0.15 * adj_rescaled, 0, 100)
    else:
        capacity_final = capacity_raw

    # A weighted average of several already-bounded 0-100 sub-scores naturally regresses
    # to the middle (it's rare for every sub-score to max out together), which would
    # otherwise leave almost nobody near the top or bottom of the scale. Percentile-stretch
    # the blended score back out to span close to 0-100 -- ranking is unchanged, only the
    # scale calibration, similar to how bureau scores are themselves rescaled distributions.
    lo, hi = np.percentile(capacity_final, 2), np.percentile(capacity_final, 98)
    span = max(hi - lo, 1e-9)
    capacity_final = np.clip((capacity_final - lo) / span * 100, 0, 100)
    return np.round(capacity_final, 1)


# --------------------------------------------------------------------------- Propensity score

PROPENSITY_EVENTS = [
    ("salary_hike_detected", 25, "salary_hike_pct",
     lambda f, i: f"Salary hike detected: +{f['salary_hike_pct'].iloc[i]:.0f}% over the last 6 months"),
    ("new_rent_detected", 20, None,
     lambda f, i: "New rent payment started — likely relocation, may need a Home/Personal loan"),
    ("education_spend_increase", 15, None,
     lambda f, i: "Education-related spending has risen sharply — school/college fee season"),
    ("medical_expenses_spike", 15, None,
     lambda f, i: "Spike in medical/hospital spending detected in recent months"),
    ("marriage_indicators", 20, None,
     lambda f, i: "Large one-off spend across shopping & entertainment — consistent with a wedding in the family"),
    ("investment_maturity", 15, None,
     lambda f, i: "A large lump-sum credit was detected — likely an FD/investment maturity, a reinvestment opportunity"),
    ("vehicle_insurance_lapse", 10, None,
     lambda f, i: "Vehicle insurance premium lapsed and was not renewed — may be planning to buy a new vehicle"),
]


def compute_propensity_score(features: pd.DataFrame):
    n = len(features)
    points = np.zeros(n)
    for flag_col, weight, _, _ in PROPENSITY_EVENTS:
        points = points + weight * features[flag_col].to_numpy().astype(float)
    propensity_score = np.round(np.clip(points, 0, 100), 1)

    detected_events_list = []
    for i in range(n):
        events = []
        for flag_col, weight, _, msg_fn in PROPENSITY_EVENTS:
            if bool(features[flag_col].iloc[i]):
                events.append({"event": flag_col, "points": weight, "message": msg_fn(features, i)})
        detected_events_list.append(events)
    return propensity_score, detected_events_list


# --------------------------------------------------------------------------- Composite / RAG

def compute_composite(intent: np.ndarray, capacity: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    composite = (
        COMPOSITE_WEIGHTS["intent"] * intent
        + COMPOSITE_WEIGHTS["capacity"] * capacity
        + COMPOSITE_WEIGHTS["propensity"] * propensity
    )
    return np.round(composite, 1)


def rag_status(composite: np.ndarray) -> np.ndarray:
    return np.where(composite > RAG_HOT_THRESHOLD, "Hot", np.where(composite >= RAG_WARM_THRESHOLD, "Warm", "Cold"))


# --------------------------------------------------------------------------- Product recommendation

def _emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    r = annual_rate / 12
    if tenure_months <= 0:
        return 0.0
    if r == 0:
        return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)


def _loan_type_scores(row_feats: pd.Series, row_cust: pd.Series) -> dict:
    """Score each of the 4 loan types for a single customer using detected events + browsing signal."""
    scores = {lt: 1.0 for lt in LOAN_TYPES}  # small base so every type is representable

    if row_feats["salary_hike_detected"]:
        scores["Home"] += 25
        scores["Auto"] += 8
    if row_feats["new_rent_detected"]:
        scores["Home"] += 15
        scores["Personal"] += 5
    if row_feats["education_spend_increase"]:
        scores["Personal"] += 20
    if row_feats["medical_expenses_spike"]:
        scores["Personal"] += 18
    if row_feats["marriage_indicators"]:
        scores["Personal"] += 25
    if row_feats["investment_maturity"]:
        scores["Home"] += 18
        scores["Mortgage"] += 10
    if row_feats["vehicle_insurance_lapse"]:
        scores["Auto"] += 25

    products_viewed = str(row_cust.get("products_viewed", "") or "")
    for lt in LOAN_TYPES:
        if lt in products_viewed.split(";"):
            scores[lt] += 15

    if row_cust["existing_loan_count"] == 0 and row_feats["estimated_monthly_income"] * 12 > 900000:
        scores["Home"] += 5

    return scores


def recommend_product(row_feats: pd.Series, row_cust: pd.Series) -> dict:
    scores = _loan_type_scores(row_feats, row_cust)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(s for _, s in ranked) or 1.0

    est_income = float(row_feats["estimated_monthly_income"])
    capacity_for_emi = float(row_feats["capacity_for_new_emi"])

    recs = []
    for loan_type, raw_score in ranked[:3]:
        lo_mult, hi_mult = LOAN_INCOME_MULTIPLE[loan_type]
        annual_income = est_income * 12
        income_based_amount = annual_income * (lo_mult + hi_mult) / 2
        rate = LOAN_INTEREST_RATES[loan_type]
        tenure_lo, tenure_hi = LOAN_TENURE_MONTHS[loan_type]
        tenure = int((tenure_lo + tenure_hi) / 2)

        # cap the amount so the resulting EMI stays within the customer's affordable capacity
        max_affordable_principal = None
        if capacity_for_emi > 0:
            r = rate / 12
            factor = (1 + r) ** tenure
            max_affordable_principal = capacity_for_emi * (factor - 1) / (r * factor)

        amount = income_based_amount
        if max_affordable_principal is not None:
            amount = min(amount, max(max_affordable_principal, annual_income * lo_mult * 0.3))
        amount = min(amount, LOAN_AMOUNT_CEILING[loan_type])
        amount = max(amount, 50000)
        emi = _emi(amount, rate, tenure)

        recs.append({
            "loan_type": loan_type,
            "suggested_amount": round(amount, -2),
            "suggested_amount_range": [round(amount * 0.85, -2), round(amount * 1.15, -2)],
            "suggested_tenure_months": tenure,
            "estimated_emi": round(emi, 0),
            "interest_rate": rate,
            "confidence": round(min(0.95, 0.35 + raw_score / total), 2),
            "score": round(raw_score, 1),
        })
    return {"primary": recs[0], "alternatives": recs[1:]}


def build_talking_points(row_feats: pd.Series, row_cust: pd.Series, detected_events: list, rec: dict) -> list:
    points = []
    if row_cust["loan_page_visits"] > 0:
        points.append(
            f"Visited loan pages {int(row_cust['loan_page_visits'])} time(s) in the last 30 days"
            + (f", used the EMI calculator {int(row_cust['loan_calculator_usage'])} time(s)"
               if row_cust["loan_calculator_usage"] > 0 else "")
        )
    if bool(row_cust["application_started_not_completed"]):
        points.append("Started a loan application but did not complete it — a quick follow-up call could close this")
    for ev in detected_events:
        points.append(ev["message"])
    foir_pct = row_feats["foir"] * 100
    points.append(
        f"FOIR is {foir_pct:.0f}% and estimated disposable income is about "
        f"Rs.{row_feats['disposable_income']:,.0f}/month — supports an EMI of up to "
        f"Rs.{row_feats['capacity_for_new_emi']:,.0f}/month"
    )
    points.append(
        f"Recommended: {rec['primary']['loan_type']} Loan of ~Rs.{rec['primary']['suggested_amount']:,.0f} "
        f"over {rec['primary']['suggested_tenure_months']} months (EMI ~Rs.{rec['primary']['estimated_emi']:,.0f})"
    )
    if not detected_events and row_cust["loan_page_visits"] == 0:
        points.append("No active browsing signal yet — approach via automated nurture campaign, not a direct call")
    return points
