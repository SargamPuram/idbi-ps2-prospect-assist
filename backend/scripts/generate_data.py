"""
Synthetic data generator for PS2 — Prospect Assist AI.

Generates 20,000 existing IDBI Bank customer profiles with 6 months of
transaction/behavioral history, digital engagement (app/web analytics) signals,
credit bureau data, and hidden "life event" ground-truth used to drive
(with genuine noise) the ~5% actually_took_loan conversion label.

Vectorized with numpy/pandas throughout — no per-row python loops for the
heavy numeric work (6-month transaction matrices across 20,000 customers).
A couple of small, unavoidably-ragged per-row loops remain (name generation via
Faker, products-viewed subset sampling, converted-only loan-type assignment) —
each only touches a small fraction of rows or is O(N) with trivial per-row cost.

IMPORTANT: the "_true_*" columns below are hidden GROUND TRUTH used only to
build the raw transaction numbers and the conversion label. The scoring engine
(scoring/models.py) never reads these columns — it must detect life events by
analyzing the actual monthly transaction time series, exactly as it would for
a genuinely new customer. This avoids label leakage (see PROGRESS.md notes).

Run from the backend/ directory:
    python scripts/generate_data.py
"""

import os
import sys

import numpy as np
import pandas as pd
from faker import Faker
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.constants import (
    ACCOUNT_TYPES, APP_LOGIN_FREQUENCIES, APP_LOGIN_WEIGHTS, CITIES,
    EMPLOYERS, EMPLOYMENT_TYPES, EMPLOYMENT_TYPE_WEIGHTS, LOAN_TYPES,
    N_MONTHS, OCCUPATIONS_BY_EMPLOYMENT, UPI_CATEGORIES,
)

N = 20000
SEED = 42
rng = np.random.default_rng(SEED)
Faker.seed(SEED)
fake = Faker("en_IN")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INCOME_PARAMS = {
    "IT Professional": (85000, 0.35),
    "Government Employee": (55000, 0.22),
    "Doctor": (150000, 0.50),
    "Business Owner": (120000, 0.60),
    "Teacher": (40000, 0.20),
    "Engineer": (70000, 0.30),
    "CA": (110000, 0.40),
    "Freelancer": (45000, 0.60),
    "Gig Worker": (25000, 0.50),
}

UPI_SHARES = {
    "food_dining": 0.05, "groceries": 0.06, "shopping": 0.05, "rent": 0.15,
    "investments_sip": 0.05, "insurance_premium": 0.02, "education": 0.05,
    "travel": 0.03, "entertainment": 0.02, "fuel": 0.03, "medical": 0.015,
    "subscription_services": 0.008, "utility_bills": 0.035, "loan_repayments": 0.01,
}


def zscore(x):
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def main():
    print(f"[generate_data] Generating {N} synthetic IDBI customer profiles...")

    customer_id = np.array([f"CUST{100000 + i}" for i in range(N)])

    # ---------------------------------------------------------------- Demographics
    age = rng.integers(22, 61, size=N)
    gender = rng.choice(["M", "F"], size=N, p=[0.62, 0.38])
    city = rng.choice(CITIES, size=N)
    employment_type = rng.choice(EMPLOYMENT_TYPES, size=N, p=EMPLOYMENT_TYPE_WEIGHTS)

    occupation = np.empty(N, dtype=object)
    for et in EMPLOYMENT_TYPES:
        mask = employment_type == et
        occupation[mask] = rng.choice(OCCUPATIONS_BY_EMPLOYMENT[et], size=int(mask.sum()))

    salaried_mask = employment_type == "Salaried"
    self_emp_mask = employment_type == "Self-employed"
    gig_mask = employment_type == "Gig/Freelance"

    employer_name = np.empty(N, dtype=object)
    employer_name[salaried_mask] = rng.choice(EMPLOYERS, size=int(salaried_mask.sum()))
    employer_name[~salaried_mask] = "Self-employed / Independent"

    print("  - generating names (Faker en_IN)...")
    names = [fake.name_male() if g == "M" else fake.name_female() for g in gender]

    account_type = rng.choice(ACCOUNT_TYPES, size=N, p=[0.82, 0.18])
    account_tenure_years = rng.integers(1, 16, size=N)

    # ---------------------------------------------------------------- Income
    base_income = np.empty(N)
    for occ, (median, sigma) in INCOME_PARAMS.items():
        mask = occupation == occ
        if mask.sum():
            base_income[mask] = rng.lognormal(mean=np.log(median), sigma=sigma, size=int(mask.sum()))
    age_factor = np.clip(0.7 + (age - 22) / 38 * 0.6, 0.7, 1.3)
    base_income = np.clip(base_income * age_factor, 8000, 800000)

    avg_monthly_balance = np.clip(rng.lognormal(mean=np.log(70000), sigma=1.05, size=N), 10000, 2_000_000)

    # ---------------------------------------------------------------- Credit bureau
    credit_score = np.clip(
        rng.normal(loc=600 + 40 * np.log1p(base_income / 50000) + account_tenure_years * 3, scale=70, size=N),
        300, 900,
    ).astype(int)
    payment_history_score = np.clip((credit_score - 300) / 600 * 100 + rng.normal(0, 10, N), 0, 100)
    existing_loan_count = np.clip(rng.poisson(0.8, N), 0, 5)
    credit_utilization = np.clip(rng.beta(2, 5, N) * 100, 0, 100)
    avg_loan_emi_each = base_income * rng.uniform(0.05, 0.15, N)
    total_emi_burden = existing_loan_count * avg_loan_emi_each

    # ---------------------------------------------------------------- Hidden life-event ground truth
    # (computed before digital behavior so browsing propensity can realistically correlate with them —
    # e.g. someone who just got a raise or is planning a wedding is *more* likely to be browsing loan
    # pages, not an independent coin-flip. This also thickens the joint high-intent/high-capacity/
    # high-propensity tail that genuinely deserves a "Hot" lead classification.)
    _true_salary_hike_flag = salaried_mask & (rng.random(N) < 0.16)
    _true_salary_hike_pct = np.where(_true_salary_hike_flag, rng.uniform(15, 45, N), 0.0)
    _true_hike_month = np.where(_true_salary_hike_flag, rng.integers(3, 7, N), 0)

    _true_new_rent_flag = rng.random(N) < 0.13
    _true_rent_move_month = np.where(_true_new_rent_flag, rng.integers(3, 7, N), 0)

    has_education_baseline = rng.random(N) < 0.35
    _true_education_spike_flag = has_education_baseline & (rng.random(N) < 0.42)

    _true_medical_spike_flag = rng.random(N) < 0.11
    _true_medical_spike_month = np.where(_true_medical_spike_flag, rng.integers(4, 7, N), 0)

    _true_marriage_flag = rng.random(N) < 0.08
    _true_marriage_month = np.where(_true_marriage_flag, rng.integers(3, 7, N), 0)
    _true_marriage_amount = np.where(_true_marriage_flag, rng.uniform(50000, 500000, N), 0.0)

    _true_investment_maturity_flag = rng.random(N) < 0.10
    _true_maturity_month = np.where(_true_investment_maturity_flag, rng.integers(2, 7, N), 0)
    _true_maturity_amount = np.where(_true_investment_maturity_flag, rng.uniform(100000, 1500000, N), 0.0)

    has_insurance_baseline = rng.random(N) < 0.45
    _true_vehicle_insurance_lapse_flag = has_insurance_baseline & (rng.random(N) < 0.25)

    has_any_life_event = (
        _true_salary_hike_flag | _true_new_rent_flag | _true_education_spike_flag | _true_medical_spike_flag
        | _true_marriage_flag | _true_investment_maturity_flag | _true_vehicle_insurance_lapse_flag
    )
    financial_quality = (
        zscore(credit_score) + zscore(payment_history_score) - zscore(existing_loan_count) + zscore(base_income)
    )
    financially_strong = financial_quality > 0.3

    # ---------------------------------------------------------------- Digital behavior (correlated with the above)
    visit_prob = np.clip(0.65 + 0.18 * financially_strong + 0.22 * has_any_life_event, 0.10, 0.95)
    no_visit_mask = rng.random(N) >= visit_prob
    power_browser_prob = np.clip(0.22 + 0.18 * financially_strong + 0.22 * has_any_life_event, 0.05, 0.85)
    power_browser_mask = (~no_visit_mask) & (rng.random(N) < power_browser_prob)
    loan_page_visits = np.where(
        no_visit_mask, 0,
        np.where(power_browser_mask, rng.poisson(10, N) + 4, rng.poisson(4, N) + 1),
    )
    loan_calculator_usage = np.where(
        loan_page_visits > 0,
        np.where(power_browser_mask, rng.poisson(4.5, N), rng.poisson(1.8, N)),
        0,
    )
    loan_calculator_usage = np.minimum(loan_calculator_usage, loan_page_visits)
    time_on_loan_pages = np.where(
        loan_page_visits > 0,
        loan_page_visits * rng.uniform(1.5, 7.0, N) + np.clip(rng.normal(0, 1, N), 0, None),
        np.clip(rng.uniform(0, 0.5, N), 0, None),
    )
    app_login_frequency = rng.choice(APP_LOGIN_FREQUENCIES, size=N, p=APP_LOGIN_WEIGHTS)
    application_started_not_completed = rng.random(N) < np.clip(
        0.02 + 0.30 * (loan_calculator_usage > 0) + 0.25 * (loan_page_visits >= 5) + 0.20 * power_browser_mask, 0, 1
    )
    last_visit_days_ago = np.where(
        loan_page_visits > 0,
        np.clip(rng.exponential(8, N), 0, 30),
        np.clip(rng.uniform(30, 180, N), 30, 180),
    )
    last_visit_timestamp = pd.Timestamp.now().normalize() - pd.to_timedelta(last_visit_days_ago, unit="D")

    print("  - sampling products viewed (ragged, small per-row loop)...")
    products_viewed = []
    for v in loan_page_visits:
        if v == 0:
            products_viewed.append("")
        else:
            k = int(min(rng.integers(1, 5), 4))
            products_viewed.append(";".join(rng.choice(LOAN_TYPES, size=k, replace=False)))
    products_viewed = np.array(products_viewed, dtype=object)
    products_viewed_diversity = np.array([0 if p == "" else len(p.split(";")) for p in products_viewed])

    has_rent_baseline = (rng.random(N) < 0.45) & (~_true_new_rent_flag)
    has_sip_baseline = rng.random(N) < 0.40

    # ---------------------------------------------------------------- Monthly matrices (N x 6)
    print("  - building 6-month transaction matrices (vectorized)...")
    months = np.arange(1, N_MONTHS + 1)
    month_grid = np.tile(months, (N, 1))  # N x 6

    salary_noise = rng.normal(1.0, 0.03, size=(N, N_MONTHS))
    salary_matrix = np.where(salaried_mask[:, None], base_income[:, None] * salary_noise, 0.0)
    hike_mult = np.where(
        (month_grid >= _true_hike_month[:, None]) & (_true_hike_month[:, None] > 0),
        1 + _true_salary_hike_pct[:, None] / 100,
        1.0,
    )
    salary_matrix = salary_matrix * hike_mult

    other_income = np.zeros((N, N_MONTHS))
    side_noise = rng.lognormal(mean=np.log(0.04), sigma=0.9, size=(N, N_MONTHS))
    side_has_income = rng.random((N, N_MONTHS)) < 0.3
    other_income = np.where(salaried_mask[:, None], base_income[:, None] * side_noise * side_has_income, other_income)

    se_noise = np.clip(rng.normal(1.0, 0.35, size=(N, N_MONTHS)), 0.1, 3.0)
    other_income = np.where(self_emp_mask[:, None], base_income[:, None] * se_noise, other_income)

    gig_noise = np.clip(rng.normal(1.0, 0.55, size=(N, N_MONTHS)), 0.0, 3.5)
    drought = rng.random((N, N_MONTHS)) < 0.15
    gig_vals = base_income[:, None] * gig_noise
    gig_vals = np.where(drought, gig_vals * 0.25, gig_vals)
    other_income = np.where(gig_mask[:, None], gig_vals, other_income)

    maturity_month_mask = (month_grid == _true_maturity_month[:, None]) & (_true_maturity_month[:, None] > 0)
    other_income = other_income + maturity_month_mask * _true_maturity_amount[:, None]

    cat_matrices = {}
    for cat, share in UPI_SHARES.items():
        noise = np.clip(rng.lognormal(mean=0.0, sigma=0.25, size=(N, N_MONTHS)), 0.2, 3.0)
        cat_matrices[cat] = base_income[:, None] * share * noise

    cat_matrices["rent"] = cat_matrices["rent"] * has_rent_baseline[:, None]
    cat_matrices["investments_sip"] = cat_matrices["investments_sip"] * has_sip_baseline[:, None]
    cat_matrices["insurance_premium"] = cat_matrices["insurance_premium"] * has_insurance_baseline[:, None]
    cat_matrices["education"] = cat_matrices["education"] * has_education_baseline[:, None]

    emi_noise = np.clip(rng.normal(1.0, 0.04, size=(N, N_MONTHS)), 0.85, 1.15)
    cat_matrices["emi_payments"] = total_emi_burden[:, None] * emi_noise

    # life-event injections into the raw numbers
    rent_start_mask = (month_grid >= _true_rent_move_month[:, None]) & (_true_rent_move_month[:, None] > 0)
    new_rent_amount = base_income[:, None] * 0.15 * np.clip(rng.lognormal(0, 0.2, size=(N, N_MONTHS)), 0.5, 2.0)
    cat_matrices["rent"] = cat_matrices["rent"] + rent_start_mask * new_rent_amount

    edu_spike_mask = (month_grid >= 4) & _true_education_spike_flag[:, None]
    cat_matrices["education"] = cat_matrices["education"] + edu_spike_mask * (
        base_income[:, None] * 0.05 * rng.uniform(1.5, 3.0, size=(N, N_MONTHS))
    )

    medical_spike_mask = (month_grid == _true_medical_spike_month[:, None]) & (_true_medical_spike_month[:, None] > 0)
    cat_matrices["medical"] = cat_matrices["medical"] + medical_spike_mask * (
        base_income[:, None] * rng.uniform(0.15, 0.5, size=(N, N_MONTHS))
    )

    marriage_mask = (month_grid == _true_marriage_month[:, None]) & (_true_marriage_month[:, None] > 0)
    cat_matrices["shopping"] = cat_matrices["shopping"] + marriage_mask * (_true_marriage_amount[:, None] * 0.6)
    cat_matrices["entertainment"] = cat_matrices["entertainment"] + marriage_mask * (_true_marriage_amount[:, None] * 0.4)

    lapse_zero_mask = (month_grid >= 5) & _true_vehicle_insurance_lapse_flag[:, None]
    cat_matrices["insurance_premium"] = np.where(lapse_zero_mask, 0.0, cat_matrices["insurance_premium"])

    atm_withdrawal_amt = base_income[:, None] * 0.08 * np.clip(rng.lognormal(0, 0.3, size=(N, N_MONTHS)), 0.2, 3.0)
    atm_withdrawal_count = rng.poisson(3, size=(N, N_MONTHS)) + 1
    card_txn_amt = base_income[:, None] * 0.10 * np.clip(rng.lognormal(0, 0.3, size=(N, N_MONTHS)), 0.2, 3.0)
    card_txn_count = rng.poisson(8, size=(N, N_MONTHS)) + 2
    recurring_debits_amt = cat_matrices["emi_payments"] + cat_matrices["subscription_services"] + cat_matrices["utility_bills"]

    # ---------------------------------------------------------------- Assemble monthly long dataframe
    print("  - assembling monthly_transactions long dataframe...")
    monthly_df = pd.DataFrame({
        "customer_id": np.repeat(customer_id, N_MONTHS),
        "month": np.tile(months, N),
    })
    monthly_df["salary_credit"] = np.round(salary_matrix.reshape(-1), 2)
    monthly_df["other_income_credit"] = np.round(other_income.reshape(-1), 2)
    for cat in UPI_CATEGORIES:
        monthly_df[f"upi_{cat}"] = np.round(cat_matrices[cat].reshape(-1), 2)
    monthly_df["atm_withdrawal_amt"] = np.round(atm_withdrawal_amt.reshape(-1), 2)
    monthly_df["atm_withdrawal_count"] = atm_withdrawal_count.reshape(-1)
    monthly_df["card_txn_amt"] = np.round(card_txn_amt.reshape(-1), 2)
    monthly_df["card_txn_count"] = card_txn_count.reshape(-1)
    monthly_df["recurring_debits_amt"] = np.round(recurring_debits_amt.reshape(-1), 2)

    # ---------------------------------------------------------------- Label generation (noised latent, ~5% positive)
    print("  - generating actually_took_loan label with injected noise (avoiding trivial leakage)...")
    foir_proxy = total_emi_burden / np.maximum(base_income, 5000)
    needs_proxy = base_income * 0.35
    disposable_proxy = base_income - needs_proxy - total_emi_burden

    intent_raw = (
        0.30 * zscore(loan_page_visits) + 0.30 * zscore(loan_calculator_usage)
        + 0.20 * zscore(time_on_loan_pages) + 1.20 * application_started_not_completed.astype(float)
        + 0.20 * zscore(products_viewed_diversity) - 0.30 * zscore(last_visit_days_ago)
    )
    capacity_raw = (
        -0.30 * zscore(foir_proxy) + 0.25 * zscore(disposable_proxy) + 0.20 * zscore(credit_score)
        + 0.15 * zscore(payment_history_score) - 0.10 * zscore(credit_utilization)
    )
    propensity_raw = (
        1.0 * _true_salary_hike_flag + 0.8 * _true_new_rent_flag + 0.7 * _true_education_spike_flag
        + 0.6 * _true_medical_spike_flag + 0.9 * _true_marriage_flag
        + 0.7 * _true_investment_maturity_flag + 0.5 * _true_vehicle_insurance_lapse_flag
    ).astype(float)

    # NOTE: these label-generation weights are deliberately NOT the same as the 0.40/0.35/0.25
    # composite-score formula used later to rank leads (scoring/constants.COMPOSITE_WEIGHTS) --
    # that formula combines the three *computed scores* for prioritization. Here we're only
    # constructing a believable ground-truth conversion label with intent as the dominant (but
    # not exclusive) driver, plus real noise, so the eventual Intent-only classifier lands at a
    # believable ~0.80-0.90 AUC rather than a suspicious ~1.0 (leakage) or a useless ~0.5 (noise).
    latent = 0.72 * zscore(intent_raw) + 0.18 * zscore(capacity_raw) + 0.10 * zscore(propensity_raw)
    noise = rng.normal(0, 0.22, N)
    combined = latent + noise

    slope = 1.2
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p_mean = expit(mid + slope * combined).mean()
        if p_mean > 0.05:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    conversion_prob = expit(intercept + slope * combined)
    actually_took_loan = rng.random(N) < conversion_prob

    n_converted = int(actually_took_loan.sum())
    print(f"    -> actually_took_loan positive rate: {actually_took_loan.mean():.3%} ({n_converted} customers)")

    loan_type_taken = np.full(N, "", dtype=object)
    loan_amount_taken = np.zeros(N)
    days_to_conversion = np.zeros(N, dtype=int)

    conv_idx = np.where(actually_took_loan)[0]
    LOAN_MULT = {"Personal": (1.0, 3.0), "Home": (3.5, 6.0), "Auto": (0.8, 1.5), "Mortgage": (2.5, 4.5)}
    LOAN_CEIL = {"Personal": 2_000_000, "Home": 15_000_000, "Auto": 2_500_000, "Mortgage": 10_000_000}
    for i in conv_idx:
        if _true_marriage_flag[i] or _true_education_spike_flag[i] or _true_medical_spike_flag[i]:
            lt = "Personal"
        elif _true_salary_hike_flag[i] or _true_new_rent_flag[i] or _true_investment_maturity_flag[i]:
            lt = "Home" if base_income[i] * 12 > 700000 else "Personal"
        elif _true_vehicle_insurance_lapse_flag[i]:
            lt = "Auto"
        else:
            lt = rng.choice(LOAN_TYPES, p=[0.45, 0.25, 0.20, 0.10])
        loan_type_taken[i] = lt
        lo_m, hi_m = LOAN_MULT[lt]
        amt = base_income[i] * 12 * rng.uniform(lo_m, hi_m)
        loan_amount_taken[i] = min(amt, LOAN_CEIL[lt])
        # hotter latent -> faster conversion
        skew = np.clip(1 - (combined[i] - combined.min()) / (combined.max() - combined.min() + 1e-9), 0.05, 1.0)
        days_to_conversion[i] = int(np.clip(rng.exponential(10 + 60 * skew), 1, 90))

    # ---------------------------------------------------------------- Assemble customers dataframe
    customers_df = pd.DataFrame({
        "customer_id": customer_id,
        "name": names,
        "age": age,
        "gender": gender,
        "city": city,
        "occupation": occupation,
        "employer_name": employer_name,
        "employment_type": employment_type,
        "account_type": account_type,
        "account_tenure_years": account_tenure_years,
        "avg_monthly_balance": np.round(avg_monthly_balance, 2),
        "app_login_frequency": app_login_frequency,
        "loan_page_visits": loan_page_visits,
        "loan_calculator_usage": loan_calculator_usage,
        "time_on_loan_pages": np.round(time_on_loan_pages, 1),
        "products_viewed": products_viewed,
        "products_viewed_diversity": products_viewed_diversity,
        "application_started_not_completed": application_started_not_completed,
        "last_visit_days_ago": np.round(last_visit_days_ago, 1),
        "last_visit_timestamp": last_visit_timestamp,
        "credit_score": credit_score,
        "existing_loan_count": existing_loan_count,
        "total_emi_burden": np.round(total_emi_burden, 2),
        "credit_utilization": np.round(credit_utilization, 1),
        "payment_history_score": np.round(payment_history_score, 1),
        "actually_took_loan": actually_took_loan,
        "loan_type_taken": loan_type_taken,
        "loan_amount_taken": np.round(loan_amount_taken, 2),
        "days_to_conversion": days_to_conversion,
        "_true_salary_hike_flag": _true_salary_hike_flag,
        "_true_salary_hike_pct": np.round(_true_salary_hike_pct, 1),
        "_true_new_rent_flag": _true_new_rent_flag,
        "_true_education_spike_flag": _true_education_spike_flag,
        "_true_medical_spike_flag": _true_medical_spike_flag,
        "_true_marriage_flag": _true_marriage_flag,
        "_true_marriage_amount": np.round(_true_marriage_amount, 2),
        "_true_investment_maturity_flag": _true_investment_maturity_flag,
        "_true_maturity_amount": np.round(_true_maturity_amount, 2),
        "_true_vehicle_insurance_lapse_flag": _true_vehicle_insurance_lapse_flag,
    })

    customers_path = os.path.join(DATA_DIR, "customers.csv")
    monthly_path = os.path.join(DATA_DIR, "monthly_transactions.csv")
    customers_df.to_csv(customers_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    print(f"[generate_data] Wrote {len(customers_df):,} customers -> {customers_path}")
    print(f"[generate_data] Wrote {len(monthly_df):,} monthly transaction rows -> {monthly_path}")
    print("[generate_data] Done.")


if __name__ == "__main__":
    main()
