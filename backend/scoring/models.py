import pandas as pd
import numpy as np
import joblib

# NEEDS_CATEGORIES/WANTS_CATEGORIES/LUXURY_CATEGORIES/INVESTMENT_CATEGORIES/EMI_CATEGORIES
# already existed in scoring/constants.py (written for an earlier, more detailed version
# of this pipeline) but were unused by this module. Reused here instead of re-deriving
# a new set of category groupings from scratch.
from constants import (
    EMI_CATEGORIES, INVESTMENT_CATEGORIES, LUXURY_CATEGORIES, NEEDS_CATEGORIES, WANTS_CATEGORIES,
)


def _get_income(row):
    if row['employment_type'] == 'Salaried':
        return row['salary_credits']
    return row['other_income_credits']


def calculate_cash_flow_segments(row):
    """
    Segment a customer's monthly cash flow into need / want / retained-for-savings
    buckets from UPI spend-category signals, instead of a single static FOIR-style
    ratio. This mirrors IDBI's stated preference (see PS2 pitch deck USP section)
    for assessing gig/self-employed repayment capacity: essential needs,
    discretionary wants, and what's actually retained (either actively moved to
    savings/investment instruments, or simply left over) as a share of UPI-derived
    income, rather than a static demographic/FOIR lookup.

    - need_amount:  essential spend (groceries, rent, utilities, medical, fuel,
      education) + all EMI/loan-repayment commitments (bureau-reported total_emi_burden
      plus UPI emi/loan-repayment transactions), i.e. money that isn't discretionary.
    - want_amount:  discretionary spend (dining out, shopping, subscriptions,
      travel, entertainment).
    - retained_amount: income - need_amount - want_amount. What's left over each
      month -- whether idle in the account or actively swept into SIPs/insurance
      (active_savings_ratio breaks out that actively-saved slice separately).
    """
    income = _get_income(row)
    if income is None or income <= 0:
        return {
            'monthly_income_estimate': 0.0,
            'need_amount': 0.0, 'want_amount': 0.0, 'retained_amount': 0.0,
            'need_ratio': 0.0, 'want_ratio': 0.0, 'retained_income_ratio': 0.0,
            'active_savings_ratio': 0.0,
        }

    needs = sum(row[f'upi_{c}'] for c in NEEDS_CATEGORIES)
    needs += sum(row[f'upi_{c}'] for c in EMI_CATEGORIES)
    needs += row['total_emi_burden']

    wants = sum(row[f'upi_{c}'] for c in WANTS_CATEGORIES)
    wants += sum(row[f'upi_{c}'] for c in LUXURY_CATEGORIES)

    active_savings = sum(row[f'upi_{c}'] for c in INVESTMENT_CATEGORIES)

    retained_amount = income - needs - wants

    return {
        'monthly_income_estimate': float(income),
        'need_amount': float(needs),
        'want_amount': float(wants),
        'retained_amount': float(retained_amount),
        'need_ratio': round(float(np.clip(needs / income, 0, 5)), 4),
        'want_ratio': round(float(np.clip(wants / income, 0, 5)), 4),
        'retained_income_ratio': round(float(np.clip(retained_amount / income, -5, 1)), 4),
        'active_savings_ratio': round(float(np.clip(active_savings / income, 0, 1)), 4),
    }


def calculate_discipline_score(row):
    """
    0-100 spend-discipline score built from the salary-credit velocity signal
    (scripts/generate_data.py: salary_credit_day_of_month, pct_income_spent_within_3_days,
    days_to_balance_depletion). Flags the "salary credited day one, entire balance
    spent almost immediately" pattern IDBI gave as its own worked example of poor
    financial discipline -- a signal a monthly-aggregate FOIR ratio can't see
    because it never looks at intra-month timing, only totals.

    100 = spends gradually across the month (disciplined). 0 = nearly the whole
    month's income is gone within days of being credited (red flag).
    """
    pct_3d = float(row.get('pct_income_spent_within_3_days', 0.15))
    days_to_deplete = float(row.get('days_to_balance_depletion', 20))

    velocity_score = (1 - pct_3d) * 100
    depletion_score = np.clip(days_to_deplete / 20 * 100, 0, 100)
    score = 0.6 * velocity_score + 0.4 * depletion_score
    return float(np.clip(score, 0, 100))


def calculate_confidence(row):
    """
    Data-quality/confidence flag, structurally the same pattern as PS3's
    confidence_level (ps3-financial-health/backend/scoring/engine.py
    confidence_level_for): count how many independent, reasonably fresh data
    sources are actually available for this customer, rather than assuming every
    field is equally reliable. Addresses IDBI's own Q&A concern about
    unreliable/unverifiable/thin data (wrong account numbers, thin files) --
    PS3 already had an analogous flag; PS2 previously had none.

    Sources checked (5 total, same >=4/>=2 High/Medium/Low thresholds as PS3):
      1. income data available (salary/other-income credits > 0)
      2. UPI transaction data available (any spend-category signal present)
      3. credit bureau record available (not a thin-file customer)
      4. sufficient transaction history (>= 4 months of data available)
      5. established account relationship (>= 1 year tenure)
    """
    sources_present = 0
    total_sources = 5

    if _get_income(row) > 0:
        sources_present += 1

    upi_total = sum(row[f'upi_{c}'] for c in NEEDS_CATEGORIES + WANTS_CATEGORIES)
    if upi_total > 0:
        sources_present += 1

    if bool(row.get('credit_bureau_available', 1)):
        sources_present += 1

    if row.get('months_of_data_available', 6) >= 4:
        sources_present += 1

    if row.get('account_tenure_years', 0) >= 1:
        sources_present += 1

    data_completeness_score = round(sources_present / total_sources * 100, 1)
    if sources_present >= 4:
        level = 'High'
    elif sources_present >= 2:
        level = 'Medium'
    else:
        level = 'Low'
    return data_completeness_score, level


def calculate_capacity_score(row):
    # Need / want / retained-for-savings segmentation (see calculate_cash_flow_segments)
    segments = calculate_cash_flow_segments(row)
    income = segments['monthly_income_estimate']
    if income == 0:
        return 0

    retained_ratio = segments['retained_income_ratio']
    # Scale to 0-100 (>= 40% retained after needs+wants -> full 100, same headroom
    # the previous FOIR-style disposable_ratio used)
    base_score = (retained_ratio / 0.40) * 100

    # Fold in the salary-velocity spend-discipline signal: a customer who looks
    # fine on a monthly-aggregate basis but blows through their salary in the
    # first few days is a weaker repayment bet than the ratio alone suggests.
    discipline_score = calculate_discipline_score(row)
    score = 0.85 * base_score + 0.15 * discipline_score

    return float(np.clip(score, 0, 100))

def calculate_propensity_score(row):
    score = 0
    if row['salary_hike_detected']: score += 30
    if row['new_rent_payment']: score += 20
    if row['education_spend_increase']: score += 20
    if row['medical_expenses_spike']: score += 15
    if row['marriage_indicators']: score += 25
    if row['investment_maturity']: score += 10
    if row['vehicle_insurance_lapse']: score += 20

    return np.clip(score, 0, 100)

def calculate_scores(df, intent_model):
    # Intent Score (0-100)
    intent_features = ['loan_page_visits', 'loan_calculator_usage', 'time_on_loan_pages', 'application_started_not_completed', 'last_visit_days_ago']
    probs = intent_model.predict_proba(df[intent_features])[:, 1]

    # Scale probabilities to 0-100.
    # Since background conversion is ~4%, we stretch the top end so it looks like a 0-100 score.
    # We will use percentile scaling for better distribution.
    from scipy.stats import rankdata
    ranks = rankdata(probs)
    df['intent_score'] = (ranks / len(ranks)) * 100

    # Cash-flow need/want/retained segmentation + spend-discipline + capacity score
    segments = df.apply(calculate_cash_flow_segments, axis=1, result_type='expand')
    df['need_ratio'] = segments['need_ratio']
    df['want_ratio'] = segments['want_ratio']
    df['retained_income_ratio'] = segments['retained_income_ratio']
    df['active_savings_ratio'] = segments['active_savings_ratio']

    df['discipline_score'] = df.apply(calculate_discipline_score, axis=1).round(1)

    # Data-quality / confidence flag (structurally mirrors PS3's confidence_level)
    confidence = df.apply(calculate_confidence, axis=1, result_type='expand')
    df['data_completeness_score'] = confidence[0]
    df['confidence_level'] = confidence[1]

    # Capacity Score
    df['capacity_score'] = df.apply(calculate_capacity_score, axis=1)

    # Propensity Score
    df['propensity_score'] = df.apply(calculate_propensity_score, axis=1)

    # Composite Score
    df['composite_score'] = (0.40 * df['intent_score']) + (0.35 * df['capacity_score']) + (0.25 * df['propensity_score'])

    # RAG Status
    conditions = [
        (df['composite_score'] >= 75),
        (df['composite_score'] >= 50),
        (df['composite_score'] < 50)
    ]
    choices = ['Hot', 'Warm', 'Cold']
    df['rag_status'] = np.select(conditions, choices)

    # Format and round
    for col in ['intent_score', 'capacity_score', 'propensity_score', 'composite_score']:
        df[col] = df[col].round(1)

    return df
