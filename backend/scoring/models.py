import pandas as pd
import numpy as np
import joblib

def calculate_capacity_score(row):
    # Estimate income
    if row['employment_type'] == 'Salaried':
        income = row['salary_credits']
    else:
        income = row['other_income_credits']
    
    if income == 0:
        return 0
        
    # Calculate Needs (estimated from UPI)
    needs = row['upi_food_dining'] + row['upi_groceries'] + row['upi_rent'] + row['upi_utility_bills'] + row['upi_medical']
    
    # Existing commitments
    commitments = row['total_emi_burden']
    
    disposable = income - (needs + commitments)
    disposable_ratio = disposable / income if income > 0 else 0
    
    # Scale to 0-100 (If they have > 40% disposable, score is 100)
    score = (disposable_ratio / 0.4) * 100
    return np.clip(score, 0, 100)

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
