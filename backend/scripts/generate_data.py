import pandas as pd
import numpy as np
from faker import Faker
import json
import os
from datetime import datetime, timedelta

# Create the data directory
os.makedirs('../data', exist_ok=True)

print("Starting data generation for PS2 Prospect Assist (20,000 customers)...")

fake = Faker('en_IN')
np.random.seed(42)

NUM_CUSTOMERS = 20000

# 1. Demographics
print("Generating demographics...")
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow']
occupations = ['IT Professional', 'Government Employee', 'Doctor', 'Business Owner', 'Teacher', 'Engineer', 'CA', 'Freelancer', 'Gig Worker']
employers = ['TCS', 'Infosys', 'SBI', 'Reliance', 'Wipro', 'HDFC Bank', 'HCL', 'Government of India', 'Self', 'Freelance']

data = {
    'customer_id': [f"CUST{str(i).zfill(6)}" for i in range(1, NUM_CUSTOMERS + 1)],
    'name': [fake.name() for _ in range(NUM_CUSTOMERS)],
    'age': np.random.randint(22, 60, NUM_CUSTOMERS),
    'gender': np.random.choice(['Male', 'Female'], NUM_CUSTOMERS, p=[0.6, 0.4]),
    'city': np.random.choice(cities, NUM_CUSTOMERS),
    'occupation': np.random.choice(occupations, NUM_CUSTOMERS),
    'employer_name': np.random.choice(employers, NUM_CUSTOMERS),
    'employment_type': np.random.choice(['Salaried', 'Self-employed', 'Gig/Freelance'], NUM_CUSTOMERS, p=[0.6, 0.25, 0.15])
}
df = pd.DataFrame(data)

# Fix employers based on employment type
df.loc[df['employment_type'] == 'Self-employed', 'employer_name'] = 'Self'
df.loc[df['employment_type'] == 'Gig/Freelance', 'employer_name'] = 'Freelance'
df.loc[df['occupation'] == 'Government Employee', 'employer_name'] = 'Government of India'

# 2. Account Information
print("Generating account information...")
df['account_type'] = np.random.choice(['Savings', 'Current'], NUM_CUSTOMERS, p=[0.8, 0.2])
df.loc[df['employment_type'] == 'Self-employed', 'account_type'] = np.random.choice(['Savings', 'Current'], sum(df['employment_type'] == 'Self-employed'), p=[0.3, 0.7])
df['account_tenure_years'] = np.random.randint(1, 16, NUM_CUSTOMERS)
df['avg_monthly_balance'] = np.random.lognormal(mean=10, sigma=1.5, size=NUM_CUSTOMERS).astype(int)
df['avg_monthly_balance'] = np.clip(df['avg_monthly_balance'], 10000, 2000000)

# 3. Transaction History (6 months)
print("Generating transaction history...")
df['salary_credits'] = np.where(df['employment_type'] == 'Salaried', np.random.normal(80000, 40000, NUM_CUSTOMERS), 0).astype(int)
df['salary_credits'] = np.clip(df['salary_credits'], 0, None)
df['other_income_credits'] = np.where(df['employment_type'] != 'Salaried', np.random.normal(100000, 80000, NUM_CUSTOMERS), np.random.normal(10000, 20000, NUM_CUSTOMERS)).astype(int)
df['other_income_credits'] = np.clip(df['other_income_credits'], 0, None)

# UPI Categories
upi_categories = ['food_dining', 'groceries', 'shopping', 'rent', 'emi_payments', 'investments_sip', 
                  'insurance_premium', 'education', 'travel', 'entertainment', 'fuel', 'medical', 
                  'subscription_services', 'utility_bills', 'loan_repayments']

for cat in upi_categories:
    df[f'upi_{cat}'] = np.random.exponential(scale=2000, size=NUM_CUSTOMERS).astype(int)

df['atm_withdrawals_count'] = np.random.poisson(lam=2, size=NUM_CUSTOMERS)
df['atm_withdrawals_amount'] = df['atm_withdrawals_count'] * np.random.normal(2000, 500, NUM_CUSTOMERS).astype(int)
df['atm_withdrawals_amount'] = np.clip(df['atm_withdrawals_amount'], 0, None)

# Life Events (Propensity Signals)
df['salary_hike_detected'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.9, 0.1])
df['new_rent_payment'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.95, 0.05])
df['education_spend_increase'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.92, 0.08])
df['medical_expenses_spike'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.96, 0.04])
df['marriage_indicators'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.97, 0.03])
df['investment_maturity'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.94, 0.06])
df['vehicle_insurance_lapse'] = np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.98, 0.02])

# 4. Digital Behavior
print("Generating digital behavior...")
df['app_login_frequency'] = np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], NUM_CUSTOMERS, p=[0.2, 0.4, 0.3, 0.1])
df['loan_page_visits'] = np.random.poisson(lam=1.5, size=NUM_CUSTOMERS)
df['loan_calculator_usage'] = np.where(df['loan_page_visits'] > 2, np.random.poisson(lam=2, size=NUM_CUSTOMERS), 0)
df['time_on_loan_pages'] = df['loan_page_visits'] * np.random.normal(2, 1, NUM_CUSTOMERS)
df['time_on_loan_pages'] = np.clip(df['time_on_loan_pages'], 0, None).astype(int)
df['application_started_not_completed'] = np.where(df['loan_calculator_usage'] > 0, np.random.choice([0, 1], NUM_CUSTOMERS, p=[0.8, 0.2]), 0)
df['last_visit_days_ago'] = np.random.randint(1, 30, NUM_CUSTOMERS)

def get_product(visits):
    if visits > 0:
        return np.random.choice(['Personal Loan', 'Home Loan', 'Auto Loan', 'Mortgage'], p=[0.5, 0.25, 0.15, 0.1])
    return 'None'
df['products_viewed'] = df['loan_page_visits'].apply(get_product)

# 5. Credit Bureau Data
print("Generating credit bureau data...")
df['credit_score'] = np.random.normal(720, 80, NUM_CUSTOMERS).astype(int)
df['credit_score'] = np.clip(df['credit_score'], 300, 900)
df['existing_loan_count'] = np.random.poisson(lam=1, size=NUM_CUSTOMERS)
df['existing_loan_count'] = np.clip(df['existing_loan_count'], 0, 5)
df['total_emi_burden'] = df['existing_loan_count'] * np.random.normal(15000, 5000, NUM_CUSTOMERS).astype(int)
df['total_emi_burden'] = np.clip(df['total_emi_burden'], 0, None)
df['credit_utilization'] = np.random.uniform(0.1, 0.9, NUM_CUSTOMERS)
df['payment_history_score'] = np.clip(np.random.normal(90, 15, NUM_CUSTOMERS), 0, 100).astype(int)

# 6. Labels (actually_took_loan)
# Create a realistic target based on intent and propensity
print("Generating labels...")
base_prob = 0.01
intent_boost = (df['loan_calculator_usage'] > 0).astype(float) * 0.1 + df['application_started_not_completed'] * 0.15
prop_boost = (df['salary_hike_detected'] + df['new_rent_payment'] + df['marriage_indicators']) * 0.05
credit_boost = (df['credit_score'] > 750).astype(float) * 0.02
credit_penalty = (df['credit_score'] < 600).astype(float) * -0.1

prob = base_prob + intent_boost + prop_boost + credit_boost + credit_penalty
prob = np.clip(prob, 0, 1)

df['actually_took_loan'] = np.random.binomial(1, prob)
df['loan_type_taken'] = np.where(df['actually_took_loan'] == 1, 
                                np.random.choice(['Personal Loan', 'Home Loan', 'Auto Loan', 'Mortgage'], size=NUM_CUSTOMERS, p=[0.6, 0.2, 0.15, 0.05]),
                                'None')
df['loan_amount_taken'] = np.where(df['actually_took_loan'] == 1, 
                                  np.random.lognormal(mean=12, sigma=1, size=NUM_CUSTOMERS), 
                                  0).astype(int)

# Save
df.to_csv('../data/synthetic_customers.csv', index=False)
print(f"Generated {len(df)} customers. Saved to data/synthetic_customers.csv.")
print(f"Total converted: {df['actually_took_loan'].sum()} ({(df['actually_took_loan'].sum()/NUM_CUSTOMERS)*100:.2f}%)")
