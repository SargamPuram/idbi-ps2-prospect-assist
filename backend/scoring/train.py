import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('../models', exist_ok=True)
os.makedirs('../data', exist_ok=True)

print("Loading data...")
# keep_default_na=False: 'products_viewed'/'loan_type_taken' legitimately store the
# literal string "None". Without this, pandas parses "None" as NaN, which then
# round-trips as an empty string on the way back out to scored_customers.csv,
# silently breaking every `!= "None"` fallback check in app/main.py.
df = pd.read_csv('../data/synthetic_customers.csv', keep_default_na=False)

# 1. Train Intent Model (XGBoost)
print("Training Intent Model...")
intent_features = ['loan_page_visits', 'loan_calculator_usage', 'time_on_loan_pages', 'application_started_not_completed', 'last_visit_days_ago']
X = df[intent_features]
# For training intent, we use actually_took_loan as the proxy target
y = df['actually_took_loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, '../models/intent_model.pkl')
print("Intent Model trained and saved.")

# 2. Batch Scoring
print("Running Batch Scoring...")
from models import calculate_scores
scored_df = calculate_scores(df, clf)

scored_df.to_csv('../data/scored_customers.csv', index=False)
print("Batch scoring complete. Saved to data/scored_customers.csv.")
