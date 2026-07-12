import os
import json
import pandas as pd
import numpy as np
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from scoring.safety import sanitize_lead_fields

load_dotenv()
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash").strip()
_deepseek_client = httpx.Client(
    base_url="https://api.deepseek.com",
    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
    timeout=30.0,
) if DEEPSEEK_API_KEY else None

# Resolve the data directory relative to this file's location (backend/data),
# so the app works regardless of the current working directory it's launched
# from (local dev, `uvicorn app.main:app` from backend/, or Docker WORKDIR /app).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "scored_customers.csv")

app = FastAPI(title="Prospect Assist AI", description="IDBI PS2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
df = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    global df
    try:
        # keep_default_na=False: the CSV legitimately stores the literal string
        # "None" (no product viewed / no loan taken) in a couple of columns.
        # Pandas' default NA-string sniffing otherwise silently turns that into
        # NaN -> 0 after fillna, breaking every `!= "None"` fallback check below.
        df = pd.read_csv(DATA_PATH, keep_default_na=False)
        df.fillna(0, inplace=True)
        # Ensure ID is string
        df['customer_id'] = df['customer_id'].astype(str)
        print(f"Loaded {len(df)} customers from scored data.")
    except Exception as e:
        print(f"Error loading data: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "service": "PS2 Prospect Assist AI"}

@app.get("/dashboard")
def get_dashboard():
    total_prospects = len(df)
    
    # RAG Status Counts
    hot_count = int(sum(df['rag_status'] == 'Hot'))
    warm_count = int(sum(df['rag_status'] == 'Warm'))
    cold_count = int(sum(df['rag_status'] == 'Cold'))
    
    # Calculate pipeline value (Estimated loan amount for hot + warm)
    # Using 10 lakhs avg for simplicity or average of loan_amount_taken
    pipeline_value = df[df['rag_status'].isin(['Hot', 'Warm'])]['loan_amount_taken'].sum()
    if pipeline_value == 0:
        pipeline_value = (hot_count * 1500000) + (warm_count * 500000)
    
    # Predicted Conversion Rate (Base 1% + lift from Hot/Warm leads)
    # Assuming Hot converts at 35%, Warm at 10%
    predicted_conversions = (hot_count * 0.35) + (warm_count * 0.10)
    predicted_conversion_rate = round((predicted_conversions / total_prospects) * 100, 1)
    
    avg_score = round(df['composite_score'].mean(), 1)
    
    return {
        "total_prospects": total_prospects,
        "hot_leads": hot_count,
        "warm_leads": warm_count,
        "cold_leads": cold_count,
        "pipeline_value_cr": round(pipeline_value / 10000000, 2),
        "predicted_conversion_rate": predicted_conversion_rate,
        "avg_lead_score": avg_score
    }

@app.get("/leads")
def get_leads(
    status: Optional[str] = None,
    loan_type: Optional[str] = None,
    city: Optional[str] = None,
    sort: Optional[str] = 'composite_score',
    page: int = 1,
    limit: int = 50
):
    filtered_df = df.copy()
    
    if status:
        filtered_df = filtered_df[filtered_df['rag_status'].str.lower() == status.lower()]
    
    if loan_type and loan_type.lower() != 'all':
        filtered_df = filtered_df[filtered_df['products_viewed'].str.lower() == loan_type.lower()]
        
    if city and city.lower() != 'all':
        filtered_df = filtered_df[filtered_df['city'].str.lower() == city.lower()]
        
    # Sorting
    ascending = False
    if sort in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort, ascending=ascending)
        
    # Pagination
    total = len(filtered_df)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    results = filtered_df.iloc[start_idx:end_idx].to_dict(orient='records')
    
    # Format for frontend
    formatted_results = []
    for r in results:
        formatted_results.append({
            "customer_id": r["customer_id"],
            "name": r["name"],
            "city": r["city"],
            "composite_score": r["composite_score"],
            "intent_score": r["intent_score"],
            "capacity_score": r["capacity_score"],
            "propensity_score": r["propensity_score"],
            "rag_status": r["rag_status"],
            "recommended_product": r["products_viewed"] if r["products_viewed"] != "None" else "Personal Loan",
            "estimated_amount": int(r["loan_amount_taken"]) if r["loan_amount_taken"] > 0 else 500000,
            "status": "Pending"
        })
        
    return {
        "data": formatted_results,
        "total": total,
        "page": page,
        "limit": limit
    }

@app.get("/lead/{customer_id}")
def get_lead_details(customer_id: str):
    lead = df[df['customer_id'] == customer_id]
    if len(lead) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    lead = lead.iloc[0].to_dict()
    
    # Life Events
    life_events = []
    if lead.get("salary_hike_detected") == 1:
        life_events.append({"event": "Salary Hike Detected", "icon": "trending-up", "description": "Likely increase in disposable income."})
    if lead.get("new_rent_payment") == 1:
        life_events.append({"event": "New Rent Payment", "icon": "home", "description": "Relocation detected. High propensity for Personal Loan."})
    if lead.get("education_spend_increase") == 1:
        life_events.append({"event": "Education Spend Increase", "icon": "book", "description": "Possible need for Education Loan."})
    if lead.get("medical_expenses_spike") == 1:
        life_events.append({"event": "Medical Expense Spike", "icon": "activity", "description": "High probability of need for Personal Loan."})
    if lead.get("marriage_indicators") == 1:
        life_events.append({"event": "Marriage Indicators", "icon": "users", "description": "Jewelry/Event spending detected."})
        
    recommended = lead["products_viewed"] if lead["products_viewed"] != "None" else "Personal Loan"

    return {
        "profile": {
            "customer_id": lead["customer_id"],
            "name": lead["name"],
            "age": int(lead["age"]),
            "gender": lead["gender"],
            "city": lead["city"],
            "occupation": lead["occupation"],
            "employer": lead["employer_name"],
            "account_tenure": int(lead["account_tenure_years"])
        },
        "scores": {
            "composite": lead["composite_score"],
            "intent": lead["intent_score"],
            "capacity": lead["capacity_score"],
            "propensity": lead["propensity_score"],
            "rag_status": lead["rag_status"]
        },
        "behavior": {
            "app_login_frequency": lead["app_login_frequency"],
            "loan_page_visits": int(lead["loan_page_visits"]),
            "calculator_usage": int(lead["loan_calculator_usage"]),
            "application_started": bool(lead["application_started_not_completed"])
        },
        "life_events": life_events,
        # Need/want/retained-for-savings cash-flow segmentation (see
        # scoring/models.py::calculate_cash_flow_segments) -- IDBI's own PS2
        # pitch calls for segmenting gig/self-employed cash flow this way
        # instead of a static FOIR ratio, using UPI-derived signals.
        "cash_flow_segmentation": {
            "need_ratio": float(lead.get("need_ratio", 0)),
            "want_ratio": float(lead.get("want_ratio", 0)),
            "retained_income_ratio": float(lead.get("retained_income_ratio", 0)),
            "active_savings_ratio": float(lead.get("active_savings_ratio", 0)),
        },
        # Salary-credit velocity / spend-discipline red flag: IDBI's own worked
        # example of poor financial discipline ("salary credited day one, entire
        # balance spent immediately"), detected from intra-month timing signals
        # rather than monthly totals.
        "spend_discipline": {
            "discipline_score": float(lead.get("discipline_score", 0)),
            "low_financial_discipline_flag": bool(lead.get("low_financial_discipline_flag", 0)),
            "salary_credit_day_of_month": int(lead.get("salary_credit_day_of_month", 0)),
            "pct_income_spent_within_3_days": float(lead.get("pct_income_spent_within_3_days", 0)),
            "days_to_balance_depletion": int(lead.get("days_to_balance_depletion", 0)),
        },
        # Data-quality/confidence flag, same pattern as PS3's confidence_level:
        # how many independent, reasonably fresh data sources are actually
        # available for this customer (addresses IDBI's Q&A concern about
        # unreliable/unverifiable/thin data).
        "data_quality": {
            "confidence_level": lead.get("confidence_level", "Low"),
            "data_completeness_score": float(lead.get("data_completeness_score", 0)),
            "months_of_data_available": int(lead.get("months_of_data_available", 0)),
            "credit_bureau_available": bool(lead.get("credit_bureau_available", 0)),
        },
        "recommendation": {
            "product": recommended,
            "estimated_amount": int(lead["loan_amount_taken"]) if lead["loan_amount_taken"] > 0 else 500000,
            "confidence": f"{int(lead['composite_score'])}%"
        }
    }

@app.get("/lead/{customer_id}/income")
def get_lead_income(customer_id: str):
    lead = df[df['customer_id'] == customer_id]
    if len(lead) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    lead = lead.iloc[0].to_dict()
    
    income = lead["salary_credits"] if lead["employment_type"] == "Salaried" else lead["other_income_credits"]
    
    # Expenses
    needs = lead["upi_food_dining"] + lead["upi_groceries"] + lead["upi_rent"] + lead["upi_utility_bills"] + lead["upi_medical"]
    wants = lead["upi_shopping"] + lead["upi_entertainment"] + lead["upi_travel"] + lead["upi_subscription_services"]
    investments = lead["upi_investments_sip"]
    emis = lead["total_emi_burden"]
    
    disposable = income - (needs + emis)
    
    return {
        "estimated_monthly_income": int(income),
        "income_type": lead["employment_type"],
        "disposable_income": int(disposable),
        "foir": round((emis / income) * 100, 1) if income > 0 else 0,
        "breakdown": {
            "needs": int(needs),
            "wants": int(wants),
            "investments": int(investments),
            "emis": int(emis)
        },
        "trend": [
            int(income * np.random.uniform(0.9, 1.1)) for _ in range(6)
        ]
    }

@app.get("/lead/{customer_id}/spending")
def get_lead_spending(customer_id: str):
    lead = df[df['customer_id'] == customer_id]
    if len(lead) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    lead = lead.iloc[0].to_dict()
    
    categories = [
        {"name": "Food & Dining", "value": int(lead["upi_food_dining"])},
        {"name": "Groceries", "value": int(lead["upi_groceries"])},
        {"name": "Shopping", "value": int(lead["upi_shopping"])},
        {"name": "Rent", "value": int(lead["upi_rent"])},
        {"name": "Travel", "value": int(lead["upi_travel"])},
        {"name": "Medical", "value": int(lead["upi_medical"])}
    ]
    
    return sorted(categories, key=lambda x: x["value"], reverse=True)

@app.post("/recommend/{customer_id}")
def generate_recommendation(customer_id: str):
    lead = df[df['customer_id'] == customer_id]
    if len(lead) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    lead = lead.iloc[0].to_dict()

    recommended = lead["products_viewed"] if lead["products_viewed"] != "None" else "Personal Loan"

    # Pre-LLM defensive check (see scoring/safety.py). These three values come
    # from the synthetic dataset via a customer_id lookup rather than raw
    # free-text user input, but they still cross the boundary into an LLM
    # prompt below, so they're validated/sanitized the same way any untrusted
    # string would be before interpolation -- defense in depth, not a claim
    # that this endpoint is high-risk today.
    safe_fields = sanitize_lead_fields(
        name=lead.get("name"),
        occupation=lead.get("occupation"),
        recommended_product=recommended,
    )
    safe_name = safe_fields["name"]
    safe_occupation = safe_fields["occupation"]
    safe_recommended = safe_fields["recommended_product"]

    if _deepseek_client is None:
        return {
            "script": f"Hi {safe_name}, this is your RM from IDBI Bank. I noticed you were exploring our {safe_recommended} options online recently. I'd love to help you get the best interest rate.",
            "reasons": [
                f"Customer frequently visited {safe_recommended} pages.",
                "Has sufficient disposable income for EMI.",
                "Strong past relationship with IDBI."
            ]
        }

    prompt = f"""
    You are an AI assistant helping a bank Relationship Manager pitch a loan.
    Customer Name: {safe_name}
    Age: {lead['age']}, Occupation: {safe_occupation}
    Recommended Product: {safe_recommended}
    Behavior: Visited loan pages {lead['loan_page_visits']} times. Used calculator: {lead['loan_calculator_usage']>0}.

    Generate:
    1. A short, highly personalized 3-sentence script for the RM to say on a phone call.
    2. 3 bullet points on 'Why this product fits'.
    Format as JSON: {{"script": "...", "reasons": ["...", "..."]}}
    """

    try:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "stream": False,
        }
        resp = _deepseek_client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        # Extract JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
            
        return json.loads(text)
    except Exception as e:
        return {
            "script": f"Hi {safe_name}, this is your RM from IDBI Bank. I noticed you were exploring our {safe_recommended} options online recently. I'd love to help you get the best interest rate.",
            "reasons": [
                f"Customer frequently visited {safe_recommended} pages.",
                "Has sufficient disposable income for EMI.",
                "Strong past relationship with IDBI."
            ]
        }

@app.get("/analytics")
def get_analytics():
    return {
        "score_distribution": {
            "hot": int(sum(df['rag_status'] == 'Hot')),
            "warm": int(sum(df['rag_status'] == 'Warm')),
            "cold": int(sum(df['rag_status'] == 'Cold'))
        },
        "funnel": [
            {"stage": "Total Customers", "count": len(df)},
            {"stage": "Qualified", "count": int(sum(df['composite_score'] > 20))},
            {"stage": "Warm Leads", "count": int(sum(df['rag_status'] == 'Warm'))},
            {"stage": "Hot Leads", "count": int(sum(df['rag_status'] == 'Hot'))}
        ]
    }

@app.get("/analytics/conversion")
def get_conversion_analytics():
    total = len(df)
    hot_count = int(sum(df['rag_status'] == 'Hot'))
    warm_count = int(sum(df['rag_status'] == 'Warm'))

    actual_converted = int(df['actually_took_loan'].sum())
    actual_rate = round((actual_converted / total) * 100, 2) if total else 0

    # Same conversion assumptions used by /dashboard: Hot converts at 35%, Warm at 10%
    predicted_conversions = (hot_count * 0.35) + (warm_count * 0.10)
    predicted_rate = round((predicted_conversions / total) * 100, 2) if total else 0

    baseline_rate = 1.0  # Bank's stated current baseline (<1% conversion)

    return {
        "baseline_conversion_rate": baseline_rate,
        "actual_conversion_rate": actual_rate,
        "predicted_conversion_rate": predicted_rate,
        "actual_converted_count": actual_converted,
        "predicted_converted_count": int(round(predicted_conversions)),
        "improvement_multiplier": round(predicted_rate / baseline_rate, 1) if baseline_rate else 0,
        "by_loan_type": [
            {
                "loan_type": loan_type,
                "converted": int(sub["actually_took_loan"].sum()),
                "total": int(len(sub))
            }
            for loan_type, sub in df[df['products_viewed'] != 'None'].groupby('products_viewed')
        ]
    }
