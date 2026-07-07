import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

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

class RecommendRequest(BaseModel):
    api_key: str

@app.post("/recommend/{customer_id}")
def generate_recommendation(customer_id: str, payload: RecommendRequest):
    lead = df[df['customer_id'] == customer_id]
    if len(lead) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    lead = lead.iloc[0].to_dict()
    
    genai.configure(api_key=payload.api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    recommended = lead["products_viewed"] if lead["products_viewed"] != "None" else "Personal Loan"
    
    prompt = f"""
    You are an AI assistant helping a bank Relationship Manager pitch a loan.
    Customer Name: {lead['name']}
    Age: {lead['age']}, Occupation: {lead['occupation']}
    Recommended Product: {recommended}
    Behavior: Visited loan pages {lead['loan_page_visits']} times. Used calculator: {lead['loan_calculator_usage']>0}.
    
    Generate:
    1. A short, highly personalized 3-sentence script for the RM to say on a phone call.
    2. 3 bullet points on 'Why this product fits'.
    Format as JSON: {{"script": "...", "reasons": ["...", "..."]}}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Extract JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
            
        return json.loads(text)
    except Exception as e:
        return {
            "script": f"Hi {lead['name']}, this is your RM from IDBI Bank. I noticed you were exploring our {recommended} options online recently. I'd love to help you get the best interest rate.",
            "reasons": [
                f"Customer frequently visited {recommended} pages.",
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
