"""
Pydantic response/request models for PS2 Prospect Assist AI's serving layer.

Most read-only GET endpoints in app/main.py return plain dicts assembled from
pandas (same pattern used in ps4-default-prediction/backend/app/main.py) since the
payload shapes are naturally tabular/nested and vary by filter. The models below
cover the two places where a fixed, documented contract is genuinely useful: the
POST /recommend/{customer_id} response, and shared small building blocks reused
across several endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ProductRecommendation(BaseModel):
    loan_type: str
    suggested_amount: float
    suggested_amount_range: list[float]
    suggested_tenure_months: int
    estimated_emi: float
    interest_rate: float
    confidence: float
    score: float


class RecommendationOut(BaseModel):
    customer_id: str
    name: str
    lead_score: float
    lead_status: str
    primary_recommendation: ProductRecommendation
    alternative_recommendations: list[ProductRecommendation]
    why_this_product: str
    talking_points: list[str]
    detected_life_events: list[dict]


class DetectedEvent(BaseModel):
    event: str
    points: float
    message: str


class LeadListItem(BaseModel):
    customer_id: str
    name: str
    age: int
    city: str
    occupation: str
    employment_type: str
    lead_score: float
    lead_status: str
    intent_score: float
    capacity_score: float
    propensity_score: float
    recommended_product: str
    recommended_amount: float
    best_contact_time: str
    contact_status: str
    income_bracket: str


class LeadListOut(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    leads: list[LeadListItem]


class HealthOut(BaseModel):
    status: str
    service: str
    customers_loaded: int
    models_loaded: bool
