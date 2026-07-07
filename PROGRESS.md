# PS2 — Prospect Assist AI — Build Progress

Backend port 8002, frontend port 5175. Python 3.11 venv at `backend/venv`.

## Status

- [x] Backend venv created (`py -3.11 -m venv venv`), requirements.txt installed
      (fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, xgboost, joblib, faker).
- [ ] `scripts/generate_data.py` — 20,000 synthetic IDBI customers, vectorized with pandas/numpy.
- [ ] `scoring/models.py` — shared Intent/Capacity/Propensity/Composite scoring logic (used by
      both the training/batch-scoring pipeline and the live FastAPI endpoints).
- [ ] `scoring/train_intent_model.py` (or folded into models.py) — XGBoost classifier for Intent,
      calibrated probabilities, noise injected so AUC lands ~85-95% (not 99%+).
- [ ] `app/main.py` — FastAPI app, all 9 endpoints from spec.
- [ ] Data generated, models trained, batch scoring run — numbers recorded below.
- [ ] All 9 endpoints curl-tested on port 8002.
- [ ] `backend/README.md` written.
- [ ] Frontend scaffolded (Vite + React), 4 pages built.
- [ ] Frontend dev server confirmed running on 5175, no console errors.

## Endpoints (spec) — tick off as tested

- [ ] GET /
- [ ] GET /dashboard
- [ ] GET /leads (+ filters: status, loan_type, sort)
- [ ] GET /lead/{customer_id}
- [ ] GET /lead/{customer_id}/income
- [ ] GET /lead/{customer_id}/spending
- [ ] POST /recommend/{customer_id}
- [ ] GET /analytics
- [ ] GET /analytics/conversion

## Key numbers (fill in once data/models are built)

- Total customers: TBD
- Hot / Warm / Cold counts: TBD
- Predicted conversion rate: TBD
- Intent model AUC: TBD

## Notes for a fresh session picking this up

- Read `CLAUDE_PROMPTS/03_PS2_PROSPECT_ASSIST.md` for the full spec (schema, formula, endpoints, frontend pages).
- Composite = 0.40*Intent + 0.35*Capacity + 0.25*Propensity. RAG: Hot >75, Warm 50-75, Cold <50.
- Reference pattern: `ps4-default-prediction/backend/ml/feature_engineering.py` + `risk_logic.py`
  for splitting shared logic between training and live serving.
