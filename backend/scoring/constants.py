"""
Shared constants used by BOTH the synthetic data generator (scripts/generate_data.py)
and the scoring engine (scoring/models.py), so the two never drift out of sync.
"""

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]

EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "Gig/Freelance"]
EMPLOYMENT_TYPE_WEIGHTS = [0.60, 0.25, 0.15]

OCCUPATIONS_BY_EMPLOYMENT = {
    "Salaried": ["IT Professional", "Government Employee", "Teacher", "Engineer", "Doctor"],
    "Self-employed": ["Business Owner", "Doctor", "CA"],
    "Gig/Freelance": ["Freelancer", "Gig Worker"],
}

EMPLOYERS = [
    "TCS", "Infosys", "Wipro", "HCL Technologies", "Cognizant", "Accenture India",
    "Reliance Industries", "HDFC Bank", "State Bank of India", "ICICI Bank",
    "L&T", "Government of India", "State Government Dept.", "Amazon India",
    "Flipkart", "Tech Mahindra", "Capgemini", "Byju's", "Tata Motors", "Bajaj Auto",
    "Mahindra & Mahindra", "Axis Bank", "IBM India", "Deloitte India", "ONGC",
]

ACCOUNT_TYPES = ["Savings", "Current"]

LOAN_TYPES = ["Personal", "Home", "Auto", "Mortgage"]

APP_LOGIN_FREQUENCIES = ["Daily", "Weekly", "Monthly"]
APP_LOGIN_WEIGHTS = [0.35, 0.45, 0.20]

# UPI spend categories tracked monthly for every customer
UPI_CATEGORIES = [
    "food_dining", "groceries", "shopping", "rent", "emi_payments",
    "investments_sip", "insurance_premium", "education", "travel",
    "entertainment", "fuel", "medical", "subscription_services",
    "utility_bills", "loan_repayments",
]

# Category groupings used for expense-ratio / capacity analysis
NEEDS_CATEGORIES = ["groceries", "utility_bills", "rent", "medical", "fuel", "education"]
WANTS_CATEGORIES = ["food_dining", "shopping", "subscription_services", "travel"]
LUXURY_CATEGORIES = ["entertainment"]
EMI_CATEGORIES = ["emi_payments", "loan_repayments"]
INVESTMENT_CATEGORIES = ["investments_sip", "insurance_premium"]

N_MONTHS = 6

RAG_HOT_THRESHOLD = 75
RAG_WARM_THRESHOLD = 50

COMPOSITE_WEIGHTS = {"intent": 0.40, "capacity": 0.35, "propensity": 0.25}

# Assumed annual interest rates per loan type for EMI computation (illustrative, IDBI-style pricing)
LOAN_INTEREST_RATES = {"Personal": 0.105, "Home": 0.085, "Auto": 0.095, "Mortgage": 0.090}
LOAN_TENURE_MONTHS = {"Personal": (12, 60), "Home": (120, 240), "Auto": (36, 84), "Mortgage": (60, 180)}
# Loan amount = multiple of estimated ANNUAL income, capped at ceiling (INR)
LOAN_INCOME_MULTIPLE = {"Personal": (1.0, 3.0), "Home": (3.5, 6.0), "Auto": (0.8, 1.5), "Mortgage": (2.5, 4.5)}
LOAN_AMOUNT_CEILING = {"Personal": 2_000_000, "Home": 15_000_000, "Auto": 2_500_000, "Mortgage": 10_000_000}
