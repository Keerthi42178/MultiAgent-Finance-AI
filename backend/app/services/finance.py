"""Core financial calculations shared by API routes and agents.

Everything here reads/writes the database for a single authenticated user.
No LLM involvement — these are the validated backend operations.
"""
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.db_models import (
    Bill, Budget, BudgetCategory, Expense, Income, SavingsGoal, UserProfile,
)

EXPENSE_CATEGORIES = [
    "Food & Groceries", "Transportation", "Shopping", "Housing & Rent",
    "Utilities", "Healthcare", "Education", "Entertainment", "Travel",
    "Personal Care", "Subscriptions", "Other",
]
INCOME_SOURCES = ["Salary", "Freelance", "Business", "Interest", "Other"]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def parse_month(month: Optional[str], year: Optional[int], db: Session, user_id: int):
    """Resolve 'YYYY-MM' or explicit year/month into (year, month)."""
    today = date.today()
    if month and "-" in month:
        try:
            y, m = month.split("-")
            return int(y), int(m)
        except ValueError:
            pass
    y = year or today.year
    m = int(month) if month and str(month).isdigit() else today.month
    return y, m


def expense_totals(db: Session, user_id: int, start: date, end: date):
    rows = (
        db.query(Expense.category, func.sum(Expense.amount), func.count(Expense.id))
        .filter(Expense.user_id == user_id, Expense.date >= start, Expense.date <= end)
        .group_by(Expense.category).all()
    )
    return [{"category": c, "amount": round(float(a or 0), 2), "count": int(n)} for c, n, a in
            [(r[0], r[2], r[1]) for r in rows]]


def income_totals(db: Session, user_id: int, start: date, end: date):
    total = db.query(func.coalesce(func.sum(Income.amount), 0.0)).filter(
        Income.user_id == user_id, Income.date >= start, Income.date <= end).scalar()
    return round(float(total), 2)


def expenses_in_range(db: Session, user_id: int, start: date, end: date, category: str | None = None):
    q = db.query(Expense).filter(Expense.user_id == user_id, Expense.date >= start, Expense.date <= end)
    if category:
        q = q.filter(Expense.category == category)
    return q.order_by(Expense.date.desc()).all()


def incomes_in_range(db: Session, user_id: int, start: date, end: date):
    return (db.query(Income)
            .filter(Income.user_id == user_id, Income.date >= start, Income.date <= end)
            .order_by(Income.date.desc()).all())


def monthly_summary(db: Session, user_id: int, year: int, month: int) -> dict:
    start, end = month_bounds(year, month)
    exp = expense_totals(db, user_id, start, end)
    total_exp = round(sum(e["amount"] for e in exp), 2)
    inc = income_totals(db, user_id, start, end)
    # count transactions
    n_tx = db.query(func.count(Expense.id)).filter(
        Expense.user_id == user_id, Expense.date >= start, Expense.date <= end).scalar()
    # previous month for comparison
    pm = month - 1 or 12
    py = year - 1 if month == 1 else year
    ps, pe = month_bounds(py, pm)
    prev_exp = round(sum(e["amount"] for e in expense_totals(db, user_id, ps, pe)), 2)
    prev_inc = income_totals(db, user_id, ps, pe)
    top = max(exp, key=lambda e: e["amount"], default=None)
    return {
        "year": year, "month": month,
        "start": start.isoformat(), "end": end.isoformat(),
        "total_income": inc,
        "total_expenses": total_exp,
        "net": round(inc - total_exp, 2),
        "savings_rate": round((inc - total_exp) / inc * 100, 1) if inc > 0 else 0.0,
        "transaction_count": int(n_tx or 0),
        "by_category": sorted(exp, key=lambda e: -e["amount"]),
        "top_category": top,
        "prev_month": {"expenses": prev_exp, "income": prev_inc,
                       "expense_delta": round(total_exp - prev_exp, 2),
                       "income_delta": round(inc - prev_inc, 2)},
    }


def trend(db: Session, user_id: int, months: int = 6) -> list[dict]:
    """Last N months of income vs expenses."""
    today = date.today()
    out = []
    y, m = today.year, today.month
    for _ in range(months):
        s = monthly_summary(db, user_id, y, m)
        out.append({"month": f"{y}-{m:02d}", "income": s["total_income"],
                    "expenses": s["total_expenses"], "net": s["net"]})
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def daily_series(db: Session, user_id: int, start: date, end: date):
    rows = (
        db.query(Expense.date, func.sum(Expense.amount))
        .filter(Expense.user_id == user_id, Expense.date >= start, Expense.date <= end)
        .group_by(Expense.date).all()
    )
    return [{"date": d.isoformat(), "amount": round(float(a), 2)} for d, a in rows]


def weekly_series(db: Session, user_id: int, start: date, end: date):
    rows = (
        db.query(func.strftime("%Y-%W", Expense.date), func.sum(Expense.amount))
        .filter(Expense.user_id == user_id, Expense.date >= start, Expense.date <= end)
        .group_by(func.strftime("%Y-%W", Expense.date)).all()
    )
    return [{"week": w, "amount": round(float(a), 2)} for w, a in rows]


def budget_status(db: Session, user_id: int, year: int, month: int) -> dict | None:
    budget = db.query(Budget).filter_by(user_id=user_id, year=year, month=month).first()
    if not budget:
        return None
    start, end = month_bounds(year, month)
    spent = {r.category: round(float(r.total or 0), 2) for r in
             db.query(Expense.category, func.sum(Expense.amount).label("total"))
             .filter(Expense.user_id == user_id, Expense.date >= start, Expense.date <= end)
             .group_by(Expense.category)}
    cats = []
    total_limit = 0.0
    total_spent = 0.0
    for bc in budget.categories:
        s = spent.get(bc.category, 0.0)
        total_limit += bc.limit_amount
        total_spent += s
        pct = round(s / bc.limit_amount * 100, 1) if bc.limit_amount > 0 else 0.0
        cats.append({"category": bc.category, "limit": bc.limit_amount, "spent": s,
                     "remaining": round(bc.limit_amount - s, 2), "percent": pct,
                     "essential": bc.essential, "over": s > bc.limit_amount})
    return {
        "id": budget.id, "month": month, "year": year,
        "total_income": budget.total_income, "total_budget": budget.total_budget,
        "savings_target": budget.savings_target,
        "categories": cats,
        "total_limit": round(total_limit, 2), "total_spent": round(total_spent, 2),
        "overall_percent": round(total_spent / total_limit * 100, 1) if total_limit > 0 else 0.0,
    }


def upcoming_bills(db: Session, user_id: int, days: int = 30) -> list[dict]:
    today = date.today()
    end = today + timedelta(days=days)
    # refresh overdue status
    overdue = db.query(Bill).filter(Bill.user_id == user_id, Bill.status == "upcoming",
                                    Bill.due_date < today).all()
    for b in overdue:
        b.status = "overdue"
    if overdue:
        db.commit()
    rows = (db.query(Bill).filter(Bill.user_id == user_id, Bill.status != "paid",
                                  Bill.due_date <= end)
            .order_by(Bill.due_date).all())
    return [{"id": b.id, "name": b.name, "amount": b.amount, "due_date": b.due_date.isoformat(),
             "status": "overdue" if b.due_date < today else b.status, "frequency": b.frequency}
            for b in rows]


def goals_progress(db: Session, user_id: int) -> list[dict]:
    goals = db.query(SavingsGoal).filter_by(user_id=user_id).all()
    out = []
    for g in goals:
        remaining = max(0.0, g.target_amount - g.saved_amount)
        pct = round(g.saved_amount / g.target_amount * 100, 1) if g.target_amount > 0 else 0.0
        months_left = None
        required_monthly = 0.0
        est_date = None
        if g.target_date:
            months_left = max(1, (g.target_date.year - date.today().year) * 12 +
                              (g.target_date.month - date.today().month))
            required_monthly = round(remaining / months_left, 2)
            est_date = g.target_date.isoformat()
        elif g.monthly_contribution and g.monthly_contribution > 0:
            months_left = int(remaining / g.monthly_contribution) + 1
            est = date.today() + timedelta(days=30 * months_left)
            est_date = est.isoformat()
        out.append({"id": g.id, "name": g.name, "target": g.target_amount,
                    "saved": g.saved_amount, "remaining": round(remaining, 2),
                    "percent": pct, "target_date": g.target_date.isoformat() if g.target_date else None,
                    "monthly_contribution": g.monthly_contribution,
                    "required_monthly": required_monthly, "months_left": months_left,
                    "estimated_completion": est_date})
    return out


def financial_profile(db: Session, user_id: int) -> dict:
    p = db.query(UserProfile).filter_by(user_id=user_id).first()
    if not p:
        return {}
    return {
        "currency": p.currency, "monthly_income": p.monthly_income,
        "income_source": p.income_source, "additional_income": p.additional_income,
        "monthly_rent": p.monthly_rent, "household_expenses": p.household_expenses,
        "transportation_expenses": p.transportation_expenses,
        "education_expenses": p.education_expenses, "loan_emi": p.loan_emi,
        "other_recurring": p.other_recurring,
        "savings_goal_amount": p.savings_goal_amount,
        "preferred_monthly_savings": p.preferred_monthly_savings,
        "onboarding_completed": p.onboarding_completed,
    }


def suggest_budget(db: Session, user_id: int, monthly_income: float) -> dict:
    """Rule-based 50/30/20 style allocation, editable by the user."""
    p = db.query(UserProfile).filter_by(user_id=user_id).first()
    essentials = 0.0
    if p:
        essentials = (p.monthly_rent or 0) + (p.household_expenses or 0) + \
                     (p.transportation_expenses or 0) + (p.education_expenses or 0) + \
                     (p.loan_emi or 0) + (p.other_recurring or 0)
    if monthly_income <= 0:
        monthly_income = (p.monthly_income if p else 0) or 0
    if monthly_income <= 0:
        return {"suggestions": [], "total_income": 0}
    needs = min(essential_or_default(essentials, monthly_income), monthly_income * 0.6)
    wants = monthly_income * 0.25
    savings = max(0.0, monthly_income - needs - wants)
    if p and p.preferred_monthly_savings:
        savings = min(p.preferred_monthly_savings, monthly_income - needs)
        wants = max(0.0, monthly_income - needs - savings)
    sugg = [
        {"category": "Housing & Rent", "amount": round(p.monthly_rent if p else 0, 2) or round(needs * 0.4, 2), "essential": True},
        {"category": "Food & Groceries", "amount": round(needs * 0.3, 2), "essential": True},
        {"category": "Utilities", "amount": round(needs * 0.1, 2), "essential": True},
        {"category": "Transportation", "amount": round((p.transportation_expenses if p else 0) or needs * 0.1, 2), "essential": True},
        {"category": "Education", "amount": round((p.education_expenses if p else 0), 2), "essential": True},
        {"category": "Healthcare", "amount": round(needs * 0.05, 2), "essential": True},
        {"category": "Shopping", "amount": round(wants * 0.3, 2), "essential": False},
        {"category": "Entertainment", "amount": round(wants * 0.25, 2), "essential": False},
        {"category": "Travel", "amount": round(wants * 0.2, 2), "essential": False},
        {"category": "Personal Care", "amount": round(wants * 0.15, 2), "essential": False},
        {"category": "Subscriptions", "amount": round(wants * 0.1, 2), "essential": False},
    ]
    return {"suggestions": sugg, "total_income": monthly_income,
            "savings_suggestion": round(savings, 2),
            "needs_total": round(needs, 2), "wants_total": round(wants, 2)}


def essential_or_default(essential: float, income: float) -> float:
    return essential if essential > 0 else income * 0.5


def categorize(description: str) -> str:
    """Keyword-based categorization for CSV imports and manual entries."""
    d = description.lower()
    rules = [
        ("food", ["grocery", "groceries", "food", "restaurant", "cafe", "coffee", "pizza", "swiggy", "zomato", "dmart", "reliance fresh", "vegetable", "milk", "lunch", "dinner", "breakfast", "meal", "tiffin", "snack", "canteen", "bakery", "biryani", "burger"]),
        ("transport", ["fuel", "petrol", "diesel", "uber", "ola", "cab", "metro", "bus", "train", "parking", "toll", "rapido"]),
        ("shopping", ["amazon", "flipkart", "myntra", "shopping", "clothes", "shoes", "mall"]),
        ("housing", ["rent", "housing", "maintenance", "society"]),
        ("utilities", ["electricity", "water", "gas", "broadband", "wifi", "recharge", "utility", "bill payment", "dth"]),
        ("healthcare", ["hospital", "medical", "pharmacy", "medicine", "doctor", "clinic", "lab"]),
        ("education", ["course", "tuition", "school", "college", "udemy", "books", "exam"]),
        ("entertainment", ["movie", "netflix", "spotify", "game", "entertainment", "concert", "ott"]),
        ("travel", ["flight", "hotel", "airbnb", "booking", "travel", "trip", "irctc"]),
        ("personal", ["salon", "spa", "gym", "personal care", "beauty"]),
        ("subscriptions", ["subscription", "prime", "membership", "netflix", "spotify", "youtube"]),
    ]
    for cat, keys in rules:
        for k in keys:
            if k in d:
                return {"food": "Food & Groceries", "transport": "Transportation",
                        "shopping": "Shopping", "housing": "Housing & Rent",
                        "utilities": "Utilities", "healthcare": "Healthcare",
                        "education": "Education", "entertainment": "Entertainment",
                        "travel": "Travel", "personal": "Personal Care",
                        "subscriptions": "Subscriptions"}[cat]
    return "Other"


# ---------- loan math (configurable assumptions for purchase planning) ----------

LOAN_ASSUMPTIONS = {
    "annual_interest_rate": 0.09,   # 9% p.a. — configurable
    "tenures_months": [36, 60, 84], # 3 / 5 / 7 years
    "down_payment_ratio": 0.20,     # suggested 20% down payment
    "safety_buffer_ratio": 0.20,    # keep 20% of income as unallocated buffer
    "emi_capacity_ratio": 0.50,     # max 50% of post-commitment surplus to new EMI
}


def emi_amount(principal: float, annual_rate: float, months: int) -> float:
    """Standard reducing-balance EMI."""
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def loan_scenarios(principal: float, rate: float | None = None,
                   tenures: list[int] | None = None) -> list[dict]:
    """EMI + total interest for each tenure."""
    rate = LOAN_ASSUMPTIONS["annual_interest_rate"] if rate is None else rate
    tenures = tenures or LOAN_ASSUMPTIONS["tenures_months"]
    out = []
    for m in tenures:
        emi = emi_amount(principal, rate, m)
        out.append({"tenure_months": m, "emi": round(emi, 2),
                    "total_payment": round(emi * m, 2),
                    "total_interest": round(emi * m - principal, 2)})
    return out
