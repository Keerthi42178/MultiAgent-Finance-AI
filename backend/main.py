import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.database.db import SessionLocal, init_db

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="FinPilot AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    if settings.seed_demo:
        _seed_demo_account()


def _seed_demo_account():
    from app.models.db_models import User
    from app.services.auth import create_user
    db = SessionLocal()
    try:
        email = "demo@finpilot.ai"
        if not db.query(User).filter_by(email=email).first():
            user = create_user(db, email, "demo1234", "Demo User")
            user.is_demo = True
            db.commit()
            from datetime import date, timedelta
            from app.models.db_models import Bill, Expense, SavingsGoal, Income
            today = date.today()
            cats = ["Food & Groceries", "Transportation", "Shopping", "Utilities",
                    "Entertainment", "Subscriptions", "Healthcare", "Travel"]
            amounts = [320, 450, 890, 1200, 560, 199, 780, 2400, 350, 1500, 420, 99]
            for i in range(24):
                d = today - timedelta(days=i * 3)
                db.add(Expense(user_id=user.id, amount=amounts[i % len(amounts)], date=d,
                               category=cats[i % len(cats)], description=f"Demo expense {i+1}",
                               payment_method="UPI", source="manual", is_demo=True))
            for i in range(3):
                mm = today.month - i
                yy = today.year
                while mm <= 0:
                    mm += 12
                    yy -= 1
                db.add(Income(user_id=user.id, amount=30000 + i * 1000,
                              date=date(yy, mm, 1), source="Salary",
                              description=f"Demo salary {yy}-{mm:02d}", is_demo=True))
            db.add(Bill(user_id=user.id, name="Demo Electricity", amount=1450,
                        due_date=today + timedelta(days=5), frequency="monthly", is_demo=True))
            db.add(SavingsGoal(user_id=user.id, name="Demo Emergency Fund",
                               target_amount=100000, saved_amount=25000,
                               monthly_contribution=5000, is_demo=True))
            db.commit()
            print("Seeded demo account: demo@finpilot.ai / demo1234")
    finally:
        db.close()


app.include_router(api_router)


@app.get("/")
def root():
    return {"name": "FinPilot AI API", "docs": "/docs", "version": "1.0.0"}
