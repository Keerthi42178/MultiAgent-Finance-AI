"""REST API routes for FinPilot AI."""
import csv
import io
import json
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.db import get_db
from app.memory import blackboard
from app.models.db_models import (
    AgentExecutionHistory, AuditLog, Bill, Budget, BudgetCategory, ConversationMessage,
    ConversationSession, Expense, ExpenseCategory, ImportHistory, Income,
    Notification, PolicyActionHistory, SavingsGoal, TaskExecution, User, UserProfile,
)
from app.integrations import gemini
from app.orchestration import orchestrator
from app.reinforcement_learning.actor_critic import ACTIONS, get_policy
from app.schemas import schemas as S
from app.services import csv_import, notifications
from app.services.auth import (
    audit, create_access_token, create_user, get_current_user, seed_categories,
    verify_password, hash_password,
)
from app.services.finance import (
    EXPENSE_CATEGORIES, budget_status, categorize, daily_series, expense_totals,
    expenses_in_range, financial_profile, goals_progress, income_totals,
    incomes_in_range, monthly_summary, parse_month, suggest_budget, trend,
    upcoming_bills, weekly_series,
)

router = APIRouter(prefix="/api")


def _user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "full_name": u.full_name,
            "is_demo": u.is_demo, "created_at": u.created_at.isoformat()}


# ================= AUTH =================

@router.post("/auth/register", response_model=S.TokenOut)
def register(body: S.RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(func.lower(User.email) == body.email.lower()).first():
        raise HTTPException(409, "Email already registered")
    user = create_user(db, body.email, body.password, body.full_name)
    audit(db, user.id, "register", ip="")
    return {"access_token": create_access_token(user.id), "user": _user_out(user)}


@router.post("/auth/login", response_model=S.TokenOut)
def login(body: S.LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(func.lower(User.email) == body.email.lower()).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")
    audit(db, user.id, "login")
    return {"access_token": create_access_token(user.id), "user": _user_out(user)}


@router.post("/auth/logout")
def logout(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audit(db, user.id, "logout")
    return {"ok": True}


@router.post("/auth/forgot-password")
def forgot_password(body: S.PasswordResetIn, db: Session = Depends(get_db)):
    # No email provider configured: return a reset token the user can use via /auth/reset-password.
    # In production, email this token instead.
    user = db.query(User).filter(func.lower(User.email) == body.email.lower()).first()
    if not user:
        return {"ok": True, "message": "If the email exists, a reset token was generated."}
    token = create_access_token(user.id)
    return {"ok": True, "reset_token": token,
            "message": "Use this token with /auth/reset-password (dev mode: email delivery not configured)."}


@router.post("/auth/reset-password")
def reset_password(token: str = Query(...), new_password: str = Query(..., min_length=6),
                   db: Session = Depends(get_db)):
    import jwt as pyjwt
    try:
        payload = pyjwt.decode(token, settings.secret_key, algorithms=["HS256"])
        uid = int(payload["sub"])
    except Exception:
        raise HTTPException(400, "Invalid or expired token")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(404, "User not found")
    user.hashed_password = hash_password(new_password)
    db.commit()
    audit(db, user.id, "password_reset")
    return {"ok": True}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {"user": _user_out(user), "profile": financial_profile(db, user.id)}


# ================= PROFILE / ONBOARDING =================

@router.get("/profile")
def get_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = db.query(UserProfile).filter_by(user_id=user.id).first()
    return financial_profile(db, user.id) | {"theme": p.theme if p else "dark"}


@router.put("/profile")
def update_profile(body: S.ProfileIn, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    p = db.query(UserProfile).filter_by(user_id=user.id).first()
    if not p:
        p = UserProfile(user_id=user.id)
        db.add(p)
    data = body.model_dump(exclude_unset=True)
    prefs = data.pop("notification_prefs", None)
    for k, v in data.items():
        if hasattr(p, k):
            setattr(p, k, v)
    if prefs is not None:
        p.notification_prefs = json.dumps(prefs)
    if body.full_name if hasattr(body, "full_name") else False:
        pass
    db.commit()
    audit(db, user.id, "profile_update")
    return financial_profile(db, user.id)


@router.put("/profile/name")
def update_name(full_name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.full_name = full_name
    db.commit()
    return {"ok": True}


@router.get("/profile/budget-suggestion")
def budget_suggestion(monthly_income: float = Query(0),
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return suggest_budget(db, user.id, monthly_income)


# ================= DASHBOARD =================

@router.get("/dashboard")
def dashboard(month: Optional[str] = None, year: Optional[int] = None,
              user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    y, m = parse_month(month, year, db, user.id)
    s, e = _bounds(y, m)
    summary = monthly_summary(db, user.id, y, m)
    return {
        "summary": summary,
        "budget": budget_status(db, user.id, y, m),
        "bills": upcoming_bills(db, user.id, days=30),
        "goals": goals_progress(db, user.id),
        "trend": trend(db, user.id, 6),
        "daily": daily_series(db, user.id, s, e),
        "weekly": weekly_series(db, user.id, s, e),
        "profile": financial_profile(db, user.id),
        "notifications_unread": db.query(func.count(Notification.id)).filter(
            Notification.user_id == user.id, Notification.is_read == False).scalar(),  # noqa: E712
    }


def _bounds(y, m):
    import calendar
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


# ================= EXPENSES =================

@router.get("/expenses")
def list_expenses(start: Optional[date] = None, end: Optional[date] = None,
                  category: Optional[str] = None, min_amount: Optional[float] = None,
                  max_amount: Optional[float] = None, payment_method: Optional[str] = None,
                  q: Optional[str] = None, limit: int = Query(500, le=2000), offset: int = 0,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    today = date.today()
    s = start or date(today.year - 1, today.month, 1)
    e = end or date(today.year, today.month, monthrange(today.year, today.month)[1])
    query = db.query(Expense).filter(Expense.user_id == user.id,
                                     Expense.date >= s, Expense.date <= e)
    if category:
        query = query.filter(Expense.category == category)
    if min_amount is not None:
        query = query.filter(Expense.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(Expense.amount <= max_amount)
    if payment_method:
        query = query.filter(Expense.payment_method.ilike(f"%{payment_method}%"))
    if q:
        query = query.filter(Expense.description.ilike(f"%{q}%"))
    total = query.count()
    rows = query.order_by(Expense.date.desc(), Expense.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [S.ExpenseOut.model_validate(r) for r in rows]}


@router.post("/expenses", response_model=S.ExpenseOut)
def create_expense(body: S.ExpenseIn, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    cat = body.category if body.category in EXPENSE_CATEGORIES else categorize(body.description) or "Other"
    exp = Expense(user_id=user.id, amount=body.amount, date=body.date, category=cat,
                  description=body.description, payment_method=body.payment_method,
                  notes=body.notes, source="manual")
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.put("/expenses/{eid}", response_model=S.ExpenseOut)
def update_expense(eid: int, body: S.ExpenseIn, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    exp = db.query(Expense).filter_by(id=eid, user_id=user.id).first()
    if not exp:
        raise HTTPException(404, "Expense not found")
    for k, v in body.model_dump().items():
        setattr(exp, k, v)
    db.commit()
    db.refresh(exp)
    return exp


@router.delete("/expenses/{eid}")
def delete_expense(eid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    exp = db.query(Expense).filter_by(id=eid, user_id=user.id).first()
    if not exp:
        raise HTTPException(404, "Expense not found")
    db.delete(exp)
    db.commit()
    return {"ok": True}


@router.post("/expenses/scan")
async def scan_expense_image(file: UploadFile = File(...),
                             user: User = Depends(get_current_user)):
    """Analyze a receipt/screenshot with Gemini vision -> candidate expenses.
    Nothing is saved here; the client reviews then bulk-imports."""
    if not gemini.available():
        raise HTTPException(503, "Image scanning requires GEMINI_API_KEY "
                                 "configured in backend/.env")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 8 MB)")
    mime = file.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPG/PNG)")
    raw = await gemini.analyze_receipt_image(data, mime)
    if raw is None:
        raise HTTPException(502, "Could not analyze the image. Try a clearer photo.")
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").removeprefix("json").strip()
    try:
        items = json.loads(txt)
        if isinstance(items, dict):
            items = items.get("expenses") or items.get("transactions") or []
        if not isinstance(items, list):
            items = []
    except Exception:
        raise HTTPException(502, "Could not parse the analysis result")
    out = []
    today = date.today()
    for it in items[:50]:
        if not isinstance(it, dict):
            continue
        try:
            amt = float(it.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        if amt <= 0:
            continue
        d = it.get("date")
        try:
            d = datetime.strptime(str(d)[:10], "%Y-%m-%d").date() if d else today
        except Exception:
            d = today
        cat = str(it.get("category") or "Other")
        if cat not in EXPENSE_CATEGORIES:
            cat = "Other"
        out.append({"amount": amt, "date": d.isoformat(), "category": cat,
                    "description": str(it.get("description") or "")[:200],
                    "payment_method": str(it.get("payment_method") or "")[:50],
                    "notes": "Scanned from image"})
    return {"items": out, "count": len(out)}


@router.post("/expenses/bulk")
def create_expenses_bulk(body: list[S.ExpenseIn],
                         user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    """Save reviewed expenses (e.g. confirmed receipt scan) in one call."""
    created = 0
    for e in body[:100]:
        cat = e.category if e.category in EXPENSE_CATEGORIES else categorize(e.description) or "Other"
        db.add(Expense(user_id=user.id, amount=e.amount, date=e.date, category=cat,
                       description=e.description, payment_method=e.payment_method,
                       notes=e.notes, source="image_scan"))
        created += 1
    db.commit()
    return {"ok": True, "created": created}


@router.get("/expenses/spending-by-day")
def spending_by_day(start: date, end: date, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    return daily_series(db, user.id, start, end)


# ================= INCOME =================

@router.get("/income")
def list_income(start: Optional[date] = None, end: Optional[date] = None,
               source: Optional[str] = None,
               user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    today = date.today()
    s = start or date(today.year - 1, 1, 1)
    e = end or date.today()
    q = db.query(Income).filter(Income.user_id == user.id, Income.date >= s, Income.date <= e)
    if source:
        q = q.filter(Income.source == source)
    rows = q.order_by(Income.date.desc()).all()
    return {"items": [S.IncomeOut.model_validate(r) for r in rows],
            "total": round(sum(r.amount for r in rows), 2)}


@router.post("/income", response_model=S.IncomeOut)
def create_income(body: S.IncomeIn, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    inc = Income(user_id=user.id, **body.model_dump())
    db.add(inc)
    db.commit()
    db.refresh(inc)
    return inc


@router.put("/income/{iid}", response_model=S.IncomeOut)
def update_income(iid: int, body: S.IncomeIn, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    inc = db.query(Income).filter_by(id=iid, user_id=user.id).first()
    if not inc:
        raise HTTPException(404, "Income not found")
    for k, v in body.model_dump().items():
        setattr(inc, k, v)
    db.commit()
    db.refresh(inc)
    return inc


@router.delete("/income/{iid}")
def delete_income(iid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    inc = db.query(Income).filter_by(id=iid, user_id=user.id).first()
    if not inc:
        raise HTTPException(404, "Income not found")
    db.delete(inc)
    db.commit()
    return {"ok": True}


# ================= CATEGORIES =================

@router.get("/categories")
def list_categories(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(ExpenseCategory).filter_by(user_id=user.id).all()
    return [{"id": c.id, "name": c.name, "kind": c.kind, "color": c.color,
             "is_custom": c.is_custom, "is_system": c.is_system} for c in rows]


@router.post("/categories")
def create_category(body: S.CategoryIn, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    if db.query(ExpenseCategory).filter_by(user_id=user.id, name=body.name).first():
        raise HTTPException(409, "Category already exists")
    c = ExpenseCategory(user_id=user.id, name=body.name, kind=body.kind,
                        color=body.color, is_custom=True)
    db.add(c)
    db.commit()
    return {"id": c.id, "name": c.name, "kind": c.kind, "color": c.color}


@router.delete("/categories/{cid}")
def delete_category(cid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = db.query(ExpenseCategory).filter_by(id=cid, user_id=user.id, is_custom=True).first()
    if not c:
        raise HTTPException(404, "Custom category not found")
    db.delete(c)
    db.commit()
    return {"ok": True}


# ================= BUDGETS =================

@router.get("/budgets")
def list_budgets(year: Optional[int] = None, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    q = db.query(Budget).filter_by(user_id=user.id)
    if year:
        q = q.filter(Budget.year == year)
    return [{"id": b.id, "month": b.month, "year": b.year,
             "total_income": b.total_income, "total_budget": b.total_budget,
             "savings_target": b.savings_target,
             "status": budget_status(db, user.id, b.year, b.month)}
            for b in q.order_by(Budget.year.desc(), Budget.month.desc()).all()]


@router.get("/budgets/current")
def current_budget(month: Optional[str] = None, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    y, m = parse_month(month, None, db, user.id)
    return budget_status(db, user.id, y, m)


@router.put("/budgets/{year}/{month}")
def upsert_budget(year: int, month: int, body: S.BudgetIn,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    budget = db.query(Budget).filter_by(user_id=user.id, year=year, month=month).first()
    if not budget:
        budget = Budget(user_id=user.id, year=year, month=month)
        db.add(budget)
        db.flush()
    budget.total_income = body.total_income
    budget.savings_target = body.savings_target
    budget.total_budget = round(sum(c.limit_amount for c in body.categories), 2)
    db.query(BudgetCategory).filter_by(budget_id=budget.id).delete()
    for c in body.categories:
        db.add(BudgetCategory(budget_id=budget.id, category=c.category,
                              limit_amount=c.limit_amount, essential=c.essential))
    db.commit()
    audit(db, user.id, "budget_upsert", f"{year}-{month}")
    return budget_status(db, user.id, year, month)


# ================= BILLS =================

@router.get("/bills")
def list_bills(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    notifications.generate_due_notifications(db, user.id)
    rows = db.query(Bill).filter_by(user_id=user.id).order_by(Bill.due_date).all()
    today = date.today()
    return [{"id": b.id, "name": b.name, "amount": b.amount,
             "due_date": b.due_date.isoformat(), "frequency": b.frequency,
             "status": "overdue" if b.due_date < today and b.status != "paid" else b.status,
             "notes": b.notes, "is_demo": b.is_demo} for b in rows]


@router.post("/bills", response_model=S.BillOut)
def create_bill(body: S.BillIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    b = Bill(user_id=user.id, **body.model_dump())
    db.add(b)
    db.commit()
    db.refresh(b)
    return b


@router.put("/bills/{bid}", response_model=S.BillOut)
def update_bill(bid: int, body: S.BillIn, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    b = db.query(Bill).filter_by(id=bid, user_id=user.id).first()
    if not b:
        raise HTTPException(404, "Bill not found")
    for k, v in body.model_dump().items():
        setattr(b, k, v)
    db.commit()
    db.refresh(b)
    return b


@router.post("/bills/{bid}/pay")
def pay_bill(bid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    b = db.query(Bill).filter_by(id=bid, user_id=user.id).first()
    if not b:
        raise HTTPException(404, "Bill not found")
    b.status = "paid"
    db.commit()
    return {"ok": True}


@router.delete("/bills/{bid}")
def delete_bill(bid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    b = db.query(Bill).filter_by(id=bid, user_id=user.id).first()
    if not b:
        raise HTTPException(404, "Bill not found")
    db.delete(b)
    db.commit()
    return {"ok": True}


# ================= SAVINGS GOALS =================

@router.get("/goals")
def list_goals(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return goals_progress(db, user.id)


@router.post("/goals", response_model=S.GoalOut)
def create_goal(body: S.GoalIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    g = SavingsGoal(user_id=user.id, **body.model_dump())
    db.add(g)
    db.commit()
    db.refresh(g)
    return g


@router.put("/goals/{gid}", response_model=S.GoalOut)
def update_goal(gid: int, body: S.GoalIn, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    g = db.query(SavingsGoal).filter_by(id=gid, user_id=user.id).first()
    if not g:
        raise HTTPException(404, "Goal not found")
    for k, v in body.model_dump().items():
        setattr(g, k, v)
    db.commit()
    db.refresh(g)
    return g


@router.post("/goals/{gid}/contribute")
def contribute_goal(gid: int, amount: float = Query(gt=0),
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    g = db.query(SavingsGoal).filter_by(id=gid, user_id=user.id).first()
    if not g:
        raise HTTPException(404, "Goal not found")
    g.saved_amount = (g.saved_amount or 0) + amount
    db.commit()
    notifications.generate_due_notifications(db, user.id)
    return {"ok": True, "saved_amount": g.saved_amount}


@router.delete("/goals/{gid}")
def delete_goal(gid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    g = db.query(SavingsGoal).filter_by(id=gid, user_id=user.id).first()
    if not g:
        raise HTTPException(404, "Goal not found")
    db.delete(g)
    db.commit()
    return {"ok": True}


# ================= CSV IMPORT =================

@router.post("/import/preview")
async def import_preview(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    content = (await file.read()).decode("utf-8-sig", errors="replace")
    prev = csv_import.preview_csv(content)
    # stash content server-side? For simplicity client re-uploads on confirm.
    return prev


@router.post("/import/confirm")
async def import_confirm(file: UploadFile = File(...),
                         mapping: Optional[str] = Form(None),
                         corrections: Optional[str] = Form(None),
                         user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    content = (await file.read()).decode("utf-8-sig", errors="replace")
    map_dict = json.loads(mapping) if mapping else None
    corr = json.loads(corrections) if corrections else None
    result = csv_import.parse_and_import(db, user.id, content, file.filename or "statement.csv",
                                         mapping=map_dict, corrections=corr)
    if "error" in result:
        raise HTTPException(400, result["error"])
    notifications.notify(db, user.id, "import",
                         f"Import complete: {result['imported']} transactions",
                         f"₹{result['expense_total']:,.0f} expenses, ₹{result['income_total']:,.0f} income imported.")
    db.commit()
    audit(db, user.id, "csv_import", file.filename or "")
    return result


@router.get("/import/history")
def import_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(ImportHistory).filter_by(user_id=user.id).order_by(ImportHistory.id.desc()).limit(50).all()
    return [{"id": r.id, "filename": r.filename, "total_rows": r.total_rows,
             "imported_rows": r.imported_rows, "skipped_rows": r.skipped_rows,
             "duplicate_rows": r.duplicate_rows, "summary": json.loads(r.summary or "{}"),
             "created_at": r.created_at.isoformat()} for r in rows]


# ================= NOTIFICATIONS =================

@router.get("/notifications")
def list_notifications(unread_only: bool = False, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    notifications.generate_due_notifications(db, user.id)
    rows = notifications.list_notifications(db, user.id, unread_only)
    return [{"id": n.id, "kind": n.kind, "title": n.title, "body": n.body,
             "is_read": n.is_read, "created_at": n.created_at.isoformat()} for n in rows]


@router.post("/notifications/read")
def mark_notifications_read(id: Optional[int] = None,
                            user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    n = notifications.mark_read(db, user.id, id)
    return {"ok": True, "updated": n}


@router.delete("/notifications/{notification_id}")
def delete_notification(notification_id: int, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    if not notifications.delete_notification(db, user.id, notification_id):
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"ok": True}


# ================= REPORTS =================

@router.get("/reports/monthly")
def monthly_report(month: Optional[str] = None, year: Optional[int] = None,
                   user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    y, m = parse_month(month, year, db, user.id)
    s = monthly_summary(db, user.id, y, m)
    bs = budget_status(db, user.id, y, m)
    return {"period": f"{y}-{m:02d}", "summary": s, "budget": bs,
            "trend": trend(db, user.id, 6), "goals": goals_progress(db, user.id)}


@router.get("/reports/trend")
def reports_trend(months: int = Query(6, le=24), user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    return trend(db, user.id, months)


@router.get("/reports/export.csv")
def export_csv(month: Optional[str] = None, year: Optional[int] = None,
               user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    y, m = parse_month(month, year, db, user.id)
    s, e = _bounds(y, m)
    txns = expenses_in_range(db, user.id, s, e)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "description", "category", "amount", "payment_method", "source"])
    for t in txns:
        w.writerow([t.date.isoformat(), t.description, t.category, t.amount,
                    t.payment_method, t.source])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":
                                      f"attachment; filename=finpilot-{y}-{m:02d}.csv"})


# ================= CHAT / AI ASSISTANT =================

@router.get("/chat/sessions")
def chat_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(ConversationSession).filter_by(user_id=user.id)\
        .order_by(ConversationSession.updated_at.desc()).limit(50).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at.isoformat(),
             "updated_at": s.updated_at.isoformat(),
             "message_count": len(s.messages)} for s in rows]


@router.post("/chat/sessions")
def create_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = ConversationSession(user_id=user.id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"id": s.id}


@router.get("/chat/sessions/{sid}")
def get_session(sid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.query(ConversationSession).filter_by(id=sid, user_id=user.id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    return {"id": s.id, "title": s.title,
            "messages": [{"id": m.id, "role": m.role, "content": m.content,
                          "data": json.loads(m.data or "{}"), "agent": m.agent,
                          "created_at": m.created_at.isoformat()} for m in s.messages]}


@router.delete("/chat/sessions/{sid}")
def delete_session(sid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.query(ConversationSession).filter_by(id=sid, user_id=user.id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    db.delete(s)
    db.commit()
    return {"ok": True}


@router.post("/chat/message")
async def chat_message(body: S.ChatIn, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    if body.session_id:
        session = db.query(ConversationSession).filter_by(id=body.session_id, user_id=user.id).first()
        if not session:
            raise HTTPException(404, "Session not found")
    else:
        session = ConversationSession(user_id=user.id, title=body.message[:60])
        db.add(session)
        db.commit()
        db.refresh(session)
    result = await orchestrator.handle_message(db, user, session, body.message)
    return {"session_id": session.id, "answer": result.get("answer", ""),
            "kind": result.get("kind", "chat"), "agent": result.get("agent", ""),
            "data": {k: v for k, v in result.items() if k not in ("answer",)}}


# ================= AGENTS / POLICY =================

@router.get("/agents")
def agents_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(AgentExecutionHistory).filter_by(user_id=user.id)\
        .order_by(AgentExecutionHistory.id.desc()).limit(50).all()
    from app.agents import AGENTS
    return {
        "agents": [{"name": k, "description": v(db).description} for k, v in AGENTS.items()],
        "recent_executions": [{"id": r.id, "agent": r.agent, "action": r.action,
                               "success": r.success, "latency_ms": round(r.latency_ms, 1),
                               "created_at": r.created_at.isoformat()} for r in rows],
    }


@router.get("/policy")
def policy_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    pol = get_policy()
    hist = db.query(PolicyActionHistory).filter_by(user_id=user.id)\
        .order_by(PolicyActionHistory.id.desc()).limit(50).all()
    return {"policy": pol.status(), "actions": ACTIONS,
            "recent_decisions": [{"id": h.id, "action": h.action, "critic_value": h.critic_value,
                                  "reward": h.reward, "source": h.source,
                                  "created_at": h.created_at.isoformat()} for h in hist]}


@router.get("/blackboard")
def blackboard_state(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    state = blackboard.refresh_state(db, user)
    dialogue = blackboard.get_dialogue(db, user)
    outputs = blackboard.get_agent_outputs(db, user)
    prefs = blackboard.get_preferences(db, user)
    return {"state": state, "dialogue": dialogue, "agent_outputs": outputs, "preferences": prefs}


@router.get("/tasks/{task_key}")
def task_trace(task_key: str, user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    """Agent execution trace for a decomposed multi-agent task."""
    row = blackboard.task_get(db, user, task_key)
    if not row:
        raise HTTPException(404, "Task not found")
    tv = blackboard.task_view(row)
    return {"task_key": row.task_key, "status": row.status,
            "original_request": row.original_request,
            "primary_goal": row.primary_goal,
            "subtasks": tv["subtasks"], "results": tv["results"],
            "trace": tv["trace"], "error": row.error,
            "final_result": json.loads(row.final_result) if row.final_result else None,
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat()}


@router.get("/tasks")
def task_list(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(TaskExecution).filter_by(user_id=user.id)\
        .order_by(TaskExecution.id.desc()).limit(20).all()
    return {"tasks": [{"task_key": r.task_key, "status": r.status,
                       "primary_goal": r.primary_goal,
                       "original_request": r.original_request[:120],
                       "created_at": r.created_at.isoformat()} for r in rows]}


# ================= SETTINGS / DATA =================

@router.get("/settings/export")
def export_data(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = {
        "user": _user_out(user),
        "profile": financial_profile(db, user.id),
        "expenses": [e.__dict__ for e in db.query(Expense).filter_by(user_id=user.id).all()],
        "incomes": [i.__dict__ for i in db.query(Income).filter_by(user_id=user.id).all()],
        "budgets": [b.__dict__ for b in db.query(Budget).filter_by(user_id=user.id).all()],
        "bills": [b.__dict__ for b in db.query(Bill).filter_by(user_id=user.id).all()],
        "goals": [g.__dict__ for g in db.query(SavingsGoal).filter_by(user_id=user.id).all()],
    }
    for coll in ("expenses", "incomes", "budgets", "bills", "goals"):
        for r in data[coll]:
            r.pop("_sa_instance_state", None)
            for k, v in list(r.items()):
                if isinstance(v, (datetime, date)):
                    r[k] = v.isoformat()
    return data


@router.delete("/settings/account")
def delete_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audit(db, user.id, "account_delete")
    db.delete(user)
    db.commit()
    return {"ok": True}


@router.get("/audit-logs")
def audit_logs(limit: int = Query(100, le=500), user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    rows = db.query(AuditLog).filter_by(user_id=user.id).order_by(AuditLog.id.desc()).limit(limit).all()
    return [{"id": r.id, "action": r.action, "detail": r.detail,
             "created_at": r.created_at.isoformat()} for r in rows]


# ================= DEMO =================

@router.post("/demo/seed")
def seed_demo(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Seed clearly-labelled synthetic demo data for the current user."""
    today = date.today()
    y, m = today.year, today.month
    cats = ["Food & Groceries", "Transportation", "Shopping", "Utilities",
            "Entertainment", "Subscriptions", "Healthcare", "Travel"]
    amounts = [320, 450, 890, 1200, 560, 199, 780, 2400, 350, 1500, 420, 99]
    for i in range(24):
        d = today - timedelta(days=i * 3)
        db.add(Expense(user_id=user.id, amount=amounts[i % len(amounts)], date=d,
                       category=cats[i % len(cats)], description=f"Demo expense {i+1}",
                       payment_method="UPI", source="manual", is_demo=True))
    for i in range(3):
        mm = m - i
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        db.add(Income(user_id=user.id, amount=30000 + i * 1000,
                      date=date(yy, mm, 1), source="Salary",
                      description=f"Demo salary {yy}-{mm:02d}", is_demo=True))
    db.add(Bill(user_id=user.id, name="Demo Electricity", amount=1450,
                due_date=today + timedelta(days=5), frequency="monthly", is_demo=True))
    db.add(Bill(user_id=user.id, name="Demo Internet", amount=999,
                due_date=today + timedelta(days=12), frequency="monthly", is_demo=True))
    db.add(SavingsGoal(user_id=user.id, name="Demo Emergency Fund", target_amount=100000,
                       saved_amount=25000, monthly_contribution=5000, is_demo=True))
    db.commit()
    return {"ok": True, "message": "Demo data seeded (all records labelled is_demo=True)."}
