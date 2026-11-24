from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ---------- auth ----------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    full_name: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class PasswordResetIn(BaseModel):
    email: EmailStr


class PasswordChangeIn(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6, max_length=128)


# ---------- profile / onboarding ----------
class ProfileIn(BaseModel):
    currency: Optional[str] = "INR"
    monthly_income: Optional[float] = 0
    income_source: Optional[str] = ""
    additional_income: Optional[float] = 0
    monthly_rent: Optional[float] = 0
    household_expenses: Optional[float] = 0
    transportation_expenses: Optional[float] = 0
    education_expenses: Optional[float] = 0
    loan_emi: Optional[float] = 0
    other_recurring: Optional[float] = 0
    savings_goal_amount: Optional[float] = 0
    preferred_monthly_savings: Optional[float] = 0
    onboarding_completed: Optional[bool] = False
    theme: Optional[str] = None
    notification_prefs: Optional[dict] = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    is_demo: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- finance ----------
class ExpenseIn(BaseModel):
    amount: float = Field(gt=0)
    date: date
    category: str = "Other"
    description: str = ""
    payment_method: str = ""
    notes: str = ""


class ExpenseOut(ExpenseIn):
    id: int
    source: str
    is_demo: bool
    created_at: datetime

    class Config:
        from_attributes = True


class IncomeIn(BaseModel):
    amount: float = Field(gt=0)
    date: date
    source: str = "Salary"
    description: str = ""
    is_recurring: bool = False


class IncomeOut(IncomeIn):
    id: int
    is_demo: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CategoryIn(BaseModel):
    name: str
    kind: str = "expense"
    color: str = "#10b981"


class BudgetCategoryIn(BaseModel):
    category: str
    limit_amount: float = Field(ge=0)
    essential: bool = False


class BudgetIn(BaseModel):
    month: int = Field(ge=1, le=12)
    year: int
    total_income: float = 0
    savings_target: float = 0
    categories: list[BudgetCategoryIn] = []


class BudgetOut(BaseModel):
    id: int
    month: int
    year: int
    total_income: float
    total_budget: float
    savings_target: float
    categories: list[dict]


class BillIn(BaseModel):
    name: str
    amount: float = Field(gt=0)
    due_date: date
    frequency: str = "monthly"
    notes: str = ""


class BillOut(BillIn):
    id: int
    status: str
    is_demo: bool

    class Config:
        from_attributes = True


class GoalIn(BaseModel):
    name: str
    target_amount: float = Field(gt=0)
    saved_amount: float = Field(default=0, ge=0)
    target_date: Optional[date] = None
    monthly_contribution: float = Field(default=0, ge=0)


class GoalOut(GoalIn):
    id: int
    is_demo: bool

    class Config:
        from_attributes = True


# ---------- chat ----------
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[int] = None


class ChatOut(BaseModel):
    session_id: int
    answer: str
    kind: str = "chat"
    agent: str = ""
    data: dict = {}


# ---------- import ----------
class ImportPreviewOut(BaseModel):
    headers: list[str]
    rows: list[list[str]]
    mapping: dict


class ImportConfirmIn(BaseModel):
    filename: str = "statement.csv"
    mapping: Optional[dict] = None
    corrections: Optional[list[dict]] = None
