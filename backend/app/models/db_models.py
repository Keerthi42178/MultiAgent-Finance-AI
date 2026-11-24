from datetime import datetime, date

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), default="")
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_demo = Column(Boolean, default=False)  # labelled demo account
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    expenses = relationship("Expense", back_populates="user", cascade="all, delete-orphan")
    incomes = relationship("Income", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    bills = relationship("Bill", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("SavingsGoal", back_populates="user", cascade="all, delete-orphan")
    categories = relationship("ExpenseCategory", back_populates="user", cascade="all, delete-orphan")
    imports = relationship("ImportHistory", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    blackboard = relationship("BlackboardMemory", back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    currency = Column(String(8), default="INR")
    monthly_income = Column(Float, default=0.0)
    income_source = Column(String(120), default="")
    additional_income = Column(Float, default=0.0)
    monthly_rent = Column(Float, default=0.0)
    household_expenses = Column(Float, default=0.0)
    transportation_expenses = Column(Float, default=0.0)
    education_expenses = Column(Float, default=0.0)
    loan_emi = Column(Float, default=0.0)
    other_recurring = Column(Float, default=0.0)
    savings_goal_amount = Column(Float, default=0.0)
    preferred_monthly_savings = Column(Float, default=0.0)
    onboarding_completed = Column(Boolean, default=False)
    theme = Column(String(16), default="dark")
    notification_prefs = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")


class ExpenseCategory(Base):
    __tablename__ = "expense_categories"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_category"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(120), nullable=False)
    kind = Column(String(16), default="expense")  # expense | income
    icon = Column(String(64), default="")
    color = Column(String(16), default="#10b981")
    is_custom = Column(Boolean, default=False)
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="categories")


class Expense(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    date = Column(Date, nullable=False, index=True)
    category = Column(String(120), nullable=False, default="Other")
    description = Column(String(500), default="")
    payment_method = Column(String(64), default="")
    notes = Column(Text, default="")
    source = Column(String(32), default="manual")  # manual | csv_import | onboarding_estimate
    import_id = Column(Integer, ForeignKey("import_histories.id", ondelete="SET NULL"), nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="expenses")
    import_record = relationship("ImportHistory")


class Income(Base):
    __tablename__ = "incomes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    date = Column(Date, nullable=False, index=True)
    source = Column(String(120), default="Salary")  # Salary | Freelance | Business | Interest | Other
    description = Column(String(500), default="")
    is_recurring = Column(Boolean, default=False)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="incomes")


class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    month = Column(Integer, nullable=False)          # 1-12
    year = Column(Integer, nullable=False)
    total_income = Column(Float, default=0.0)
    total_budget = Column(Float, default=0.0)
    savings_target = Column(Float, default=0.0)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="budgets")
    categories = relationship("BudgetCategory", back_populates="budget", cascade="all, delete-orphan")


class BudgetCategory(Base):
    __tablename__ = "budget_categories"
    __table_args__ = (UniqueConstraint("budget_id", "category", name="uq_budget_category"),)

    id = Column(Integer, primary_key=True)
    budget_id = Column(Integer, ForeignKey("budgets.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(120), nullable=False)
    limit_amount = Column(Float, default=0.0)
    essential = Column(Boolean, default=False)

    budget = relationship("Budget", back_populates="categories")


class Bill(Base):
    __tablename__ = "bills"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(160), nullable=False)
    amount = Column(Float, nullable=False)
    due_date = Column(Date, nullable=False)
    frequency = Column(String(16), default="monthly")  # monthly | yearly | one_time
    status = Column(String(16), default="upcoming")    # upcoming | paid | overdue
    notes = Column(Text, default="")
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="bills")


class SavingsGoal(Base):
    __tablename__ = "savings_goals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(160), nullable=False)
    target_amount = Column(Float, nullable=False)
    saved_amount = Column(Float, default=0.0)
    target_date = Column(Date, nullable=True)
    monthly_contribution = Column(Float, default=0.0)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="goals")


class ImportHistory(Base):
    __tablename__ = "import_histories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), default="")
    total_rows = Column(Integer, default=0)
    imported_rows = Column(Integer, default=0)
    skipped_rows = Column(Integer, default=0)
    duplicate_rows = Column(Integer, default=0)
    status = Column(String(16), default="completed")
    summary = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="imports")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind = Column(String(48), default="info")
    title = Column(String(255), nullable=False)
    body = Column(Text, default="")
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="notifications")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="New conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="ConversationMessage.created_at")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    data = Column(Text, default="{}")  # structured payload (charts, cards, agent info)
    agent = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ConversationSession", back_populates="messages")


class BlackboardMemory(Base):
    """Persistent per-user shared memory for the multi-agent system."""
    __tablename__ = "blackboard_memories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    state = Column(Text, default="{}")          # financial profile snapshot
    dialogue = Column(Text, default="{}")       # current dialogue state, missing slots
    agent_outputs = Column(Text, default="{}")  # latest outputs per agent
    preferences = Column(Text, default="{}")    # user prefs learned from chat
    action_history = Column(Text, default="[]")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="blackboard")


class AgentExecutionHistory(Base):
    __tablename__ = "agent_execution_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id", ondelete="SET NULL"), nullable=True)
    agent = Column(String(64), nullable=False)
    action = Column(String(128), default="")
    input_summary = Column(Text, default="")
    output_summary = Column(Text, default="")
    success = Column(Boolean, default=True)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PolicyActionHistory(Base):
    __tablename__ = "policy_action_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id", ondelete="SET NULL"), nullable=True)
    state = Column(Text, default="[]")
    probs = Column(Text, default="[]")
    action = Column(String(64), default="")
    critic_value = Column(Float, default=0.0)
    reward = Column(Float, nullable=True)
    source = Column(String(16), default="policy")  # policy | rule_baseline
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    action = Column(String(120), nullable=False)
    detail = Column(Text, default="")
    ip = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class TaskExecution(Base):
    """A decomposed multi-agent task: one user goal split into dependent subtasks
    executed collaboratively by the specialized agents."""
    __tablename__ = "task_executions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id", ondelete="SET NULL"), nullable=True)
    task_key = Column(String(40), nullable=False)          # short public id e.g. T-8f3a21
    original_request = Column(Text, default="")
    intent = Column(String(64), default="")
    primary_goal = Column(String(64), default="")
    status = Column(String(16), default="pending")         # pending|running|completed|failed|clarify
    subtasks = Column(Text, default="[]")                  # JSON list of subtask dicts
    intermediate_results = Column(Text, default="{}")      # JSON {agent: output}
    trace = Column(Text, default="[]")                     # JSON list of trace events
    stated = Column(Text, default="{}")                    # user-stated figures from request
    final_result = Column(Text, default="")                # JSON final aggregated result
    error = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class PolicyMetric(Base):
    __tablename__ = "policy_metrics"

    id = Column(Integer, primary_key=True)
    episode = Column(Integer, default=0)
    avg_reward = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
