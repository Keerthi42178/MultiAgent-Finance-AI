from app.agents.base import BaseAgent
from app.agents.expense_agent import ExpenseAnalysisAgent
from app.agents.budget_agent import BudgetPlanningAgent
from app.agents.bill_agent import BillReminderAgent
from app.agents.savings_agent import SavingsRecommendationAgent
from app.agents.report_agent import FinancialReportAgent

AGENTS = {
    "expense": ExpenseAnalysisAgent,
    "budget": BudgetPlanningAgent,
    "bill": BillReminderAgent,
    "savings": SavingsRecommendationAgent,
    "report": FinancialReportAgent,
}

__all__ = ["BaseAgent", "AGENTS"] + list(AGENTS.values())
