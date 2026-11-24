"""Budget Planning Agent: suggest, create and evaluate budgets."""
from datetime import date

from app.agents.base import BaseAgent
from app.models.db_models import Budget, BudgetCategory, UserProfile
from app.services.finance import budget_status, suggest_budget


class BudgetPlanningAgent(BaseAgent):
    name = "budget"
    description = "Creates personalized budgets, recalculates allocations and compares planned vs actual spending."

    def run(self, user, intent, state, **kw):
        action = intent.get("action", "budget_query")
        task = kw.get("task")
        if task:
            return self._task_analysis(user, state, task)
        today = date.today()

        if action in ("create_budget", "update_budget"):
            return self._create(user, intent, today)

        bs = budget_status(self.db, user.id, today.year, today.month)
        if action == "budget_suggest":
            income = intent.get("amount") or state.get("profile", {}).get("monthly_income") or 0
            sugg = suggest_budget(self.db, user.id, income)
            return {"ok": True, "kind": "budget_suggestion", "suggestion": sugg,
                    "answer": self._suggest_answer(sugg)}

        if not bs:
            sugg = suggest_budget(self.db, user.id, 0)
            return {"ok": True, "kind": "budget_suggestion", "suggestion": sugg,
                    "answer": "You don't have a budget for this month yet. Here's a suggested allocation based on your profile — want me to create it?",
                    "needs_confirmation": True}

        over = [c for c in bs["categories"] if c["over"]]
        near = [c for c in bs["categories"] if 80 <= c["percent"] < 100]
        ans = (f"Budget for {today.year}-{today.month:02d}: ₹{bs['total_spent']:,.0f} spent of "
               f"₹{bs['total_limit']:,.0f} ({bs['overall_percent']}%).")
        if over:
            ans += " Over budget: " + ", ".join(c["category"] for c in over) + "."
        if near:
            ans += " Nearing limit: " + ", ".join(c["category"] for c in near) + "."
        return {"ok": True, "kind": "budget", "budget": bs,
                "over_budget": over, "near_limit": near, "answer": ans}

    def _task_analysis(self, user, state, task):
        """Collaborative step: consume Expense + Bill outputs, compute
        discretionary income and affordable additional-EMI capacity."""
        from app.services.finance import LOAN_ASSUMPTIONS, loan_scenarios
        prior = task.get("prior_results") or {}
        expense_out = prior.get("expense") or {}
        bill_out = prior.get("bill") or {}

        income = expense_out.get("monthly_income") or 0
        remaining = bill_out.get("remaining_after_expenses_and_emi")
        if remaining is None:
            summary = state.get("summary") or {}
            stated = task.get("stated") or {}
            income = stated.get("income") or summary.get("total_income") or income
            expenses = stated.get("monthly_expenses") or summary.get("total_expenses") or 0
            emi = stated.get("emi") or 0
            remaining = income - expenses - emi
        remaining = max(0.0, float(remaining or 0))

        # safety buffer: keep X% of income aside; rest is discretionary
        buffer_amt = income * LOAN_ASSUMPTIONS["safety_buffer_ratio"] if income else 0
        discretionary = max(0.0, remaining - buffer_amt)
        affordable_emi = discretionary * LOAN_ASSUMPTIONS["emi_capacity_ratio"]

        # generic loan scenarios on a reference principal (per ₹1 lakh)
        scenarios = loan_scenarios(100000)
        findings = []
        if bill_out.get("existing_emi"):
            findings.append(f"Existing EMI ₹{bill_out['existing_emi']:,.0f} already committed")
        if buffer_amt:
            findings.append(f"Safety buffer ₹{buffer_amt:,.0f} preserved")
        if affordable_emi <= 0:
            findings.append("No capacity for additional EMI after commitments and buffer")
        return {
            "ok": True, "kind": "task_step", "agent": "budget",
            "discretionary_income": round(discretionary, 2),
            "safety_buffer": round(buffer_amt, 2),
            "affordable_emi": round(affordable_emi, 2),
            "loan_scenarios_per_lakh": scenarios,
            "assumptions": LOAN_ASSUMPTIONS,
            "consumed_from": [a for a in ("expense", "bill") if prior.get(a)],
            "findings": findings,
            "answer": (f"After buffer, discretionary ₹{discretionary:,.0f}/month; "
                       f"safe additional EMI capacity ~₹{affordable_emi:,.0f}."),
        }

    def _suggest_answer(self, sugg):
        if not sugg.get("suggestions"):
            return "I need your monthly income to build a budget. What's your monthly income?"
        top = sugg["suggestions"][:3]
        return (f"Based on ₹{sugg['total_income']:,.0f}/month income, I suggest: needs ~₹{sugg['needs_total']:,.0f}, "
                f"wants ~₹{sugg['wants_total']:,.0f}, savings ~₹{sugg['savings_suggestion']:,.0f}. "
                f"Top allocations: " + ", ".join(f"{s['category']} ₹{s['amount']:,.0f}" for s in top) +
                ". Want me to create this budget? You can edit every amount.")

    def _create(self, user, intent, today):
        cats = intent.get("categories")
        income = intent.get("amount") or 0
        if not cats:
            sugg = suggest_budget(self.db, user.id, income)
            cats = [{"category": s["category"], "limit_amount": s["amount"], "essential": s["essential"]}
                    for s in sugg.get("suggestions", [])]
            income = income or sugg.get("total_income", 0)
        if not cats:
            return {"ok": False, "needs_input": "categories",
                    "answer": "I need category limits to create a budget. For example: Food ₹4000, Shopping ₹3000."}

        # upsert budget for current month
        budget = self.db.query(Budget).filter_by(user_id=user.id, year=today.year, month=today.month).first()
        if not budget:
            budget = Budget(user_id=user.id, year=today.year, month=today.month)
            self.db.add(budget)
            self.db.flush()
        budget.total_income = income or budget.total_income
        budget.total_budget = round(sum(c.get("limit_amount", 0) for c in cats), 2)
        budget.savings_target = intent.get("savings_target", budget.savings_target or 0)
        self.db.query(BudgetCategory).filter_by(budget_id=budget.id).delete()
        for c in cats:
            self.db.add(BudgetCategory(
                budget_id=budget.id, category=c.get("category", "Other"),
                limit_amount=float(c.get("limit_amount", 0)),
                essential=bool(c.get("essential", False))))
        self.db.commit()
        bs = budget_status(self.db, user.id, today.year, today.month)
        return {"ok": True, "kind": "action_result", "budget": bs,
                "answer": f"Budget created for {today.year}-{today.month:02d} with "
                          f"{len(cats)} categories totaling ₹{budget.total_budget:,.0f}."}
