"""Expense Analysis Agent: spending totals, trends, category breakdown, anomalies."""
from datetime import date

from sqlalchemy import func

from app.agents.base import BaseAgent
from app.models.db_models import Expense
from app.services.finance import (
    categorize, daily_series, expenses_in_range, monthly_summary, trend,
    weekly_series,
)


class ExpenseAnalysisAgent(BaseAgent):
    name = "expense"
    description = "Analyzes transactions, spending patterns, month-over-month changes, anomalies and budget overruns."

    def run(self, user, intent, state, **kw):
        action = intent.get("action", "spending_query")
        task = kw.get("task")
        if task:
            return self._task_analysis(user, state, task)
        today = date.today()

        if action in ("add_expense", "add_expense_confirmed"):
            return self._add_expense(user, intent)

        # Determine range
        month = intent.get("month")
        if month and "-" in month:
            y, m = int(month[:4]), int(month[5:7])
        else:
            y, m = today.year, today.month
        start = date(y, m, 1)
        end = date(y, m, 28)  # safe; summary uses real bounds anyway
        summary = monthly_summary(self.db, user.id, y, m)

        if action == "why_spending":
            return self._why(summary, user.id, y, m)
        if action == "reduce_spending":
            return self._reduce(summary, state)
        if action == "spending_detail":
            cat = intent.get("category")
            txns = expenses_in_range(self.db, user.id, *self._bounds(y, m), category=cat)[:20]
            return {"ok": True, "kind": "transactions", "transactions": [self._tx(t) for t in txns],
                    "summary": summary, "answer": f"Found {len(txns)} recorded transactions."}
        # default: spending query
        daily = daily_series(self.db, user.id, start, end)
        weekly = self._weekly(user.id, y, m)
        return {
            "ok": True, "kind": "spending",
            "summary": summary,
            "daily": daily, "weekly": weekly,
            "trend": trend(self.db, user.id, 6),
            "answer": self._spending_answer(summary),
        }

    def _task_analysis(self, user, state, task):
        """Collaborative step: compute income, expenses and pre-EMI surplus.

        User-stated figures (from the request) take precedence and are labeled;
        otherwise recorded DB data is used. Output feeds Bill/Budget/Savings.
        """
        stated = task.get("stated") or {}
        summary = state.get("summary") or {}
        income = stated.get("income") or summary.get("total_income") or 0
        expenses = stated.get("monthly_expenses") or summary.get("total_expenses") or 0
        available = income - expenses
        findings = []
        top = summary.get("top_category")
        if top:
            findings.append(f"Highest recorded category: {top['category']} "
                            f"₹{top['amount']:,.0f}")
        prev = summary.get("prev_month") or {}
        delta = prev.get("expense_delta", 0)
        if delta:
            findings.append(f"Spending {'up' if delta > 0 else 'down'} "
                            f"₹{abs(delta):,.0f} vs last month")
        if not income:
            findings.append("No income figure available (neither stated nor recorded)")
        return {
            "ok": True, "kind": "task_step", "agent": "expense",
            "monthly_income": round(income, 2),
            "monthly_expenses": round(expenses, 2),
            "available_before_emi": round(available, 2),
            "income_source": "user_stated" if stated.get("income") else "records",
            "expenses_source": "user_stated" if stated.get("monthly_expenses") else "records",
            "findings": findings,
            "summary": summary,
            "answer": (f"Monthly income ₹{income:,.0f}, expenses ₹{expenses:,.0f} "
                       f"-> surplus before EMI ₹{available:,.0f}."),
        }

    def _bounds(self, y, m):
        from calendar import monthrange
        return date(y, m, 1), date(y, m, monthrange(y, m)[1])

    def _weekly(self, user_id, y, m):
        s, e = self._bounds(y, m)
        return weekly_series(self.db, user_id, s, e)

    def _tx(self, t):
        return {"id": t.id, "date": t.date.isoformat(), "amount": t.amount,
                "category": t.category, "description": t.description,
                "payment_method": t.payment_method, "source": t.source}

    def _spending_answer(self, s):
        top = s["top_category"]
        prev = s["prev_month"]
        delta = prev["expense_delta"]
        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        ans = (f"Recorded expenses for {s['year']}-{s['month']:02d} are "
               f"₹{s['total_expenses']:,.0f} across {s['transaction_count']} transactions. "
               f"Income recorded: ₹{s['total_income']:,.0f}.")
        if top:
            ans += f" Highest category: {top['category']} at ₹{top['amount']:,.0f}."
        ans += f" Versus last month, spending is {direction} by ₹{abs(delta):,.0f}."
        if s["transaction_count"] == 0:
            ans += " No transactions recorded yet — add expenses manually or import a bank statement CSV from the Expenses page."
        return ans

    def _why(self, summary, user_id, y, m):
        s, e = self._bounds(y, m)
        prev_s = self._prev_bounds(y, m)
        cur = {r[0]: float(r[1] or 0) for r in
               self.db.query(Expense.category, func.sum(Expense.amount))
               .filter(Expense.user_id == user_id, Expense.date >= s, Expense.date <= e)
               .group_by(Expense.category)}
        prev = {r[0]: float(r[1] or 0) for r in
                self.db.query(Expense.category, func.sum(Expense.amount))
                .filter(Expense.user_id == user_id, Expense.date >= prev_s[0], Expense.date <= prev_s[1])
                .group_by(Expense.category)}
        diffs = sorted(((c, cur.get(c, 0) - prev.get(c, 0), cur.get(c, 0))
                        for c in set(cur) | set(prev)), key=lambda x: -x[1])
        risers = [d for d in diffs if d[1] > 500][:3]
        ans = "Comparing with the previous month: "
        if risers:
            ans += ", ".join(f"{c} rose by ₹{d:,.0f}" for c, d, _ in risers) + ". "
        else:
            ans += "no category increased sharply. "
        # top merchants in top riser category
        if risers:
            cat = risers[0][0]
            txns = (self.db.query(Expense).filter(Expense.user_id == user_id,
                                                  Expense.category == cat,
                                                  Expense.date >= s, Expense.date <= e)
                    .order_by(Expense.amount.desc()).limit(5).all())
            if txns:
                ans += "Largest recorded items: " + "; ".join(
                    f"{t.description[:30]} ₹{t.amount:,.0f}" for t in txns) + "."
        return {"ok": True, "kind": "analysis", "summary": summary,
                "category_deltas": [{"category": c, "delta": round(d, 2), "current": round(v, 2)}
                                     for c, d, v in diffs[:8]],
                "answer": ans}

    def _prev_bounds(self, y, m):
        from calendar import monthrange
        if m == 1:
            return date(y - 1, 12, 1), date(y - 1, 12, 31)
        return date(y, m - 1, 1), date(y, m - 1, monthrange(y, m - 1)[1])

    def _reduce(self, summary, state):
        budget = state.get("budget") or {}
        over = [c for c in budget.get("categories", []) if c.get("over")]
        ans = []
        if over:
            ans.append("These categories are over budget: " +
                       ", ".join(f"{c['category']} (₹{c['spent']:,.0f} vs ₹{c['limit']:,.0f})"
                                 for c in over[:4]) + ".")
        cats = summary.get("by_category", [])[:5]
        if cats:
            ans.append("Top spending: " + ", ".join(
                f"{c['category']} ₹{c['amount']:,.0f}" for c in cats) + ".")
        ans.append("Consider reviewing recurring subscriptions and dining out first — they are usually the easiest to trim.")
        return {"ok": True, "kind": "recommendation", "summary": summary,
                "over_budget": over, "answer": " ".join(ans)}

    def _add_expense(self, user, intent):
        from app.models.db_models import Expense
        amount = intent.get("amount")
        if not amount or amount <= 0:
            return {"ok": False, "needs_input": "amount",
                    "answer": "Sure — how much was the expense?"}
        cat = intent.get("category") or categorize(intent.get("description") or "")
        if cat not in ["Food & Groceries", "Transportation", "Shopping", "Housing & Rent",
                       "Utilities", "Healthcare", "Education", "Entertainment", "Travel",
                       "Personal Care", "Subscriptions", "Other"]:
            cat = categorize(cat)
        exp = Expense(user_id=user.id, amount=float(amount), date=date.today(),
                      category=cat, description=intent.get("description") or "Added via assistant",
                      payment_method=intent.get("payment_method") or "", source="manual")
        self.db.add(exp)
        self.db.commit()
        return {"ok": True, "kind": "action_result",
                "answer": f"Recorded an expense of ₹{amount:,.0f} under {cat}.",
                "expense_id": exp.id}
