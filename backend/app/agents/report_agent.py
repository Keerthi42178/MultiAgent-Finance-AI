"""Financial Report Agent: monthly reports, comparisons, exports."""
from datetime import date

from app.agents.base import BaseAgent
from app.services.finance import monthly_summary, trend, budget_status


class FinancialReportAgent(BaseAgent):
    name = "report"
    description = "Generates monthly financial reports, income/expense comparisons, category analysis and exportable summaries."

    def run(self, user, intent, state, **kw):
        action = intent.get("action", "report")
        task = kw.get("task")
        if task:
            return self._task_aggregate(user, state, task)
        month = intent.get("month")
        if month and "-" in month:
            y, m = int(month[:4]), int(month[5:7])
        else:
            today = date.today()
            y, m = today.year, today.month

        s = monthly_summary(self.db, user.id, y, m)
        bs = budget_status(self.db, user.id, y, m)
        tr = trend(self.db, user.id, 6)

        report = {
            "period": f"{y}-{m:02d}",
            "income": s["total_income"],
            "expenses": s["total_expenses"],
            "net": s["net"],
            "savings_rate": s["savings_rate"],
            "transaction_count": s["transaction_count"],
            "by_category": s["by_category"],
            "prev_month": s["prev_month"],
            "trend": tr,
            "budget": {"total_limit": bs["total_limit"], "total_spent": bs["total_spent"],
                       "overall_percent": bs["overall_percent"]} if bs else None,
        }

    def _task_aggregate(self, user, state, task):
        """Collaborative step: consume ALL prior agent outputs, validate
        consistency, identify gaps, and produce the single final response."""
        prior = task.get("prior_results") or {}
        stated = task.get("stated") or {}
        goal = task.get("primary_goal", "")
        contributions = []
        issues = []

        expense = prior.get("expense") or {}
        bill = prior.get("bill") or {}
        budget = prior.get("budget") or {}
        savings = prior.get("savings") or {}

        # --- consistency checks ---
        if expense and bill:
            exp_surplus = expense.get("available_before_emi")
            bill_remaining = bill.get("remaining_after_expenses_and_emi")
            emi = bill.get("existing_emi") or 0
            expected = (exp_surplus or 0) - emi - (bill.get("other_upcoming_bills") or 0)
            if bill_remaining is not None and abs(expected - bill_remaining) > 1:
                issues.append("Bill Agent remaining figure inconsistent with Expense output")
        if bill and budget:
            if budget.get("discretionary_income") is not None and \
               budget["discretionary_income"] > (bill.get("remaining_after_expenses_and_emi") or 0) + 1:
                issues.append("Budget discretionary income exceeds post-commitment capacity")

        # --- missing inputs ---
        if not (stated.get("income") or (state.get("summary") or {}).get("total_income")):
            issues.append("No income figure available — affordability estimates are indicative only")
        if not expense.get("monthly_expenses") and not stated.get("monthly_expenses"):
            issues.append("Expense breakdown unavailable — using income minus stated spend only")

        # --- contributions ---
        for name, out in (("expense", expense), ("bill", bill),
                          ("budget", budget), ("savings", savings)):
            if out:
                contributions.append({"agent": name,
                                      "summary": out.get("answer", "")[:160],
                                      "findings": out.get("findings", [])})

        # --- final structured result ---
        final = {
            "financial_summary": {
                "monthly_income": expense.get("monthly_income"),
                "monthly_expenses": expense.get("monthly_expenses"),
                "existing_emi": bill.get("existing_emi"),
                "available_surplus": bill.get("remaining_after_expenses_and_emi"),
            },
            "recommended_emi_range": (
                [round((budget.get("affordable_emi") or 0) * 0.7, 2),
                 round(budget.get("affordable_emi") or 0, 2)]
                if budget.get("affordable_emi") else None),
            "down_payment": savings.get("suggested_down_payment"),
            "loan_amount": savings.get("loan_amount"),
            "loan_scenarios": savings.get("loan_scenarios") or budget.get("loan_scenarios_per_lakh"),
            "monthly_savings_requirement": savings.get("monthly_savings_needed"),
            "purchase_timeline_months": savings.get("timeline_months"),
            "verdict": savings.get("verdict"),
            "risks": (savings.get("goal_risks") or []) + issues,
            "agent_contributions": contributions,
            "assumptions": budget.get("assumptions") or {},
            "data_quality": "verified_records" if expense.get("income_source") == "records" else "user_stated",
        }

        # --- final answer text ---
        parts = []
        fs = final["financial_summary"]
        if fs["monthly_income"] is not None:
            parts.append(f"Income ₹{fs['monthly_income']:,.0f}/mo, expenses "
                         f"₹{fs['monthly_expenses']:,.0f}/mo")
        if fs["existing_emi"]:
            parts.append(f"existing EMI ₹{fs['existing_emi']:,.0f}")
        if fs["available_surplus"] is not None:
            parts.append(f"surplus after commitments ₹{fs['available_surplus']:,.0f}/mo")
        if final["down_payment"]:
            parts.append(f"suggested down payment ₹{final['down_payment']:,.0f}, "
                         f"loan ₹{final['loan_amount']:,.0f}")
        if final["verdict"]:
            parts.append(f"verdict: {final['verdict'].replace('_', ' ')}")
        if final["purchase_timeline_months"]:
            parts.append(f"~{final['purchase_timeline_months']} months to save the down payment")
        for r in (final["risks"] or [])[:3]:
            parts.append(f"Risk: {r}")
        # how each agent contributed
        contrib_txt = "; ".join(f"{c['agent'].capitalize()} Agent: {c['summary']}"
                                for c in contributions[:5])
        answer = ". ".join(p.rstrip(".") for p in parts if p) + "."
        if contrib_txt:
            answer += f"\n\nHow the team contributed — {contrib_txt}."
        if issues:
            answer += "\n\nData notes: " + "; ".join(issues) + "."

        return {"ok": True, "kind": "task_result", "agent": "report",
                "final": final, "issues": issues,
                "consumed_from": [a for a in ("expense", "bill", "budget", "savings") if prior.get(a)],
                "answer": answer or "Task analysis complete."}

        if action == "export_report":
            return {"ok": True, "kind": "report_export", "report": report,
                    "answer": "Report generated. Use the Reports page to export PDF/CSV."}

        ans = (f"Financial report for {y}-{m:02d}: income ₹{s['total_income']:,.0f}, "
               f"expenses ₹{s['total_expenses']:,.0f}, net ₹{s['net']:,.0f} "
               f"(savings rate {s['savings_rate']}%). ")
        if s["top_category"]:
            ans += f"Top category: {s['top_category']['category']}. "
        if s["prev_month"]["expense_delta"] != 0:
            d = s["prev_month"]["expense_delta"]
            ans += f"Spending is {'up' if d > 0 else 'down'} ₹{abs(d):,.0f} vs last month."
        return {"ok": True, "kind": "report", "report": report, "answer": ans}
