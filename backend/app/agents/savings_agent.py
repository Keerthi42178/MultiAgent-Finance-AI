"""Savings Recommendation Agent: goals, surplus analysis, savings plans."""
from datetime import date

from app.agents.base import BaseAgent
from app.models.db_models import SavingsGoal
from app.services.finance import goals_progress, monthly_summary


class SavingsRecommendationAgent(BaseAgent):
    name = "savings"
    description = "Calculates savings targets, analyzes surplus, suggests realistic savings plans and tracks goal progress."

    def run(self, user, intent, state, **kw):
        action = intent.get("action", "goal_query")
        task = kw.get("task")
        if task:
            return self._task_analysis(user, state, task)
        today = date.today()

        if action == "add_goal":
            return self._add_goal(user, intent)
        if action == "affordability":
            return self._affordability(user, intent, state)

        if action == "goal_contribution":
            return self._contribute(user, intent)

        if action in ("savings_plan", "save_money"):
            target = intent.get("target_amount")
            months = intent.get("target_months")
            name = intent.get("goal_name") or "Savings goal"
            if target and months:
                per_month = round(target / months, 2)
                summ = monthly_summary(self.db, user.id, today.year, today.month)
                surplus = summ["net"]
                feasible = surplus >= per_month if surplus > 0 else True
                ans = (f"To save ₹{target:,.0f} in {months} months you need ₹{per_month:,.0f}/month"
                       f" (assuming no returns). ")
                if summ["total_income"] > 0:
                    ans += f"Your recorded net this month is ₹{surplus:,.0f}. "
                    ans += "This is feasible." if feasible else "That exceeds your current surplus — consider a longer timeline or higher income."
                return {"ok": True, "kind": "savings_plan",
                        "plan": {"name": name, "target": target, "months": months,
                                 "monthly_needed": per_month, "current_surplus": surplus},
                        "needs_confirmation": True,
                        "answer": ans + " Want me to create this savings goal?"}
            return {"ok": False, "needs_input": "target_amount",
                    "answer": "Sure! How much do you want to save, and in how many months?"}

        goals = goals_progress(self.db, user.id)
        summ = monthly_summary(self.db, user.id, today.year, today.month)
        total_saved = sum(g["saved"] for g in goals)
        total_target = sum(g["target"] for g in goals)
        ans = f"You have {len(goals)} savings goals. Total saved: ₹{total_saved:,.0f}."
        if goals:
            active = [g for g in goals if g["percent"] < 100]
            if active:
                g = max(active, key=lambda x: x["remaining"])
                ans += (f" Largest remaining: {g['name']} needs ₹{g['remaining']:,.0f}"
                        + (f" (~₹{g['required_monthly']:,.0f}/month)" if g["required_monthly"] else "") + ".")
        if summ["net"] > 0:
            ans += f" This month's recorded surplus is ₹{summ['net']:,.0f}, which can be allocated to goals."
        return {"ok": True, "kind": "goals", "goals": goals, "summary": summ, "answer": ans}

    def _task_analysis(self, user, state, task):
        """Collaborative step: consume Budget Agent output, evaluate the
        purchase: down payment, loan, EMI fit, savings timeline, goal risks."""
        from math import ceil
        from app.services.finance import (LOAN_ASSUMPTIONS, goals_progress,
                                          loan_scenarios)
        prior = task.get("prior_results") or {}
        budget_out = prior.get("budget") or {}
        bill_out = prior.get("bill") or {}
        expense_out = prior.get("expense") or {}
        stated = task.get("stated") or {}
        goal = task.get("primary_goal", "")

        price = stated.get("price")
        income = expense_out.get("monthly_income") or 0
        # savings capacity = what remains after the recommended new EMI
        affordable_emi = budget_out.get("affordable_emi") or 0
        remaining = bill_out.get("remaining_after_expenses_and_emi") or 0
        monthly_capacity = max(0.0, min(remaining - affordable_emi, remaining))
        # for savings goals/trips the full remaining surplus is the capacity
        if goal in ("savings_plan", "goal_prioritization", "trip_savings"):
            monthly_capacity = max(0.0, remaining)

        findings = []
        risks = []

        out = {"ok": True, "kind": "task_step", "agent": "savings",
               "monthly_savings_capacity": round(monthly_capacity, 2),
               "consumed_from": [a for a in ("expense", "bill", "budget") if prior.get(a)],
               "findings": findings, "risks": risks}

        if goal == "purchase_affordability" and price:
            dp_ratio = LOAN_ASSUMPTIONS["down_payment_ratio"]
            down = round(price * dp_ratio, 2)
            # try larger down payment if capacity allows faster payoff
            loan = price - down
            scenarios = loan_scenarios(loan)
            best = min(scenarios, key=lambda s: s["emi"])
            emi_fit = best["emi"] <= affordable_emi if affordable_emi else None
            months_needed = ceil(down / monthly_capacity) if monthly_capacity > 0 else None
            verdict = ("affordable" if emi_fit and (months_needed or 0) <= 24
                       else "stretch" if emi_fit else "not_affordable_now")
            if not emi_fit:
                risks.append("Recommended EMI exceeds safe monthly capacity")
            if months_needed and months_needed > 24:
                risks.append(f"Down payment needs ~{months_needed} months to save")
            goals = goals_progress(self.db, user.id)
            active_goals = [g for g in goals if g.get("percent", 0) < 100]
            if active_goals:
                risks.append(f"{len(active_goals)} active savings goal(s) may be delayed")
            findings.append(f"Suggested down payment ₹{down:,.0f} ({dp_ratio:.0%} of price)")
            findings.append(f"Loan ₹{loan:,.0f}; lowest EMI ₹{best['emi']:,.0f}/mo "
                            f"over {best['tenure_months']} months")
            out.update({
                "price": price, "suggested_down_payment": down,
                "loan_amount": round(loan, 2), "loan_scenarios": scenarios,
                "recommended_emi": best["emi"], "emi_fit": emi_fit,
                "timeline_months": months_needed, "verdict": verdict,
                "goal_risks": risks,
            })
            out["answer"] = (f"Down payment ~₹{down:,.0f}; loan ₹{loan:,.0f}; "
                             f"EMI ₹{best['emi']:,.0f}/mo for {best['tenure_months']} months. "
                             f"Verdict: {verdict.replace('_', ' ')}.")
        elif goal in ("savings_plan", "goal_prioritization", "trip_savings"):
            target = stated.get("price") or stated.get("target")
            months = stated.get("months")
            if target and months and monthly_capacity > 0:
                need = round(target / months, 2)
                feasible = monthly_capacity >= need
                if not feasible:
                    risks.append("Monthly target exceeds current surplus capacity")
                goals = goals_progress(self.db, user.id)
                if goals:
                    findings.append(f"{len(goals)} existing goal(s) to balance against")
                out.update({"target_amount": target, "timeline_months": months,
                            "monthly_savings_needed": need, "feasible": feasible,
                            "goal_risks": risks})
                out["answer"] = (f"Saving ₹{target:,.0f} in {months} months needs "
                                 f"₹{need:,.0f}/month; capacity ₹{monthly_capacity:,.0f} "
                                 f"-> {'feasible' if feasible else 'not feasible at current surplus'}.")
            else:
                goals = goals_progress(self.db, user.id)
                out["answer"] = (f"Monthly savings capacity ₹{monthly_capacity:,.0f}; "
                                 f"{len(goals)} existing goal(s) tracked.")
        elif goal == "debt_management":
            emi = bill_out.get("existing_emi") or stated.get("emi") or 0
            surplus_after = max(0.0, (bill_out.get("remaining_after_expenses_and_emi") or 0))
            paydown = min(surplus_after, emi * 0.5) if emi else surplus_after
            findings.append(f"Existing EMI ₹{emi:,.0f}; surplus after commitments ₹{surplus_after:,.0f}")
            if paydown > 0:
                findings.append(f"Suggested extra debt paydown ₹{paydown:,.0f}/month")
            out.update({"existing_emi": emi, "debt_paydown_capacity": round(paydown, 2),
                        "goal_risks": risks})
            out["answer"] = (f"Existing EMI ₹{emi:,.0f}; after commitments ₹{surplus_after:,.0f} remains. "
                             f"Safe extra paydown ~₹{paydown:,.0f}/month.")
        elif goal == "budget_replan":
            out["answer"] = (f"Post-change capacity ₹{monthly_capacity:,.0f}/month "
                             f"available for EMI/savings after obligations.")
        return out

    def _add_goal(self, user, intent):
        name = intent.get("goal_name") or intent.get("description") or "Savings goal"
        target = intent.get("target_amount")
        if not target or target <= 0:
            return {"ok": False, "needs_input": "target_amount",
                    "answer": "What's the target amount for this goal?"}
        saved = intent.get("saved_amount") or 0
        months = intent.get("target_months")
        target_date = None
        contrib = 0.0
        if months:
            from datetime import timedelta
            target_date = date.today() + timedelta(days=30 * int(months))
            remaining = max(0.0, target - saved)
            contrib = round(remaining / int(months), 2)
        goal = SavingsGoal(user_id=user.id, name=name[:160], target_amount=float(target),
                           saved_amount=float(saved), target_date=target_date,
                           monthly_contribution=contrib)
        self.db.add(goal)
        self.db.commit()
        prog = goals_progress(self.db, user.id)
        g = next((x for x in prog if x["id"] == goal.id), {})
        return {"ok": True, "kind": "action_result", "goal": g,
                "answer": f"Created goal '{goal.name}': target ₹{goal.target_amount:,.0f}"
                          + (f", contributing ₹{contrib:,.0f}/month." if contrib else ".")}

    def _affordability(self, user, intent, state):
        price = intent.get("amount") or 0
        if price <= 0:
            return {"ok": False, "needs_input": "amount",
                    "answer": "What's the price of the item you're considering?"}
        summ = state.get("summary", {}) or {}
        income = summ.get("total_income") or state.get("profile", {}).get("monthly_income") or 0
        surplus = summ.get("net", 0)
        bills = state.get("bills_upcoming", []) or []
        bills_total = sum(b["amount"] for b in bills if b.get("status") != "overdue")
        if not income and not surplus:
            return {"ok": True, "kind": "affordability",
                    "answer": f"I don't have your income or expense records yet, so I can't judge affordability for ₹{price:,.0f}. "
                              "Complete onboarding or add your income and expenses first — then ask me again."}
        # months to afford from surplus
        months = int(price / surplus) + 1 if surplus > 0 else None
        ans = (f"Item cost: ₹{price:,.0f}. Recorded monthly income ₹{income:,.0f}, "
               f"net surplus ₹{surplus:,.0f}, upcoming bills ₹{bills_total:,.0f}. ")
        if surplus > 0 and price <= surplus:
            ans += "You could cover it from this month's surplus."
        elif surplus > 0:
            ans += f"At your current surplus you'd need ~{months} months of saving."
        else:
            ans += "Your recorded surplus is currently negative — consider deferring or cutting discretionary spending."
        return {"ok": True, "kind": "affordability",
                "affordability": {"price": price, "income": income, "surplus": surplus,
                                  "bills_total": bills_total, "months_needed": months},
                "answer": ans}

    def _contribute(self, user, intent):
        amount = intent.get("amount")
        if not amount or amount <= 0:
            return {"ok": False, "needs_input": "amount",
                    "answer": "How much would you like to add to savings?"}
        goals = self.db.query(SavingsGoal).filter_by(user_id=user.id).all()
        if not goals:
            return {"ok": False, "answer": "You don't have any savings goals yet."}
        name = (intent.get("goal_name") or "").lower()
        goal = next((g for g in goals if name and name in g.name.lower()), goals[0])
        goal.saved_amount = (goal.saved_amount or 0) + float(amount)
        self.db.commit()
        prog = goals_progress(self.db, user.id)
        g = next((x for x in prog if x["id"] == goal.id), {})
        return {"ok": True, "kind": "action_result", "goal": g,
                "answer": f"Added ₹{amount:,.0f} to '{goal.name}'. Now {g.get('percent', 0)}% complete."}
