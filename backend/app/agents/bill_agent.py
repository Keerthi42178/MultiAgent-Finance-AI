"""Bill Reminder Agent: bills, due dates, overdue detection, reminders."""
from datetime import date, timedelta

from app.agents.base import BaseAgent
from app.models.db_models import Bill
from app.services.finance import upcoming_bills
from app.services.notifications import notify


class BillReminderAgent(BaseAgent):
    name = "bill"
    description = "Manages bills and subscriptions, tracks due dates, flags overdue items, schedules reminders."

    def run(self, user, intent, state, **kw):
        action = intent.get("action", "bill_query")
        task = kw.get("task")
        if task:
            return self._task_analysis(user, state, task)
        today = date.today()

        if action == "add_bill":
            return self._add_bill(user, intent, today)
        if action == "mark_bill_paid":
            bid = intent.get("bill_id") or intent.get("amount")
            return self._mark_paid(user, intent)

        days = 30
        if action == "bill_reminder":
            days = 60
        bills = upcoming_bills(self.db, user.id, days=days)
        overdue = [b for b in bills if b["status"] == "overdue"]
        due_soon = [b for b in bills if b["status"] != "overdue" and
                    (date.fromisoformat(b["due_date"]) - today).days <= 7]

        # create reminder notifications for due-soon bills when asked
        if action == "bill_reminder":
            for b in due_soon[:5]:
                notify(self.db, user.id, "bill",
                       f"Reminder: {b['name']} ₹{b['amount']:,.0f} due {b['due_date']}")
            self.db.commit()

        total_upcoming = sum(b["amount"] for b in bills)
        ans = (f"You have {len(bills)} unpaid bills totaling ₹{total_upcoming:,.0f} "
               f"in the next {days} days.")
        if overdue:
            ans += f" {len(overdue)} overdue: " + ", ".join(b["name"] for b in overdue[:4]) + "."
        if due_soon:
            ans += " Due within a week: " + ", ".join(
                f"{b['name']} on {b['due_date']}" for b in due_soon[:4]) + "."
        return {"ok": True, "kind": "bills", "bills": bills,
                "overdue": overdue, "due_soon": due_soon, "answer": ans}

    def _task_analysis(self, user, state, task):
        """Collaborative step: consume Expense Agent output, layer EMI and
        upcoming bill obligations on top, compute remaining capacity."""
        prior = task.get("prior_results") or {}
        expense_out = prior.get("expense") or {}
        stated = task.get("stated") or {}
        today = date.today()

        available = expense_out.get("available_before_emi")
        if available is None:
            # dependency missing — derive from state so the task can continue
            summary = state.get("summary") or {}
            income = stated.get("income") or summary.get("total_income") or 0
            expenses = stated.get("monthly_expenses") or summary.get("total_expenses") or 0
            available = income - expenses

        # EMI: user-stated figure wins; otherwise sum bills that look like EMIs
        emi = stated.get("emi")
        emi_source = "user_stated" if emi else None
        bills = upcoming_bills(self.db, user.id, days=60)
        if emi is None:
            emi_bills = [b for b in bills
                         if any(k in b["name"].lower() for k in ("emi", "loan", "installment"))]
            emi = sum(b["amount"] for b in emi_bills) or 0
            emi_source = "records" if emi_bills else "none"
        emi = float(emi or 0)

        other_bills = sum(b["amount"] for b in bills
                          if not any(k in b["name"].lower()
                                     for k in ("emi", "loan", "installment")))
        remaining = available - emi - other_bills
        findings = []
        if emi:
            findings.append(f"Existing EMI/loan commitments: ₹{emi:,.0f} ({emi_source})")
        if other_bills:
            findings.append(f"Other upcoming bills: ₹{other_bills:,.0f}")
        if remaining < 0:
            findings.append("Commitments exceed available surplus — debt pressure")
        return {
            "ok": True, "kind": "task_step", "agent": "bill",
            "existing_emi": round(emi, 2),
            "emi_source": emi_source,
            "other_upcoming_bills": round(other_bills, 2),
            "remaining_after_expenses_and_emi": round(remaining, 2),
            "consumed_from": "expense" if expense_out.get("available_before_emi") is not None else "state",
            "findings": findings,
            "answer": (f"After expenses (surplus ₹{available:,.0f}) and EMI ₹{emi:,.0f}, "
                       f"remaining capacity is ₹{remaining:,.0f}/month."),
        }

    def _add_bill(self, user, intent, today):
        name = intent.get("bill_name") or intent.get("description") or "Bill"
        amount = intent.get("amount")
        if not amount or amount <= 0:
            return {"ok": False, "needs_input": "amount",
                    "answer": f"What's the amount for {name}?"}
        due = intent.get("due_date")
        try:
            due_d = date.fromisoformat(due) if due else today + timedelta(days=7)
        except ValueError:
            due_d = today + timedelta(days=7)
        freq = intent.get("frequency") or "monthly"
        bill = Bill(user_id=user.id, name=name[:160], amount=float(amount),
                    due_date=due_d, frequency=freq if freq in ("monthly", "yearly", "one_time") else "monthly",
                    status="upcoming")
        self.db.add(bill)
        self.db.commit()
        notify(self.db, user.id, "bill", f"Bill added: {bill.name}",
               f"₹{bill.amount:,.0f} due {bill.due_date.isoformat()}")
        self.db.commit()
        return {"ok": True, "kind": "action_result", "bill_id": bill.id,
                "answer": f"Added {bill.name}: ₹{bill.amount:,.0f} due {bill.due_date.isoformat()} ({bill.frequency})."}

    def _mark_paid(self, user, intent):
        q = self.db.query(Bill).filter_by(user_id=user.id)
        bid = intent.get("bill_id")
        if bid:
            bill = q.filter(Bill.id == int(bid)).first()
        else:
            # match by name fragment
            name = (intent.get("bill_name") or intent.get("description") or "").lower()
            bills = q.filter(Bill.status != "paid").all()
            bill = next((b for b in bills if name and name in b.name.lower()), None)
        if not bill:
            return {"ok": False, "needs_input": "bill",
                    "answer": "Which bill should I mark as paid?"}
        bill.status = "paid"
        self.db.commit()
        return {"ok": True, "kind": "action_result",
                "answer": f"Marked {bill.name} (₹{bill.amount:,.0f}) as paid."}
