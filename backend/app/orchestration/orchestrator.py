"""Dialogue orchestrator: intent -> policy -> agent -> blackboard -> response.

Coordinates the five specialized agents through shared blackboard memory.
The Actor-Critic policy selects the next action/agent; a rule-based baseline
acts as safety fallback. Safety rules (confirmation gates, read-only unless
confirmed) always override policy decisions.
"""
import json
import logging
import re
from datetime import date

from sqlalchemy.orm import Session

from app.agents import AGENTS
from app.integrations import gemini
from app.memory import blackboard
from app.models.db_models import ConversationMessage, ConversationSession, PolicyActionHistory, User
from app.reinforcement_learning.actor_critic import get_policy
from app.services.finance import categorize

logger = logging.getLogger("finpilot.orchestrator")

# intents that mutate data -> require confirmation first
WRITE_INTENTS = {"add_expense", "add_income", "add_bill", "add_goal", "update_income", "delete_expense"}


def heuristic_intent(message: str) -> dict:
    """Fast deterministic intent extraction (fallback when Gemini is absent)."""
    m = message.lower()
    amount = None
    am = re.search(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)", m)
    if am:
        try:
            amount = float(am.group(1).replace(",", ""))
        except ValueError:
            pass
    cat = categorize(message)
    if cat == "Other" and any(w in m for w in ["food", "grocery", "restaurant"]):
        cat = "Food & Groceries"

    if re.search(r"\b(save|saving|goal|emergency fund|laptop|phone|car)\b", m) and \
       re.search(r"\d", m) and re.search(r"(month|save ₹|₹)", m):
        return {"intent": "savings_plan", "amount": amount,
                "target_amount": amount, "raw": message}
    # savings-goal creation request; amount may be missing -> assistant asks
    if re.search(r"\b(save|saving|savings|goal)\b", m) and \
       re.search(r"\b(create|set|start|make|build|buy|purchase|need|want|plan)\b", m) and \
       not re.match(r"^(what|how|is|are|show|list|my|do i|did i)\b", m):
        gm = re.search(r"(?:buy|purchase|get)\s+(?:a|an|the)?\s*([a-zA-Z][a-zA-Z ]{1,30})", m) or \
             re.search(r"goal\s+(?:plan\s+)?for\s+(?:a|an|the)?\s*([a-zA-Z][a-zA-Z ]{1,30})", m)
        goal_name = None
        if gm:
            goal_name = re.split(r"\b(?:in|for|within|by|on)\b",
                                 gm.group(1))[0].strip().title()
        return {"intent": "add_goal", "amount": amount,
                "target_amount": amount, "goal_name": goal_name,
                "target_months": _months_hint(m), "raw": message}
    # savings-goal status queries -> savings agent
    if re.search(r"\b(goals?|saving|savings)\b", m) and \
       re.search(r"\b(my|status|progress|show|list|track|how much|much)\b", m) and \
       re.match(r"^(what|how|is|are|show|list|my|do i|did i|tell)\b", m):
        return {"intent": "goal_query", "raw": message}
    # trip / travel planning -> finance planning for the trip
    if re.search(r"\b(trip|travel|goa|vacation|holiday|tour|visit|plan)\b", m) and \
       re.search(r"\b(trip|travel|goa|vacation|holiday|tour)\b", m):
        return {"intent": "trip_plan", **extract_trip_slots(message),
                "raw": message}
    if re.search(r"\b(can i afford|afford|affordable)\b", m):
        return {"intent": "affordability", "amount": amount, "raw": message}
    if "remind" in m or "bill" in m or "emi" in m:
        return {"intent": "bill_query" if "remind" not in m else "bill_reminder",
                "amount": amount, "raw": message}
    if "budget" in m:
        return {"intent": "budget_query", "amount": amount, "raw": message}
    if any(w in m for w in ["report", "summary", "statement", "monthly report"]):
        return {"intent": "report", "amount": amount, "raw": message}
    if any(w in m for w in ["why", "reason", "increase", "more than"]):
        return {"intent": "why_spending", "amount": amount, "raw": message}
    if any(w in m for w in ["reduce", "cut", "decrease", "save money", "where can i"]):
        return {"intent": "reduce_spending", "amount": amount, "raw": message}
    if any(w in m for w in ["income", "salary", "earned", "received"]):
        if "increase" in m or "now" in m or "changed" in m:
            return {"intent": "update_income", "amount": amount, "raw": message}
        return {"intent": "income_query", "amount": amount, "raw": message}
    # add-expense must be checked BEFORE spending_query — the word "expense"
    # would otherwise swallow "add expense 500 for lunch"
    if re.search(r"\b(add|record|log|spent)\b.*\b(expense|spent|on)\b", m) or \
       (amount and any(w in m for w in ["spent", "expense", "paid", "bought"])):
        return {"intent": "add_expense", "amount": amount, "category": cat,
                "description": message, "needs_confirmation": True, "raw": message}
    if any(w in m for w in ["spend", "spent", "expense", "cost", "much", "how much", "where"]):
        return {"intent": "spending_query", "amount": amount, "category": cat if cat != "Other" else None,
                "raw": message}
    if "hello" in m or "hi" in m or "hey" in m:
        return {"intent": "smalltalk", "raw": message}
    return {"intent": "spending_query", "amount": amount, "category": cat if cat != "Other" else None,
            "raw": message}


async def understand(message: str, history: list[dict]) -> dict:
    """Gemini intent extraction with heuristic fallback."""
    intent = await gemini.classify_intent(message, history)
    if intent and intent.get("intent"):
        intent["raw"] = message
        intent.setdefault("amount", None)
        intent.setdefault("needs_confirmation", False)
        return intent
    return heuristic_intent(message)


def _missing_slots(intent: dict) -> list[str]:
    missing = []
    it = intent.get("intent")
    if it in ("add_expense",) and not intent.get("amount"):
        missing.append("amount")
    if it in ("add_goal", "savings_plan") and not intent.get("target_amount") and not intent.get("amount"):
        missing.append("target_amount")
    if it == "savings_plan" and not intent.get("target_months"):
        missing.append("target_months")
    if it == "add_bill" and not intent.get("amount"):
        missing.append("amount")
    return missing


async def handle_message(db: Session, user: User, session: ConversationSession,
                         message: str) -> dict:
    """Full pipeline for one user message. Returns assistant payload."""
    policy = get_policy()
    board_state = blackboard.get_state(db, user, refresh=True)
    dialogue = blackboard.get_dialogue(db, user)

    history = [{"role": m.role, "content": m.content} for m in session.messages[-10:]]

    # save user message
    db.add(ConversationMessage(session_id=session.id, role="user", content=message))
    db.commit()

    turn = len(session.messages)
    pending = dialogue.get("pending_confirmation")

    # Confirmation flow: user replies yes/no to a pending write action
    if pending:
        low = message.strip().lower()
        if re.match(r"^(yes|y|confirm|ok|okay|sure|go ahead|do it|proceed)\b", low):
            intent = pending["intent"]
            # trip_plan confirmations materialize as a savings goal
            if intent.get("intent") == "trip_plan" and intent.get("goal_intent"):
                gi = dict(intent["goal_intent"])
                gi["confirmed"] = True
                if intent.get("target_months"):
                    gi["target_months"] = intent["target_months"]
                intent = gi
            else:
                intent["confirmed"] = True
            result = _execute_write(db, user, session, intent)
            dialogue = {"pending_confirmation": None, "missing_slots": []}
            blackboard.update_dialogue(db, user, dialogue)
            blackboard.record_action(db, user, {"action": "confirm", "intent": intent.get("intent")})
            _save_assistant(db, session, result, agent="orchestrator")
            _finish_policy(policy, reward=1.0 if result.get("ok") else -0.5,
                           success=bool(result.get("ok")), turns=1)
            return result
        elif re.match(r"^(no|n|cancel|stop|don't|do not)\b", low):
            dialogue = {"pending_confirmation": None, "missing_slots": []}
            blackboard.update_dialogue(db, user, dialogue)
            result = {"ok": True, "kind": "cancelled",
                      "answer": "No problem — I cancelled that action. Anything else?"}
            _save_assistant(db, session, result, agent="orchestrator")
            _finish_policy(policy, reward=0.2, success=True, turns=1)
            return result
        # otherwise fall through and treat as new message

    # Continue a pending slot clarification: the assistant asked for missing
    # info (e.g. "what's the target amount?") and the user replies "45000
    # in 10 months". Short replies with numbers/months fill the slots.
    pending_clarify = dialogue.get("pending_clarify")
    if pending_clarify:
        if len(message.strip()) <= 80:
            base = dict(pending_clarify.get("intent") or {})
            missing = list(pending_clarify.get("missing") or [])
            low = message.lower().strip()
            am = re.search(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|thousand|lakhs?|rupees?)?", low)
            mm = _months_hint(low)
            filled = False
            for slot in list(missing):
                if slot in ("amount", "target_amount") and am:
                    val = float(am.group(1).replace(",", ""))
                    suf = am.group(2) or ""
                    if suf in ("k", "thousand"):
                        val *= 1000
                    elif suf in ("lakh", "lakhs"):
                        val *= 100000
                    base["amount"] = val
                    base["target_amount"] = val
                    missing.remove(slot)
                    filled = True
                elif slot == "target_months" and mm:
                    base["target_months"] = mm
                    missing.remove(slot)
                    filled = True
            if filled:
                still = _missing_slots(base)
                if still:
                    result = {"ok": True, "kind": "clarify", "needs_input": still[0],
                              "answer": _clarify_question(base, still[0])}
                    blackboard.update_dialogue(db, user, {
                        "pending_confirmation": None, "missing_slots": still,
                        "pending_clarify": {"intent": base, "missing": still}})
                    _save_assistant(db, session, result, agent="orchestrator")
                    return result
                summary = _write_summary(base)
                blackboard.update_dialogue(db, user, {
                    "pending_confirmation": {"intent": base, "summary": summary},
                    "missing_slots": []})
                result = {"ok": True, "kind": "confirm", "needs_confirmation": True,
                          "answer": f"{summary} Should I go ahead? (yes/no)"}
                _save_assistant(db, session, result, agent="policy")
                return result
        # unparseable or unrelated reply -> drop the clarification, new message
        blackboard.update_dialogue(db, user, {**dialogue, "pending_clarify": None})

    intent = await understand(message, history)
    intent["action"] = intent.get("intent")

    # Continue an in-progress trip-planning dialogue — but only when the
    # message is actually trip-related. Unrelated intents (report, bills,
    # budgets, ...) exit trip mode and are handled normally.
    if dialogue.get("trip_active") and intent.get("intent") != "smalltalk":
        slots = extract_trip_slots(message)
        if slots.get("cancel"):
            blackboard.update_dialogue(db, user, {**dialogue, "trip_active": False,
                                                   "trip_slots": {}})
            result = {"ok": True, "kind": "cancelled",
                      "answer": "No problem — trip planning cancelled. Start again anytime."}
            _save_assistant(db, session, result, agent="savings")
            return result
        # Explicit finance commands always win over trip-slot continuation —
        # e.g. "save ₹50,000 in 10 months" contains ₹ and months but is a
        # savings request, not trip details.
        hard_exit = {"savings_plan", "add_goal", "add_expense", "add_income",
                     "add_bill", "bill_reminder", "update_income", "report",
                     "export_report", "affordability"}
        it = intent.get("intent")
        has_slots = any(v is not None for k, v in slots.items())
        question_word = bool(re.match(r"^(what|how|when|where|why|is|are|can|could|should|do|does)\b",
                                      message.strip().lower()))
        short_answer = (len(message.strip()) <= 40 and "?" not in message
                        and not question_word and it not in hard_exit)
        if it in hard_exit:
            is_related = False
        else:
            is_related = (it == "trip_plan" or has_slots or short_answer)
        if is_related:
            filler = {"yes", "yeah", "yep", "ok", "okay", "sure", "no", "nope",
                      "maybe", "dunno", "idk", "thanks", "thank you"}
            if not has_slots and short_answer and \
               message.strip().lower().rstrip(".!?") not in filler:
                # bare answer like "goa" or "manali" -> treat as destination
                slots["destination"] = re.sub(r"^(to|in|for|visit)\s+", "",
                                              message.strip()).title()
            prev = dialogue.get("trip_slots") or {}
            merged = {**prev, **{k: v for k, v in slots.items() if v is not None}}
            # normalize key names used by the planner
            fake = {"intent": "trip_plan", "raw": message, **merged}
            if merged.get("budget"):
                fake["target_amount"] = merged["budget"]
            if merged.get("months"):
                fake["target_months"] = merged["months"]
            result = _trip_plan_response(db, user, fake)
            _store_trip_dialogue(db, user, result)
            _save_assistant(db, session, result, agent="savings")
            return result
        # unrelated question -> leave trip mode, handle normally
        blackboard.update_dialogue(db, user, {**dialogue, "trip_active": False,
                                               "trip_slots": {}})

    # Complex multi-part financial requests -> decompose into a collaborative
    # multi-agent task instead of routing to a single agent.
    from app.orchestration import decomposer, coordinator
    spec = decomposer.decompose(message, intent, board_state)
    if spec:
        return await coordinator.run_task(db, user, session, spec, board_state)

    # Trip planning: produce a finance plan and offer a savings goal
    if intent["intent"] == "trip_plan":
        result = _trip_plan_response(db, user, intent)
        _store_trip_dialogue(db, user, result)
        _save_assistant(db, session, result, agent="savings")
        return result

    # smalltalk handled without agents
    if intent["intent"] == "smalltalk":
        result = {"ok": True, "kind": "chat",
                  "answer": "Hi! I'm FinPilot AI. Ask me about your spending, budget, bills, savings goals, or say 'add expense ₹500 for lunch'."}
        _save_assistant(db, session, result, agent="orchestrator")
        return result

    # Update income is a write -> confirm
    if intent["intent"] == "update_income" and intent.get("amount"):
        dialogue = {"pending_confirmation": {"intent": intent, "summary":
                    f"Update monthly income to ₹{intent['amount']:,.0f}?"},
                    "missing_slots": []}
        blackboard.update_dialogue(db, user, dialogue)
        result = {"ok": True, "kind": "confirm", "needs_confirmation": True,
                  "answer": f"Your recorded monthly income was "
                            f"₹{board_state.get('summary', {}).get('total_income', 0):,.0f}. "
                            f"Would you like me to update it to ₹{intent['amount']:,.0f} "
                            f"and recalculate your budget while preserving your savings goals? (yes/no)"}
        _save_assistant(db, session, result, agent="policy")
        return result

    missing = _missing_slots(intent)

    # Write intents -> confirm unless already confirmed
    if intent["intent"] in WRITE_INTENTS and not intent.get("confirmed"):
        if missing:
            # ask for missing info instead of confirming
            result = {"ok": True, "kind": "clarify", "needs_input": missing[0],
                      "answer": _clarify_question(intent, missing[0])}
            dialogue = {"pending_confirmation": None, "missing_slots": missing,
                        "pending_clarify": {"intent": intent, "missing": missing}}
            blackboard.update_dialogue(db, user, dialogue)
            _save_assistant(db, session, result, agent="orchestrator")
            return result
        summary = _write_summary(intent)
        dialogue = {"pending_confirmation": {"intent": intent, "summary": summary},
                    "missing_slots": []}
        blackboard.update_dialogue(db, user, dialogue)
        result = {"ok": True, "kind": "confirm", "needs_confirmation": True,
                  "answer": f"{summary} Should I go ahead? (yes/no)"}
        _save_assistant(db, session, result, agent="policy")
        return result

    # Read intents -> policy selects agent
    decision = policy.decide(intent, board_state, dialogue, turn=turn)
    action = decision["action"]

    # Policy safety: write-ish actions on read intents are fine; ensure agent exists
    agent_name = action if action in AGENTS else policy.RULE_MAP.get(intent["intent"], "expense")
    if agent_name not in AGENTS:
        agent_name = "expense"

    _log_policy(db, user, session, decision, intent)

    agent = AGENTS[agent_name](db)
    agent_result = agent.execute(user, intent, board_state, session_id=session.id)
    blackboard.record_agent_output(db, user, agent_name, agent_result)

    # Gemini phrasing layered on factual agent output
    answer = agent_result.get("answer", "")
    facts = {k: v for k, v in agent_result.items() if k in (
        "summary", "budget", "goals", "bills", "report", "plan", "suggestion", "over_budget")}
    phrased = await gemini.phrase_response(
        f"Intent: {intent.get('intent')}. Data: {json.dumps(facts, default=str)[:3000]}",
        facts) if facts else None
    if phrased:
        answer = phrased

    result = dict(agent_result)
    result["answer"] = answer
    result["agent"] = agent_name
    result["policy_action"] = action
    if agent_result.get("needs_confirmation"):
        dialogue = {"pending_confirmation": {"intent": intent, "summary": answer},
                    "missing_slots": agent_result.get("needs_input") and [agent_result["needs_input"]] or []}
        blackboard.update_dialogue(db, user, dialogue)
        result["needs_confirmation"] = True

    blackboard.record_action(db, user, {"action": action, "intent": intent.get("intent"),
                                        "ok": agent_result.get("ok", True)})
    _save_assistant(db, session, result, agent=agent_name)
    _finish_policy(policy, reward=1.0 if agent_result.get("ok", True) else -0.5,
                   success=bool(agent_result.get("ok", True)), turns=1)
    return result


def _store_trip_dialogue(db: Session, user: User, result: dict):
    """Persist trip-planning slots / pending goal confirmation on the blackboard."""
    dialogue: dict = {}
    if result.get("needs_input") and result.get("trip_slots"):
        dialogue["trip_active"] = True
        dialogue["trip_slots"] = result["trip_slots"]
    elif result.get("needs_confirmation"):
        # keep collected slots so the user can refine ("make it 6 days")
        # and so a later 'yes' can still build the goal
        if result.get("trip_slots"):
            dialogue["trip_active"] = True
            dialogue["trip_slots"] = result["trip_slots"]
        dialogue["pending_confirmation"] = {
            "intent": result.get("goal_intent") or {
                "intent": "trip_plan",
                "target_amount": (result.get("plan") or {}).get("budget"),
                "target_months": (result.get("plan") or {}).get("months"),
                "raw": ""},
            "summary": result.get("answer", "")}
    else:
        dialogue["trip_active"] = False
        dialogue["trip_slots"] = {}
    blackboard.update_dialogue(db, user, dialogue)


# ordered questions for the trip-planning slot filling
TRIP_QUESTIONS = [
    ("destination", "Where are you planning to go? (e.g. Goa, Manali, Kerala)"),
    ("days", "How many days is the trip?"),
    ("travelers", "How many people are traveling?"),
    ("budget", "What's your total trip budget? (or tell me how many months you have to save)"),
]


def _trip_plan_response(db: Session, user: User, intent: dict) -> dict:
    """Dynamic trip finance plan driven by collected slots."""
    slots = {
        "destination": intent.get("destination"),
        "days": intent.get("days"),
        "travelers": intent.get("travelers"),
        "budget": intent.get("budget") or intent.get("target_amount") or intent.get("amount"),
        "months": intent.get("months") or intent.get("target_months"),
    }
    summ = board_state_summary(db, user)
    surplus = summ.get("net", 0)
    income = summ.get("total_income") or 0

    # figure out the first missing required slot
    missing_key = None
    for key, _q in TRIP_QUESTIONS:
        if key == "budget":
            if not slots["budget"] and not slots["months"]:
                missing_key = key
                break
        elif not slots.get(key):
            missing_key = key
            break

    if missing_key:
        q = dict(TRIP_QUESTIONS)[missing_key]
        known = ", ".join(f"{k}={v}" for k, v in slots.items() if v)
        return {"ok": True, "kind": "trip_plan", "needs_input": missing_key,
                "trip_slots": slots,
                "answer": (f"Let's plan it properly. {q}" +
                           (f" (so far: {known})" if known else "") +
                           " — or say 'stop' to cancel.")}

    # all slots present -> build the plan
    dest, days, travelers = slots["destination"], int(slots["days"]), int(slots["travelers"])
    budget = float(slots["budget"]) if slots["budget"] else None
    months = int(slots["months"]) if slots["months"] else None

    # suggest a budget if none given: 3 months of surplus or 15% of monthly income
    if budget is None:
        if surplus > 0:
            budget = round(surplus * 3, -2)
        elif income > 0:
            budget = round(income * 0.5, -2)
        else:
            budget = None
    if budget is None:
        return {"ok": True, "kind": "trip_plan", "needs_input": "budget",
                "trip_slots": slots,
                "answer": "I need a budget to plan around — what's a realistic total for the trip?"}

    per_person = budget / max(1, travelers)
    per_day_pp = per_person / max(1, days)
    # split adapts to trip length: longer trips shift spend toward stay/food
    stay_w = 0.42 if days >= 4 else 0.30
    split = [("Stay / hotel", stay_w), ("Food & dining", 0.28),
             ("Transport (to & around)", 0.15), ("Activities", 0.10), ("Misc", 1 - stay_w - 0.28 - 0.15 - 0.10)]
    monthly_needed = round(budget / months, 2) if months else None

    lines = "; ".join(f"{n} ~₹{budget * p:,.0f}" for n, p in split)
    ans = (f"{dest} plan for {days} days, {travelers} traveler{'s' if travelers > 1 else ''}, "
           f"budget ₹{budget:,.0f} (₹{per_day_pp:,.0f}/person/day): {lines}. ")
    if monthly_needed:
        ans += f"That's ₹{monthly_needed:,.0f}/month over {months} months. "
    if income or surplus:
        verdict = "comfortably within" if budget <= max(surplus * (months or 1), income * 0.3) else \
                  "tight but possible" if budget <= income * 0.6 else "ambitious"
        ans += f"Against your recorded finances (income ₹{income:,.0f}/mo, surplus ₹{surplus:,.0f}/mo) this looks {verdict}. "
    ans += "Want me to create a savings goal for this trip?"

    return {"ok": True, "kind": "trip_plan",
            "plan": {"destination": dest, "days": days, "travelers": travelers,
                     "budget": budget, "months": months, "monthly_needed": monthly_needed,
                     "per_day_per_person": round(per_day_pp, 2), "split": split},
            "trip_slots": slots, "needs_confirmation": True,
            "goal_intent": {"intent": "add_goal",
                            "goal_name": f"{dest} trip",
                            "target_amount": budget,
                            "target_months": months,
                            "confirmed": False, "raw": ""},
            "answer": ans}


def board_state_summary(db: Session, user: User) -> dict:
    from app.services.finance import monthly_summary
    t = date.today()
    return monthly_summary(db, user.id, t.year, t.month)


def _write_summary(intent: dict) -> str:
    it = intent.get("intent")
    amt = intent.get("amount") or intent.get("target_amount")
    if it == "add_expense":
        return f"Add expense of ₹{amt:,.0f} ({intent.get('category') or 'Other'})?"
    if it == "add_income":
        return f"Add income of ₹{amt:,.0f} ({intent.get('source') or 'Salary'})?"
    if it == "add_bill":
        return f"Add bill '{intent.get('bill_name') or intent.get('description') or 'Bill'}' of ₹{amt:,.0f}?"
    if it in ("add_goal", "savings_plan"):
        return f"Create savings goal '{intent.get('goal_name') or 'Goal'}' targeting ₹{amt:,.0f}?"
    if it == "update_income":
        return f"Update monthly income to ₹{amt:,.0f}?"
    return "Confirm this action?"


def _months_hint(message: str):
    m = re.search(r"(\d+)\s*(?:months?|mos?)", message.lower())
    return int(m.group(1)) if m else None


def extract_trip_slots(text: str) -> dict:
    """Pull trip-planning slots out of a free-text message."""
    t = text.lower()
    slots: dict = {}
    # budget: needs ₹/rs/inr, a budget keyword, or a k/thousand/rupees suffix —
    # bare numbers are usually days/travelers/months instead
    bm = re.search(r"(?:₹|rs\.?|inr)\s*([0-9][0-9,]*(?:\.\d+)?)|"
                   r"(?:upto|up to|under|within|around|about|budget(?:\s+of)?)\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|thousand|rupees?)?|"
                   r"([0-9][0-9,]*(?:\.\d+)?)\s*(k|thousand|lakhs?|rupees?)\b", t)
    if bm:
        try:
            num = next(g for g in bm.groups() if g and g.replace(",", "").replace(".", "").isdigit())
            val = float(num.replace(",", ""))
            suffix = bm.group(3) or bm.group(5) or ""
            if suffix in ("k", "thousand"):
                val *= 1000
            elif suffix in ("lakh", "lakhs"):
                val *= 100000
            slots["budget"] = val
        except (ValueError, StopIteration):
            pass
    dm = re.search(r"(\d+)\s*(?:days?|nights?)", t)
    if dm:
        slots["days"] = int(dm.group(1))
    elif "weekend" in t:
        slots["days"] = 2
    elif re.search(r"\ba week\b|\b7 days\b", t):
        slots["days"] = 7
    tm = re.search(r"(\d+)\s*(?:people|persons|travell?ers?|pax|of us|friends)", t)
    if tm:
        slots["travelers"] = int(tm.group(1))
    elif "couple" in t:
        slots["travelers"] = 2
    elif "family" in t:
        slots["travelers"] = 4
    elif "solo" in t or "alone" in t:
        slots["travelers"] = 1
    mm = _months_hint(t)
    if mm:
        slots["months"] = mm
    # destination: word after to/in/for near trip words, else capitalized place
    dest = re.search(r"(?:to|in|for|visit)\s+([a-zA-Z][a-zA-Z ]{2,20}?)(?:\s+(?:trip|travel|plan|vacation|holiday|tour|in|for|with|on|budget|upto|under)\b|$|[,.])", t)
    if dest:
        d = dest.group(1).strip()
        # reject captures that are really other slots ("2 people", "500 budget")
        if d not in ("a", "an", "the", "my", "our") and \
           not re.search(r"\d|people|person|budget|month|week|day|rs\.?|inr", d):
            slots["destination"] = d.title()
    # fallback: "<place> trip/travel/vacation" e.g. "goa trip", "kerala travel"
    if "destination" not in slots:
        pm = re.search(r"([a-z][a-z]+)\s+(?:trip|travel|vacation|holiday|tour)\b", t)
        if pm and pm.group(1) not in ("a", "an", "the", "my", "our", "plan", "budget", "cost"):
            slots["destination"] = pm.group(1).title()
    if re.search(r"\b(stop|cancel|never mind|forget it)\b", t):
        slots["cancel"] = True
    return slots


def _clarify_question(intent: dict, slot: str) -> str:
    return {
        "amount": "Sure — what's the amount?",
        "target_amount": "What's the target amount you want to save?",
        "target_months": "And in how many months do you want to reach it?",
        "bill": "Which bill should I mark as paid?",
        "categories": "Which categories and limits should the budget have?",
    }.get(slot, f"Could you provide the {slot.replace('_', ' ')}?")


def _execute_write(db: Session, user: User, session: ConversationSession, intent: dict) -> dict:
    """Perform confirmed write operations via the appropriate agent/service."""
    it = intent.get("intent")
    # confirmed trip plans become savings goals
    if it == "trip_plan":
        intent = {"intent": "add_goal", "confirmed": True, "raw": "",
                  "goal_name": "Trip fund",
                  "target_amount": intent.get("target_amount") or intent.get("amount"),
                  "target_months": intent.get("target_months")}
        it = "add_goal"
    board_state = blackboard.get_state(db, user, refresh=True)
    if it == "update_income":
        from app.models.db_models import UserProfile
        from app.services.notifications import notify
        p = db.query(UserProfile).filter_by(user_id=user.id).first()
        old = p.monthly_income if p else 0
        if p:
            p.monthly_income = float(intent["amount"])
            db.commit()
        notify(db, user.id, "info", "Income updated",
               f"Monthly income updated from ₹{old:,.0f} to ₹{intent['amount']:,.0f}. "
               "Ask me to recalculate your budget to apply the change.")
        db.commit()
        return {"ok": True, "kind": "action_result",
                "answer": f"Updated your monthly income to ₹{intent['amount']:,.0f}. "
                          "Would you like me to recalculate your budget with the new income?"}
    # delegate to agents with confirmed flag
    agent_map = {"add_expense": "expense", "add_goal": "savings",
                 "savings_plan": "savings", "add_bill": "bill"}
    agent_name = agent_map.get(it)
    if agent_name:
        agent = AGENTS[agent_name](db)
        res = agent.execute(user, intent, board_state, session_id=session.id)
        # savings_plan confirmed -> create the goal directly
        if it == "savings_plan" and res.get("plan"):
            plan = res["plan"]
            g_intent = {"intent": "add_goal", "goal_name": plan["name"],
                        "target_amount": plan["target"], "target_months": plan["months"],
                        "confirmed": True, "raw": ""}
            agent2 = AGENTS["savings"](db)
            res = agent2.execute(user, g_intent, board_state, session_id=session.id)
        return res
    return {"ok": False, "answer": "I couldn't process that action."}


def _log_policy(db, user, session, decision, intent):
    try:
        db.add(PolicyActionHistory(
            user_id=user.id, session_id=session.id,
            state=json.dumps(decision.get("state", [])),
            probs=json.dumps(decision.get("masked_probs", [])),
            action=decision.get("action", ""),
            critic_value=decision.get("critic_value", 0.0),
            source="policy"))
        db.commit()
    except Exception:
        db.rollback()


def _save_assistant(db, session, result, agent=""):
    db.add(ConversationMessage(
        session_id=session.id, role="assistant",
        content=result.get("answer", ""),
        data=json.dumps({k: v for k, v in result.items() if k != "answer"}, default=str)[:8000],
        agent=agent))
    db.commit()


def _finish_policy(policy, reward, success, turns):
    try:
        policy.finish_episode(reward, success, turns)
    except Exception as e:
        logger.warning("policy update failed: %s", e)
