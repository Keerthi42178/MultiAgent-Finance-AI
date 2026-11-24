"""Task decomposition engine.

Detects complex multi-part financial requests and decomposes them into a DAG
of dependent subtasks assigned to the five specialized agents. Simple requests
return None and continue through the normal single-agent path.
"""
import re

# Agent execution order used for dependency chains. Each workflow lists the
# agents in the order their findings feed the next agent.
WORKFLOWS = {
    # Big purchase: income/expense analysis -> obligations -> discretionary
    # capacity -> savings/down-payment -> consolidated recommendation
    "purchase_affordability": ["expense", "bill", "budget", "savings", "report"],
    # Savings / multiple goals: surplus analysis -> goal evaluation -> report
    "savings_plan": ["expense", "savings", "report"],
    # Debt payoff: obligations first -> budget capacity -> savings buffer -> report
    "debt_management": ["bill", "budget", "savings", "report"],
    # Competing goals: analyze all -> prioritize -> report
    "goal_prioritization": ["expense", "savings", "report"],
    # Income change: re-analyze -> rebudget -> resave -> report
    "budget_replan": ["expense", "budget", "savings", "report"],
}

SUBTASK_DESCRIPTIONS = {
    "expense": "Analyze income, expenses and available surplus",
    "bill": "Review EMI and bill obligations, compute remaining capacity",
    "budget": "Evaluate discretionary income and affordable additional EMI",
    "savings": "Assess down-payment feasibility, timeline and goal risks",
    "report": "Aggregate all agent findings into one recommendation",
}


def _amounts(text: str) -> list[float]:
    "_pull every ₹/number-with-lakh-thousand from the message._"""
    vals = []
    for m in re.finditer(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(lakh|lakhs|l|thousand|k|crore)?", text.lower()):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        suf = m.group(2) or ""
        if suf in ("lakh", "lakhs", "l"):
            v *= 100000
        elif suf in ("thousand", "k"):
            v *= 1000
        elif suf == "crore":
            v *= 10000000
        vals.append(v)
    return vals


def _extract_stated(message: str) -> dict:
    """Pull user-stated financial figures (distinct from DB records)."""
    m = message.lower()
    stated: dict = {}
    nums = _amounts(m)

    inc = re.search(r"(?:earn|income|salary|makes?|getting|ctc)[^\d]{0,20}([0-9][0-9,]*(?:\.\d+)?)\s*(lakh|lakhs|thousand|k)?", m)
    if inc:
        v = float(inc.group(1).replace(",", ""))
        s = inc.group(2) or ""
        stated["income"] = v * (100000 if s in ("lakh", "lakhs") else 1000 if s in ("thousand", "k") else 1)
    elif nums:
        # fallback: first large number near 'earn/salary' handled above;
        # otherwise leave unset — agents use DB records
        pass

    spend = re.search(r"(?:spend|spending|expenses?)[^\d]{0,20}([0-9][0-9,]*(?:\.\d+)?)", m)
    if spend:
        stated["monthly_expenses"] = float(spend.group(1).replace(",", ""))

    emi = re.search(r"(?:emi|loans?|debt|installment)[^\d]{0,20}([0-9][0-9,]*(?:\.\d+)?)", m) or \
          re.search(r"(?:pay|paying|paid)[^\d]{0,20}([0-9][0-9,]*(?:\.\d+)?)\s*(?:in\s+)?(?:emis?|loans?|installments?)", m)
    if emi:
        stated["emi"] = float(emi.group(1).replace(",", ""))

    # purchase price: amount nearest to buy/purchase/afford/car/house keywords
    price = None
    pm = re.search(r"(?:buy|purchase|afford(?:able)?|book|booked|costs?|worth|price[ds]?)[^\d]{0,30}([0-9][0-9,]*(?:\.\d+)?)\s*(lakh|lakhs|thousand|k|crore)?", m) or \
         re.search(r"([0-9][0-9,]*(?:\.\d+)?)\s*(lakh|lakhs|thousand|k|crore)?\s*(?:car|house|home|flat|bike|motorcycle|phone|laptop|item|vacation|trip|travel)", m)
    if pm:
        try:
            v = float(pm.group(1).replace(",", ""))
            s = pm.group(2) or ""
            if s in ("lakh", "lakhs"):
                v *= 100000
            elif s in ("thousand", "k"):
                v *= 1000
            elif s == "crore":
                v *= 10000000
            price = v
        except (ValueError, IndexError):
            pass
    if price:
        stated["price"] = price

    tm = re.search(r"(\d+)\s*(?:months?|mos?)", m)
    if tm:
        stated["months"] = int(tm.group(1))
    return stated


def classify_request(message: str, intent: dict) -> str | None:
    """Return a workflow key for genuinely complex multi-agent requests."""
    m = message.lower()

    # multiple competing goals -> prioritization workflow
    goal_hits = len(re.findall(r"\b(laptop|phone|car|house|flat|bike|vacation|trip|emergency fund|goal)\b", m))
    if re.search(r"\b(prioriti[sz]e|multiple goals|all three|both)\b", m) and goal_hits >= 2:
        return "goal_prioritization"

    # debt management
    if re.search(r"\b(two loans|multiple loans|debt|loans? repayment|prioriti[sz]e repayment|pay off)\b", m) and \
       re.search(r"\b(loan|debt|emi|repay|prioriti[sz]e)\b", m):
        return "debt_management"

    # income reduction / salary change -> budget replan
    if re.search(r"\b(salary (has )?(decreased|reduced|cut|dropped)|income (decreased|reduced|dropped|cut)|pay cut|lost my job|salary cut)\b", m):
        return "budget_replan"

    # big purchase affordability (car/house/expensive item + evaluate/afford/plan)
    if re.search(r"\b(car|house|home|flat|bike|motorcycle|property)\b", m) and \
       re.search(r"\b(afford|purchase|buy|emi|loan|down payment|evaluate|plan)\b", m):
        return "purchase_affordability"

    # generic large purchase with explicit evaluation request
    if re.search(r"\b(afford|can i afford|evaluate|worth)\b", m) and re.search(r"\b(buy|purchase|price|cost|₹|[0-9])\b", m):
        amt = _amounts(m)
        if amt and max(amt) >= 50000:
            return "purchase_affordability"

    # savings accumulation for a goal/trip with timeline -> collaborative savings plan
    if intent.get("intent") not in ("savings_plan", "trip_plan", "goal_query") and \
       re.search(r"\b(save|saving|savings)\b", m) and re.search(r"\b(in|within|over)\s+\d+\s*months?\b", m):
        return "savings_plan"

    # international / big trip savings plan
    if re.search(r"\binternationa\w*|\babroad\b|\bforeign\b", m) and \
       re.search(r"\b(trip|travel|vacation|tour)\b", m) and \
       re.search(r"\b(save|savings|budget|plan)\b", m):
        return "savings_plan"

    return None


def decompose(message: str, intent: dict, board_state: dict) -> dict | None:
    """Build a task spec for a complex request, or None for simple ones."""
    workflow = classify_request(message, intent)
    if not workflow:
        return None
    agents = WORKFLOWS[workflow]
    stated = _extract_stated(message)

    subtasks = []
    prev = None
    for i, agent in enumerate(agents):
        subtask = {
            "subtask_id": f"st-{i+1}",
            "assigned_agent": agent,
            "description": SUBTASK_DESCRIPTIONS[agent],
            "dependencies": [prev] if prev else [],
            "status": "pending",
            "input_data": {},
            "output_data": None,
            "retry_count": 0,
        }
        subtasks.append(subtask)
        prev = subtask["subtask_id"]

    return {
        "task_key": None,  # assigned by coordinator
        "original_request": message,
        "intent": intent.get("intent", "complex"),
        "primary_goal": workflow,
        "subtasks": subtasks,
        "stated": stated,
    }
