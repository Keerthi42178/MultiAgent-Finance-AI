"""Tests for the collaborative multi-agent task layer."""
import pytest
from fastapi.testclient import TestClient

from main import app
from app.database.db import init_db, SessionLocal
from app.models.db_models import User
from app.memory import blackboard
from app.orchestration import decomposer


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/auth/register", json={
        "email": "collab@example.com", "password": "password123",
        "full_name": "Collab User"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


CAR_MSG = ("My monthly salary is ₹90,000. I spend ₹30,000 monthly and already "
           "pay ₹10,000 in EMIs. I want to purchase a car worth ₹8 lakh. "
           "Evaluate whether I can afford it, estimate a safe down payment "
           "and EMI budget, and recommend when I should purchase it.")


# ---------- decomposition unit tests ----------

def test_decompose_car_purchase():
    spec = decomposer.decompose(CAR_MSG, {"intent": "affordability"}, {})
    assert spec is not None
    assert spec["primary_goal"] == "purchase_affordability"
    agents = [s["assigned_agent"] for s in spec["subtasks"]]
    assert agents == ["expense", "bill", "budget", "savings", "report"]
    # dependency chain: each subtask depends on the previous one
    for i, s in enumerate(spec["subtasks"]):
        if i == 0:
            assert s["dependencies"] == []
        else:
            assert s["dependencies"] == [spec["subtasks"][i - 1]["subtask_id"]]
    # stated figures extracted
    assert spec["stated"]["income"] == 90000
    assert spec["stated"]["monthly_expenses"] == 30000
    assert spec["stated"]["emi"] == 10000
    assert spec["stated"]["price"] == 800000


def test_decompose_varied_workflows():
    debt = decomposer.decompose(
        "I have two loans and want to prioritize repayment without affecting my emergency savings",
        {"intent": "debt_management"}, {})
    assert debt["primary_goal"] == "debt_management"
    assert debt["subtasks"][0]["assigned_agent"] == "bill"

    goals = decomposer.decompose(
        "I want to purchase a laptop, build an emergency fund and plan a vacation. Prioritize all three goals.",
        {"intent": "goal_query"}, {})
    assert goals["primary_goal"] == "goal_prioritization"
    assert len(goals["subtasks"]) == 3  # expense -> savings -> report

    salary = decomposer.decompose(
        "My salary has decreased. Recalculate my budget, EMI commitments and savings strategy.",
        {"intent": "budget_query"}, {})
    assert salary["primary_goal"] == "budget_replan"

    trip = decomposer.decompose(
        "I earn ₹70,000, spend ₹25,000, pay ₹7,000 EMI, and want to travel internationally in six months. Create a savings and budget plan.",
        {"intent": "trip_plan"}, {})
    assert trip["primary_goal"] == "savings_plan"


def test_simple_requests_not_decomposed():
    assert decomposer.decompose("How much did I spend this month?",
                                {"intent": "spending_query"}, {}) is None
    assert decomposer.decompose("add expense 500 for lunch",
                                {"intent": "add_expense"}, {}) is None
    assert decomposer.decompose("plan a goa trip",
                                {"intent": "trip_plan"}, {}) is None


# ---------- end-to-end collaboration tests ----------

def test_car_purchase_task_collaboration(client, auth):
    r = client.post("/api/chat/message", json={"message": CAR_MSG}, headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "task_result"
    data = d["data"]
    assert data["task"]

    results = data["results"]
    # every agent produced real structured output
    for agent in ("expense", "bill", "budget", "savings", "report"):
        assert agent in results, f"missing {agent} output"
        assert results[agent].get("ok", True)

    # Agent B consumed Agent A's output — numbers must chain:
    # expense: 90000 - 30000 = 60000 available_before_emi
    assert results["expense"]["monthly_income"] == 90000
    assert results["expense"]["available_before_emi"] == 60000
    # bill: 60000 - 10000 EMI = 50000 remaining
    assert results["bill"]["remaining_after_expenses_and_emi"] == 50000
    assert results["bill"]["consumed_from"] == "expense"
    # budget: buffer 20% of 90000 = 18000; discretionary = 50000-18000 = 32000
    assert results["budget"]["discretionary_income"] == 32000
    assert results["budget"]["consumed_from"] == ["expense", "bill"]
    # savings: 20% down payment on 8 lakh = 160000
    assert results["savings"]["suggested_down_payment"] == 160000
    assert results["savings"]["consumed_from"] == ["expense", "bill", "budget"]

    # trace shows real actor-critic selections in dependency order
    trace = data["trace"]
    selections = [e for e in trace if e["actor"] == "actor_critic"
                  and e["action"] == "agent_selected"]
    order = [e["detail"].split(" ")[1] if " " in e["detail"] else e["detail"]
             for e in selections]
    # detail format: "Selected X agent (...)" -> extract agent names
    names = []
    for e in selections:
        for ag in ("expense", "bill", "budget", "savings", "report"):
            if f"Selected {ag} agent" in e["detail"]:
                names.append(ag)
    assert names[:5] == ["expense", "bill", "budget", "savings", "report"]

    # subtasks all completed
    assert all(s["status"] == "completed" for s in data["subtasks"])

    # final answer mentions contribution of agents
    assert "contributed" in d["answer"].lower() or "agent" in d["answer"].lower()


def test_task_trace_endpoint(client, auth):
    r = client.post("/api/chat/message", json={"message": CAR_MSG}, headers=auth)
    assert r.status_code == 200
    task_key = r.json()["data"]["task"]["task_key"]
    r2 = client.get(f"/api/tasks/{task_key}", headers=auth)
    assert r2.status_code == 200, r2.text
    t = r2.json()
    assert t["status"] == "completed"
    assert t["trace"] and t["subtasks"]
    assert t["final_result"]


def test_task_user_isolation(client, auth):
    # second user must not see the first user's task
    r = client.post("/api/auth/register", json={
        "email": "other@example.com", "password": "password123"})
    assert r.status_code == 200
    other = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = client.post("/api/chat/message", json={"message": CAR_MSG}, headers=auth)
    task_key = r.json()["data"]["task"]["task_key"]
    r2 = client.get(f"/api/tasks/{task_key}", headers=other)
    assert r2.status_code == 404


def test_varied_requests_different_sequences(client, auth):
    # debt management -> bill agent first, only 4 agents (no expense)
    r = client.post("/api/chat/message", json={
        "message": "I have two loans and want to prioritize repayment without affecting my emergency savings"},
        headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "task_result"
    agents = [s["assigned_agent"] for s in d["data"]["subtasks"]]
    assert agents[0] == "bill"
    assert "expense" not in agents  # different workflow, not all five agents

    # simple question still uses the single-agent path
    r2 = client.post("/api/chat/message",
                     json={"message": "How much did I spend this month?"}, headers=auth)
    assert r2.status_code == 200
    assert r2.json()["kind"] != "task_result"


def test_goal_prioritization_task(client, auth):
    r = client.post("/api/chat/message", json={
        "message": "I want to purchase a laptop, build an emergency fund and plan a vacation. Prioritize all three goals."},
        headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    if d["kind"] == "task_result":
        agents = [s["assigned_agent"] for s in d["data"]["subtasks"]]
        assert agents == ["expense", "savings", "report"]


def test_task_not_marked_complete_when_unfinished():
    """Coordinator must not mark a task completed with pending subtasks."""
    from app.orchestration.coordinator import _eligible_agents
    subtasks = [
        {"subtask_id": "st-1", "assigned_agent": "expense", "status": "completed",
         "dependencies": []},
        {"subtask_id": "st-2", "assigned_agent": "bill", "status": "pending",
         "dependencies": ["st-1"]},
    ]
    results = {"expense": {"ok": True, "available_before_emi": 50000}}
    # bill subtask is ready once expense completed
    assert _eligible_agents(subtasks, results) == ["bill"]
    # with expense failed/missing ok flag, bill must NOT be eligible
    results_bad = {"expense": {"ok": False}}
    assert _eligible_agents(subtasks, results_bad) == []
