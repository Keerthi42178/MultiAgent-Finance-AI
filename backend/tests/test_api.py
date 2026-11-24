import pytest
from fastapi.testclient import TestClient

from main import app
from app.database.db import init_db


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/auth/register", json={
        "email": "test@example.com", "password": "password123", "full_name": "Test User"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["name"] == "FinPilot AI API"


def test_register_duplicate(client, auth):
    r = client.post("/api/auth/register", json={
        "email": "test@example.com", "password": "password123"})
    assert r.status_code == 409


def test_login_bad_password(client):
    r = client.post("/api/auth/login", json={"email": "test@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_expense_crud(client, auth):
    r = client.post("/api/expenses", json={
        "amount": 450.0, "date": "2025-11-05", "category": "Food & Groceries",
        "description": "groceries", "payment_method": "UPI"}, headers=auth)
    assert r.status_code == 200
    eid = r.json()["id"]

    r = client.get("/api/expenses", headers=auth)
    assert r.status_code == 200 and r.json()["total"] >= 1

    r = client.put(f"/api/expenses/{eid}", json={
        "amount": 500.0, "date": "2025-11-05", "category": "Food & Groceries",
        "description": "groceries", "payment_method": "UPI"}, headers=auth)
    assert r.status_code == 200 and r.json()["amount"] == 500.0

    r = client.delete(f"/api/expenses/{eid}", headers=auth)
    assert r.status_code == 200


def test_income_crud(client, auth):
    r = client.post("/api/income", json={
        "amount": 30000, "date": "2025-11-01", "source": "Salary"}, headers=auth)
    assert r.status_code == 200
    iid = r.json()["id"]
    r = client.delete(f"/api/income/{iid}", headers=auth)
    assert r.status_code == 200


def test_budget_upsert_and_status(client, auth):
    r = client.put("/api/budgets/2025/11", json={
        "month": 11, "year": 2025, "total_income": 30000,
        "categories": [{"category": "Food & Groceries", "limit_amount": 5000, "essential": True}]},
        headers=auth)
    assert r.status_code == 200
    data = r.json()
    assert any(c["category"] == "Food & Groceries" for c in data["categories"])


def test_bills_and_goals(client, auth):
    r = client.post("/api/bills", json={
        "name": "Electricity", "amount": 1200, "due_date": "2025-11-15",
        "frequency": "monthly"}, headers=auth)
    assert r.status_code == 200
    bid = r.json()["id"]
    r = client.post(f"/api/bills/{bid}/pay", headers=auth)
    assert r.status_code == 200

    r = client.post("/api/goals", json={
        "name": "Laptop", "target_amount": 80000, "saved_amount": 10000,
        "monthly_contribution": 5000}, headers=auth)
    assert r.status_code == 200
    gid = r.json()["id"]
    r = client.get("/api/goals", headers=auth)
    assert any(g["id"] == gid for g in r.json())


def test_csv_import(client, auth):
    csv_content = (
        "Date,Description,Debit,Credit,Balance\n"
        "01/11/2025,Salary,,30000,30000\n"
        "02/11/2025,DMart,1200,,28800\n"
        "03/11/2025,Petrol,2000,,26800\n"
    )
    r = client.post("/api/import/confirm",
                    files={"file": ("stmt.csv", csv_content, "text/csv")}, headers=auth)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["imported"] == 3
    assert data["expense_total"] == 3200
    assert data["income_total"] == 30000


def test_csv_preview(client, auth):
    csv_content = "Date,Description,Debit,Credit\n01/11/2025,X,100,\n"
    r = client.post("/api/import/preview",
                    files={"file": ("s.csv", csv_content, "text/csv")}, headers=auth)
    assert r.status_code == 200
    assert "Date" in r.json()["headers"]


def test_chat_spending_query(client, auth):
    r = client.post("/api/chat/message", json={"message": "How much did I spend this month?"},
                    headers=auth)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"]
    assert data["session_id"]


def test_chat_add_expense_confirmation(client, auth):
    r = client.post("/api/chat/message", json={"message": "add expense 500 for lunch"},
                    headers=auth)
    assert r.status_code == 200
    # should ask for confirmation or amount
    assert r.json()["answer"]


def test_chat_trip_plan(client, auth):
    # full details in one message -> immediate plan + goal offer
    r = client.post("/api/chat/message",
                    json={"message": "plan a 5 day goa trip for 2 people with budget 10000 in 3 months"},
                    headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "Goa" in d["answer"]
    # confirm -> creates savings goal
    r2 = client.post("/api/chat/message", json={"message": "yes"}, headers=auth)
    assert r2.status_code == 200
    assert "goal" in r2.json()["answer"].lower() or "Goal" in r2.json()["answer"]


def test_chat_trip_plan_slot_filling(client, auth):
    # sparse request -> assistant asks for missing details, one at a time
    r = client.post("/api/chat/message",
                    json={"message": "i want to plan a manali trip"},
                    headers=auth)
    assert r.status_code == 200, r.text
    assert "?" in r.json()["answer"]  # asks a clarifying question
    # answer the questions across turns; plan must adapt to each reply
    r = client.post("/api/chat/message",
                    json={"message": "7 days for 2 people budget 20000 in 4 months"},
                    headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "Manali" in d["answer"] and "7" in d["answer"]
    assert "20,000" in d["answer"] or "20000" in d["answer"]
    r2 = client.post("/api/chat/message", json={"message": "yes"}, headers=auth)
    assert r2.status_code == 200
    assert "goal" in r2.json()["answer"].lower() or "Goal" in r2.json()["answer"]


def test_chat_trip_mode_does_not_hijack_other_intents(client, auth):
    # start trip planning, then ask an unrelated question mid-flow
    client.post("/api/chat/message",
                json={"message": "i want to plan a goa trip"}, headers=auth)
    r = client.post("/api/chat/message",
                    json={"message": "generate my monthly financial report"}, headers=auth)
    assert r.status_code == 200, r.text
    ans = r.json()["answer"].lower()
    assert "trip plan" not in ans and "savings goal for this trip" not in ans


def test_chat_savings_during_trip_mode(client, auth):
    # mid trip-planning, a savings request must not be read as trip slots
    client.post("/api/chat/message",
                json={"message": "i want to plan a goa trip"}, headers=auth)
    r = client.post("/api/chat/message",
                    json={"message": "I want to save ₹50,000 in 10 months"}, headers=auth)
    assert r.status_code == 200, r.text
    ans = r.json()["answer"].lower()
    assert "trip plan" not in ans and "traveler" not in ans


def test_chat_goal_creation_clarify_flow(client, auth):
    # goal request without amount -> asks, then fills from short reply, then confirms
    r = client.post("/api/chat/message",
                    json={"message": "create a saving goal plan for to buy a laptop"},
                    headers=auth)
    assert r.status_code == 200, r.text
    assert "?" in r.json()["answer"]  # asks for the target amount
    r = client.post("/api/chat/message", json={"message": "45000 in 10 months"}, headers=auth)
    assert r.status_code == 200, r.text
    assert "45,000" in r.json()["answer"] or "45000" in r.json()["answer"]
    r = client.post("/api/chat/message", json={"message": "yes"}, headers=auth)
    assert r.status_code == 200
    goals = client.get("/api/goals", headers=auth).json()
    assert any("laptop" in (g.get("name") or "").lower() for g in goals)


def test_chat_affordability(client, auth):
    r = client.post("/api/chat/message", json={"message": "can I afford a laptop worth 45000?"},
                    headers=auth)
    assert r.status_code == 200
    assert r.json()["answer"]


def test_dashboard(client, auth):
    r = client.get("/api/dashboard", headers=auth)
    assert r.status_code == 200
    d = r.json()
    assert "summary" in d and "trend" in d and "goals" in d


def test_notifications(client, auth):
    r = client.get("/api/notifications", headers=auth)
    assert r.status_code == 200


def test_policy_status(client, auth):
    r = client.get("/api/policy", headers=auth)
    assert r.status_code == 200
    assert "actions" in r.json()


def test_profile_update(client, auth):
    r = client.put("/api/profile", json={"monthly_income": 35000, "onboarding_completed": True},
                   headers=auth)
    assert r.status_code == 200
    assert r.json()["monthly_income"] == 35000


def test_unauthorized(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 401
