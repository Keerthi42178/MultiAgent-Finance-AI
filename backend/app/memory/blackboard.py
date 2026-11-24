"""Shared blackboard memory: structured per-user state for all agents."""
import json
from datetime import date, datetime

from sqlalchemy.orm import Session

from app.models.db_models import BlackboardMemory, User
from app.services.finance import (
    budget_status, financial_profile, goals_progress, monthly_summary,
    upcoming_bills,
)


def _load_json(raw, default):
    try:
        return json.loads(raw) if raw else default
    except Exception:
        return default


def get_board(db: Session, user: User) -> BlackboardMemory:
    board = db.query(BlackboardMemory).filter_by(user_id=user.id).first()
    if not board:
        board = BlackboardMemory(user_id=user.id)
        db.add(board)
        db.commit()
        db.refresh(board)
    return board


def refresh_state(db: Session, user: User) -> dict:
    """Rebuild the authoritative financial snapshot on the blackboard."""
    board = get_board(db, user)
    today = date.today()
    summary = monthly_summary(db, user.id, today.year, today.month)
    state = {
        "profile": financial_profile(db, user.id),
        "month": f"{today.year}-{today.month:02d}",
        "summary": summary,
        "budget": budget_status(db, user.id, today.year, today.month),
        "bills_upcoming": upcoming_bills(db, user.id, days=30)[:10],
        "goals": goals_progress(db, user.id),
        "currency": "₹",
        "generated_at": today.isoformat(),
    }
    board.state = json.dumps(state, default=str)
    db.commit()
    return state


def get_state(db: Session, user: User, refresh: bool = False) -> dict:
    board = get_board(db, user)
    if refresh or not board.state:
        return refresh_state(db, user)
    return _load_json(board.state, {})


def update_dialogue(db: Session, user: User, dialogue: dict):
    board = get_board(db, user)
    board.dialogue = json.dumps(dialogue, default=str)
    db.commit()


def get_dialogue(db: Session, user: User) -> dict:
    board = get_board(db, user)
    return _load_json(board.dialogue, {})


def record_agent_output(db: Session, user: User, agent: str, output: dict):
    board = get_board(db, user)
    outputs = _load_json(board.agent_outputs, {})
    outputs[agent] = output
    board.agent_outputs = json.dumps(outputs, default=str)
    db.commit()


def get_agent_outputs(db: Session, user: User) -> dict:
    board = get_board(db, user)
    return _load_json(board.agent_outputs, {})


def record_action(db: Session, user: User, action: dict):
    board = get_board(db, user)
    history = _load_json(board.action_history, [])
    history.append(action)
    board.action_history = json.dumps(history[-200:], default=str)
    db.commit()


def get_preferences(db: Session, user: User) -> dict:
    board = get_board(db, user)
    return _load_json(board.preferences, {})


def set_preference(db: Session, user: User, key: str, value):
    board = get_board(db, user)
    prefs = _load_json(board.preferences, {})
    prefs[key] = value
    board.preferences = json.dumps(prefs, default=str)
    db.commit()


# ---------- task-scoped shared memory ----------
# Task state lives in the task_executions table (per-user isolated). These
# helpers give agents a uniform read/write interface into a running task.

def task_create(db: Session, user: User, task: dict):
    """Persist a new task. Returns the TaskExecution row."""
    from app.models.db_models import TaskExecution
    row = TaskExecution(
        user_id=user.id, session_id=task.get("session_id"),
        task_key=task["task_key"], original_request=task.get("original_request", ""),
        intent=task.get("intent", ""), primary_goal=task.get("primary_goal", ""),
        status="pending", subtasks=json.dumps(task.get("subtasks", []), default=str),
        intermediate_results="{}", trace="[]",
        stated=json.dumps(task.get("stated", {}), default=str))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def task_get(db: Session, user: User, task_key: str):
    from app.models.db_models import TaskExecution
    return db.query(TaskExecution).filter_by(user_id=user.id, task_key=task_key).first()


def task_view(row) -> dict:
    """Dict view of a task row for agents/policy."""
    if row is None:
        return {}
    return {
        "task_key": row.task_key, "status": row.status,
        "original_request": row.original_request, "intent": row.intent,
        "primary_goal": row.primary_goal,
        "subtasks": _load_json(row.subtasks, []),
        "results": _load_json(row.intermediate_results, {}),
        "stated": _load_json(row.stated, {}),
        "trace": _load_json(row.trace, []),
    }


def task_set_status(db: Session, row, status: str):
    row.status = status
    db.commit()


def task_set_subtasks(db: Session, row, subtasks: list):
    row.subtasks = json.dumps(subtasks, default=str)
    db.commit()


def task_add_result(db: Session, user: User, row, agent: str, output: dict):
    """Store an agent's output into the task AND the shared agent_outputs
    blackboard so downstream agents can consume it."""
    results = _load_json(row.intermediate_results, {})
    results[agent] = output
    row.intermediate_results = json.dumps(results, default=str)
    record_agent_output(db, user, agent, output)
    db.commit()


def task_add_trace(db: Session, row, actor: str, action: str, detail: str = "",
                   status: str = "completed", **extra):
    trace = _load_json(row.trace, [])
    ev = {"actor": actor, "action": action, "detail": detail,
          "status": status, "ts": datetime.utcnow().isoformat()}
    ev.update(extra)
    trace.append(ev)
    row.trace = json.dumps(trace[-300:], default=str)
    db.commit()


def task_set_final(db: Session, row, final: dict):
    row.final_result = json.dumps(final, default=str)
    row.status = "completed"
    db.commit()
