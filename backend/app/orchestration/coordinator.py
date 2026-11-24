"""Multi-agent coordination engine.

Executes a decomposed task as a dependency-aware sequence: the Actor-Critic
policy selects the next eligible agent at every step, agents consume prior
agents' outputs from task-scoped blackboard memory, and the Report Agent
aggregates everything into one final response.
"""
import json
import secrets

from app.memory import blackboard
from app.models.db_models import User
from app.orchestration import decomposer
from app.reinforcement_learning.actor_critic import get_policy

MAX_STEPS = 12          # safety bound: agents x retries
MAX_RETRIES = 2


def _eligible_agents(subtasks: list, results: dict) -> list[str]:
    """Agents whose subtasks are READY: status pending/ready and every
    dependency subtask completed with a successful output."""
    by_id = {s["subtask_id"]: s for s in subtasks}
    eligible = []
    for s in subtasks:
        if s.get("status") not in ("pending", "ready"):
            continue
        deps_ok = all(
            by_id.get(d, {}).get("status") == "completed" and
            (results.get(by_id[d]["assigned_agent"], {}) or {}).get("ok", True)
            for d in (s.get("dependencies") or [])
        )
        if deps_ok and s["assigned_agent"] not in eligible:
            eligible.append(s["assigned_agent"])
    return eligible


def _ready_subtask(subtasks: list, agent: str) -> dict | None:
    for s in subtasks:
        if s.get("assigned_agent") == agent and s.get("status") in ("pending", "ready"):
            return s
    return None


def _agent_ok(result: dict) -> bool:
    return bool(result) and result.get("ok", True) and not result.get("error")


def _task_answer(task_view: dict, final: dict) -> dict:
    return {
        "ok": True,
        "kind": "task_result",
        "agent": "orchestrator",
        "task": {
            "task_key": task_view.get("task_key"),
            "status": task_view.get("status"),
            "primary_goal": task_view.get("primary_goal"),
            "trace": task_view.get("trace", []),
        },
        "answer": final.get("answer", ""),
        # flattened so /api/chat/message surfaces them in `data` directly
        "subtasks": task_view.get("subtasks", []),
        "results": {k: {kk: vv for kk, vv in (v or {}).items()
                        if kk != "answer"} for k, v in
                    (task_view.get("results") or {}).items()},
        "trace": task_view.get("trace", []),
        "final": final,
    }


async def run_task(db, user: User, session, spec: dict, board_state: dict) -> dict:
    """Execute a decomposed task end-to-end. Returns the chat response dict."""
    from app.agents import AGENTS
    from app.orchestration.orchestrator import _save_assistant

    policy = get_policy()
    spec["task_key"] = f"T-{secrets.token_hex(3)}"
    spec["session_id"] = getattr(session, "id", None)
    row = blackboard.task_create(db, user, spec)
    blackboard.task_add_trace(db, row, "orchestrator", "task_created",
                              f"Decomposed '{spec['original_request'][:80]}' into "
                              f"{len(spec['subtasks'])} subtasks "
                              f"({spec['primary_goal']})")

    executed = []          # agent names already run (for duplicate prevention)
    steps = 0
    final = {}
    reward = 0.0
    success = False

    while steps < MAX_STEPS:
        steps += 1
        tv = blackboard.task_view(row)
        subtasks = tv["subtasks"]
        results = tv["results"]

        eligible = _eligible_agents(subtasks, results)
        decision = policy.decide_task_step(tv, board_state, eligible)
        agent_name = decision["action"]
        blackboard.task_add_trace(
            db, row, "actor_critic", "agent_selected",
            detail=f"Selected {agent_name} agent (eligible: {', '.join(eligible) or 'none'})",
            probs=decision.get("masked_probs"))

        if agent_name == "complete" or agent_name not in AGENTS:
            # nothing eligible -> aggregation
            break

        subtask = _ready_subtask(subtasks, agent_name)
        if subtask is None:
            # policy picked an agent with no ready subtask; mark its pending
            # subtasks failed-safe by skipping this choice
            blackboard.task_add_trace(db, row, "coordinator", "skip",
                                      f"{agent_name} has no ready subtask; re-evaluating")
            continue

        if agent_name in executed and subtask.get("retry_count", 0) >= MAX_RETRIES:
            subtask["status"] = "failed"
            blackboard.task_set_subtasks(db, row, subtasks)
            continue

        # mark running
        subtask["status"] = "running"
        subtask["retry_count"] = subtask.get("retry_count", 0) + 1
        blackboard.task_set_subtasks(db, row, subtasks)
        blackboard.task_add_trace(db, row, agent_name, "started",
                                  subtask["description"])

        # inputs: subtask data + every prior agent output + user-stated figures
        # (each prior output keeps a truncated 'answer' so downstream agents —
        # especially the Report Agent — can cite what earlier agents found)
        prior = {}
        for a, o in results.items():
            entry = {k: v for k, v in (o or {}).items() if k != "answer"}
            ans = (o or {}).get("answer")
            if ans:
                entry["answer"] = str(ans)[:300]
            prior[a] = entry
        intent_for_agent = {
            "intent": spec["intent"] if agent_name == "report" else "task_step",
            "raw": spec["original_request"],
            "primary_goal": spec["primary_goal"],
        }
        task_ctx = {
            "task_key": row.task_key,
            "original_request": spec["original_request"],
            "primary_goal": spec["primary_goal"],
            "stated": tv.get("stated", {}),
            "prior_results": prior,
            "subtask": {k: v for k, v in subtask.items() if k != "output_data"},
        }

        agent = AGENTS[agent_name](db)
        try:
            result = agent.execute(
                user, intent_for_agent, board_state,
                session_id=getattr(session, "id", None),
                task=task_ctx)
        except Exception as exc:  # pragma: no cover - defensive
            result = {"agent": agent_name, "ok": False, "error": str(exc)}

        ok = _agent_ok(result)
        output = result if isinstance(result, dict) else {"ok": False}
        if not ok:
            output = {**output, "ok": False}

        blackboard.task_add_result(db, user, row, agent_name, output)
        executed.append(agent_name)

        # refresh subtask state
        subtasks = blackboard._load_json(row.subtasks, [])
        for s in subtasks:
            if s["subtask_id"] == subtask["subtask_id"]:
                s["status"] = "completed" if ok else (
                    "failed" if s.get("retry_count", 0) > MAX_RETRIES else "pending")
                s["output_data"] = {k: v for k, v in output.items()
                                    if k != "answer"} if ok else None
        blackboard.task_set_subtasks(db, row, subtasks)

        summary_bits = []
        for key in ("available_before_emi", "remaining_after_commitments",
                    "discretionary_income", "affordable_emi", "monthly_surplus",
                    "down_payment", "timeline_months"):
            if isinstance(output, dict) and output.get(key) is not None:
                summary_bits.append(f"{key}={output[key]}")
        blackboard.task_add_trace(
            db, row, agent_name, "completed" if ok else "failed",
            "; ".join(summary_bits) or (output.get("answer", "") or "")[:120],
            status="completed" if ok else "failed")

        if ok and agent_name == "report":
            final = output

    # task finished or step limit hit
    tv = blackboard.task_view(row)
    subtasks = tv["subtasks"]
    unfinished = [s for s in subtasks if s.get("status") not in ("completed", "failed", "skipped")]

    if not final:
        # aggregate via report agent if it hasn't run yet
        prior = {}
        for a, o in tv["results"].items():
            entry = {k: v for k, v in (o or {}).items() if k != "answer"}
            ans = (o or {}).get("answer")
            if ans:
                entry["answer"] = str(ans)[:300]
            prior[a] = entry
        intent_for_agent = {"intent": "task_step", "raw": spec["original_request"],
                            "primary_goal": spec["primary_goal"]}
        task_ctx = {"task_key": row.task_key, "original_request": spec["original_request"],
                    "primary_goal": spec["primary_goal"], "stated": tv.get("stated", {}),
                    "prior_results": prior, "subtask": {}}
        try:
            final = AGENTS["report"](db).execute(
                user, intent_for_agent, board_state,
                session_id=getattr(session, "id", None), task=task_ctx)
            blackboard.task_add_result(db, user, row, "report", final)
        except Exception as exc:
            final = {"ok": False, "error": str(exc),
                     "answer": "Task analysis completed but final aggregation failed."}

    all_ok = all(s.get("status") in ("completed", "skipped") for s in subtasks)
    any_failed = any(s.get("status") == "failed" for s in subtasks)
    if unfinished:
        blackboard.task_set_status(db, row, "failed")
        row.error = f"Task incomplete: {len(unfinished)} subtasks unfinished"
        db.commit()
        blackboard.task_add_trace(db, row, "coordinator", "task_failed",
                                  row.error, status="failed")
        reward = -0.5
    elif any_failed:
        blackboard.task_set_status(db, row, "failed")
        row.error = "One or more subtasks failed"
        db.commit()
        blackboard.task_add_trace(db, row, "coordinator", "task_failed",
                                  row.error, status="failed")
        reward = -0.3
    else:
        blackboard.task_set_status(db, row, "completed")
        blackboard.task_add_trace(db, row, "coordinator", "task_completed",
                                  "All subtasks completed; report aggregated",
                                  status="completed")
        reward = 1.0 if _agent_ok(final) else 0.5
        success = _agent_ok(final)

    if final.get("ok"):
        blackboard.task_set_final(db, row, final)
    else:
        blackboard.task_set_final(db, row, {"ok": False, "answer": final.get("answer", "Task failed.")})

    # feed the learning signal back to the policy (honest RL bookkeeping)
    policy.finish_episode(reward=reward, success=success,
                          turns=max(1, len(executed)))

    tv = blackboard.task_view(row)
    result = _task_answer(tv, final if isinstance(final, dict) else {})
    _save_assistant(db, session, result, agent="orchestrator")
    return result
