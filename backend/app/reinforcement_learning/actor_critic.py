"""Actor-Critic policy for dialogue-action and agent selection.

The Actor picks the next dialogue action / target agent; the Critic estimates
the value of the current dialogue state. Trained online with REINFORCE-style
updates from simulated conversations; a rule-based baseline is always kept
for comparison and as a safety fallback.
"""
import json
import logging
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("finpilot.rl")

# Dialogue-level actions the policy can select.
ACTIONS = [
    "expense",        # -> Expense Analysis Agent
    "budget",         # -> Budget Planning Agent
    "bill",           # -> Bill Reminder Agent
    "savings",        # -> Savings Recommendation Agent
    "report",         # -> Financial Report Agent
    "ask_user",       # ask a clarifying question
    "confirm",        # request explicit user confirmation
    "complete",       # finish the turn with an answer
]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)
STATE_DIM = 18

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class ActorNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_state(intent: dict, state: dict, dialogue: dict, turn: int = 0) -> torch.Tensor:
    """Fixed-length numeric representation of the dialogue/financial state."""
    summary = state.get("summary", {}) or {}
    profile = state.get("profile", {}) or {}
    budget = state.get("budget") or {}
    goals = state.get("goals", []) or []
    bills = state.get("bills_upcoming", []) or []

    has_income = 1.0 if (summary.get("total_income") or profile.get("monthly_income")) else 0.0
    has_expenses = 1.0 if summary.get("transaction_count") else 0.0
    has_budget = 1.0 if budget else 0.0
    budget_over = 1.0 if any(c.get("over") for c in budget.get("categories", [])) else 0.0
    has_goals = 1.0 if goals else 0.0
    has_bills = 1.0 if bills else 0.0
    missing = len(dialogue.get("missing_slots", []) or [])
    confirmed = 1.0 if dialogue.get("pending_confirmation") else 0.0
    amount = min(1.0, (intent.get("amount") or 0) / 100000.0)
    inc_amount = min(1.0, (summary.get("total_income") or 0) / 200000.0)
    exp_ratio = min(1.5, (summary.get("total_expenses") or 0) / max(1.0, summary.get("total_income") or profile.get("monthly_income") or 1))
    savings_rate = min(1.0, max(0.0, (summary.get("savings_rate") or 0) / 50.0))
    turns = min(1.0, turn / 10.0)
    intent_kind = {
        "spending_query": 0.1, "income_query": 0.15, "why_spending": 0.2,
        "budget_query": 0.3, "bill_query": 0.4, "goal_query": 0.5,
        "add_expense": 0.7, "add_income": 0.7, "add_bill": 0.75,
        "add_goal": 0.8, "savings_plan": 0.55, "affordability": 0.6,
        "reduce_spending": 0.35, "report": 0.25, "smalltalk": 0.0,
        "other": 0.1,
    }.get(intent.get("intent", "other"), 0.1)
    needs_conf = 1.0 if intent.get("needs_confirmation") else 0.0

    vec = [
        has_income, has_expenses, has_budget, budget_over, has_goals, has_bills,
        min(1.0, missing / 4.0), confirmed, amount, inc_amount, exp_ratio,
        savings_rate, turns, intent_kind, needs_conf,
        1.0 if state.get("month") else 0.0,
        min(1.0, len(goals) / 5.0), min(1.0, len(bills) / 10.0),
    ]
    return torch.tensor(vec[:STATE_DIM], dtype=torch.float32)


def encode_task_state(task: dict, board_state: dict) -> torch.Tensor:
    """18-dim state vector for multi-agent task execution.

    Same dimensionality as encode_state (checkpoint-compatible) but encodes
    task progress: completed/pending subtasks, dependency readiness, prior
    agent outputs, and the primary goal type.
    """
    subtasks = task.get("subtasks", []) or []
    results = task.get("results", {}) or {}
    total = max(1, len(subtasks))
    done = sum(1 for s in subtasks if s.get("status") == "completed")
    failed = sum(1 for s in subtasks if s.get("status") == "failed")
    ready = sum(1 for s in subtasks
                if s.get("status") in ("pending", "ready") and
                all(results.get(d, {}).get("ok", True) for d in (s.get("dependencies") or [])))
    progress = done / total
    summary = board_state.get("summary", {}) or {}
    profile = board_state.get("profile", {}) or {}
    has_income = 1.0 if (summary.get("total_income") or profile.get("monthly_income")) else 0.0
    has_expenses = 1.0 if summary.get("transaction_count") else 0.0
    goal_kind = {
        "purchase_affordability": 0.2, "savings_plan": 0.4, "debt_management": 0.6,
        "goal_prioritization": 0.8, "budget_replan": 0.5, "trip_savings": 0.3,
    }.get(task.get("primary_goal", ""), 0.1)
    vec = [
        has_income, has_expenses, progress, min(1.0, ready / total),
        min(1.0, failed / total), min(1.0, done / 5.0),
        min(1.0, len(results) / 5.0), 1.0,  # task active
        min(1.0, (task.get("stated", {}) or {}).get("price", 0) / 100000.0),
        min(1.0, (summary.get("total_income") or 0) / 200000.0),
        min(1.5, (summary.get("total_expenses") or 0) / max(1.0, summary.get("total_income") or 1)),
        min(1.0, max(0.0, (summary.get("savings_rate") or 0) / 50.0)),
        min(1.0, len(task.get("trace", []) or []) / 20.0),
        goal_kind, 0.0,
        1.0 if board_state.get("month") else 0.0,
        min(1.0, len(board_state.get("goals", []) or []) / 5.0),
        min(1.0, len(board_state.get("bills_upcoming", []) or []) / 8.0),
    ]
    return torch.tensor(vec[:STATE_DIM], dtype=torch.float32)


class AgentPolicy:
    """Actor-Critic controller. Falls back to rule-based selection when untrained."""

    # intent -> preferred agent (rule baseline)
    RULE_MAP = {
        "spending_query": "expense", "why_spending": "expense", "reduce_spending": "expense",
        "income_query": "report", "budget_query": "budget", "bill_query": "bill",
        "goal_query": "savings", "savings_plan": "savings", "save_money": "savings",
        "report": "report", "export_report": "report",
        "add_expense": "expense", "add_income": "report", "add_bill": "bill",
        "add_goal": "savings", "update_income": "budget", "affordability": "savings",
        "trip_plan": "savings", "smalltalk": "complete", "other": "expense",
    }

    def __init__(self, lr: float = 1e-3, gamma: float = 0.95):
        self.actor = ActorNet()
        self.critic = CriticNet()
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.episodes = 0
        self.metrics = {"tasks": 0, "successes": 0, "turns_total": 0}
        self.history: list[dict] = []
        self._pending: list[dict] = []
        self._load()

    # ---------- inference ----------
    def decide(self, intent: dict, state: dict, dialogue: dict, turn: int = 0) -> dict:
        state_vec = encode_state(intent, state, dialogue, turn)
        with torch.no_grad():
            probs = self.actor(state_vec)
            value = self.critic(state_vec)
        probs_list = probs.tolist()
        allowed = self._allowed(intent, dialogue)
        masked = [p if ACTIONS[i] in allowed else 0.0 for i, p in enumerate(probs_list)]
        total = sum(masked)
        if total <= 0:
            masked = [1.0 if ACTIONS[i] in allowed else 0.0 for i in range(N_ACTIONS)]
            total = sum(masked)
        masked = [p / total for p in masked]

        # rule baseline action
        rule_action = self.RULE_MAP.get(intent.get("intent", "other"), "expense")
        if rule_action not in allowed:
            rule_action = allowed[0] if allowed else "complete"

        # While untrained, follow the rule baseline deterministically so users
        # always get the right agent; the policy still records probs for learning.
        if self.episodes < 10:
            chosen = rule_action
        else:
            # blend policy with rule baseline
            blend = 0.85
            for i, a in enumerate(ACTIONS):
                if a == rule_action:
                    masked[i] = masked[i] * (1 - blend) + blend
            total = sum(masked)
            masked = [p / total for p in masked]

            r = random.random() * sum(masked)
            acc = 0.0
            chosen = "complete"
            for i, p in enumerate(masked):
                acc += p
                if r <= acc and ACTIONS[i] in allowed:
                    chosen = ACTIONS[i]
                    break
        idx = ACTION_INDEX[chosen]
        log_prob = math.log(max(masked[idx], 1e-9))
        decision = {
            "state": [round(x, 4) for x in state_vec.tolist()],
            "probs": [round(p, 4) for p in probs_list],
            "masked_probs": [round(p, 4) for p in masked],
            "action": chosen,
            "critic_value": round(float(value.item()), 4),
            "rule_action": rule_action,
            "allowed": allowed,
        }
        self._pending.append({"state_vec": state_vec, "action_index": idx,
                              "log_prob": log_prob})
        self.history.append(decision)
        self.history = self.history[-200:]
        return decision

    def decide_task_step(self, task: dict, board_state: dict,
                         eligible_agents: list[str]) -> dict:
        """Select the next agent for a decomposed task.

        eligible_agents: agent names whose subtasks are currently READY
        (dependencies satisfied). "complete" is only eligible when every
        subtask finished or the task is otherwise done.
        """
        state_vec = encode_task_state(task, board_state)
        with torch.no_grad():
            probs = self.actor(state_vec)
            value = self.critic(state_vec)
        probs_list = probs.tolist()

        subtasks = task.get("subtasks", []) or []
        all_done = all(s.get("status") in ("completed", "failed", "skipped")
                       for s in subtasks) if subtasks else True
        allowed = list(eligible_agents)
        if all_done:
            allowed = allowed + ["complete"]
        if not allowed:
            allowed = ["complete"]

        masked = [p if ACTIONS[i] in allowed else 0.0 for i, p in enumerate(probs_list)]
        total = sum(masked)
        if total <= 0:
            masked = [1.0 if ACTIONS[i] in allowed else 0.0 for i in range(N_ACTIONS)]
            total = sum(masked)
        masked = [p / total for p in masked]

        # rule baseline: first ready subtask's assigned agent (dependency order)
        rule_action = "complete"
        for s in subtasks:
            if s.get("status") in ("pending", "ready") and s.get("assigned_agent") in allowed:
                rule_action = s["assigned_agent"]
                break
        if rule_action not in allowed:
            rule_action = allowed[0]

        if self.episodes < 10:
            chosen = rule_action
        else:
            blend = 0.85
            for i, a in enumerate(ACTIONS):
                if a == rule_action:
                    masked[i] = masked[i] * (1 - blend) + blend
            total = sum(masked)
            masked = [p / total for p in masked]
            r = random.random() * sum(masked)
            acc = 0.0
            chosen = allowed[0]
            for i, p in enumerate(masked):
                acc += p
                if r <= acc and ACTIONS[i] in allowed:
                    chosen = ACTIONS[i]
                    break

        idx = ACTION_INDEX[chosen]
        log_prob = math.log(max(masked[idx], 1e-9))
        decision = {
            "state": [round(x, 4) for x in state_vec.tolist()],
            "probs": [round(p, 4) for p in probs_list],
            "masked_probs": [round(p, 4) for p in masked],
            "action": chosen,
            "critic_value": round(float(value.item()), 4),
            "rule_action": rule_action,
            "allowed": allowed,
            "task_key": task.get("task_key"),
        }
        self._pending.append({"state_vec": state_vec, "action_index": idx,
                              "log_prob": log_prob})
        self.history.append(decision)
        self.history = self.history[-200:]
        return decision

    def _allowed(self, intent: dict, dialogue: dict) -> list[str]:
        """Legal actions given dialogue state."""
        allowed = list(ACTIONS)
        # if awaiting confirmation, only confirm/ask_user/complete make sense
        if dialogue.get("pending_confirmation"):
            return ["confirm", "ask_user", "complete"]
        # write intents must confirm before mutating
        if intent.get("needs_confirmation") and intent.get("intent") in (
                "add_expense", "add_income", "add_bill", "add_goal", "update_income"):
            allowed = ["confirm", "ask_user"] + [a for a in allowed
                   if a in ("expense", "budget", "bill", "savings")]
        return allowed

    # ---------- learning ----------
    def finish_episode(self, reward: float, success: bool, turns: int):
        """Apply policy-gradient update for the completed dialogue."""
        self.metrics["tasks"] += 1
        self.metrics["turns_total"] += turns
        if success:
            self.metrics["successes"] += 1
        self.episodes += 1
        if not self._pending:
            return
        # discounted returns
        G = reward
        returns = []
        for _ in self._pending:
            returns.append(G)
            G *= self.gamma
        returns.reverse()
        loss = 0.0
        for step, ret in zip(self._pending, returns):
            x = step["state_vec"]
            h1 = F.relu(self.actor.net[0](x))
            h2 = F.relu(self.actor.net[2](h1))
            logits = self.actor.net[4](h2)
            logp = F.log_softmax(logits, dim=-1)[step["action_index"]]
            value = self.critic(step["state_vec"])
            advantage = ret - value.detach()
            loss = loss - logp * advantage + 0.5 * (ret - value) ** 2
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
        self.opt.step()
        self._pending = []
        if self.episodes % 10 == 0:
            self._save()

    # ---------- persistence ----------
    def _ckpt(self):
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        return CHECKPOINT_DIR / "policy.pt"

    def _load(self):
        p = self._ckpt()
        if p.exists():
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                self.actor.load_state_dict(data["actor"])
                self.critic.load_state_dict(data["critic"])
                self.episodes = data.get("episodes", 0)
                self.metrics = data.get("metrics", self.metrics)
            except Exception as e:
                logger.warning("Could not load policy checkpoint: %s", e)

    def _save(self):
        try:
            torch.save({"actor": self.actor.state_dict(),
                        "critic": self.critic.state_dict(),
                        "episodes": self.episodes, "metrics": self.metrics}, self._ckpt())
        except Exception as e:
            logger.warning("Could not save policy checkpoint: %s", e)

    def status(self) -> dict:
        return {
            "actions": ACTIONS,
            "state_dim": STATE_DIM,
            "episodes_trained": self.episodes,
            "metrics": self.metrics,
            "recent_decisions": self.history[-20:],
            "checkpoint": str(self._ckpt()),
        }


# module-level singleton
_policy: AgentPolicy | None = None


def get_policy() -> AgentPolicy:
    global _policy
    if _policy is None:
        _policy = AgentPolicy()
    return _policy
