import time
from sqlalchemy.orm import Session

from app.models.db_models import AgentExecutionHistory, User


class BaseAgent:
    name = "base"
    description = ""

    def __init__(self, db: Session):
        self.db = db

    def run(self, user: User, intent: dict, state: dict, **kwargs) -> dict:
        raise NotImplementedError

    def _record(self, user: User, session_id, action: str, inp: str, out: dict,
                success: bool, latency: float):
        try:
            self.db.add(AgentExecutionHistory(
                user_id=user.id, session_id=session_id, agent=self.name,
                action=action, input_summary=str(inp)[:500],
                output_summary=str(out)[:1500], success=success, latency_ms=latency))
            self.db.commit()
        except Exception:
            self.db.rollback()

    def execute(self, user: User, intent: dict, state: dict, session_id=None, **kwargs) -> dict:
        t0 = time.time()
        try:
            result = self.run(user, intent, state, **kwargs)
            result.setdefault("agent", self.name)
            self._record(user, session_id, intent.get("action", self.name),
                         intent.get("raw", ""), result, True, (time.time() - t0) * 1000)
            return result
        except Exception as e:
            out = {"agent": self.name, "error": str(e), "ok": False}
            self._record(user, session_id, "error", intent.get("raw", ""), out, False,
                         (time.time() - t0) * 1000)
            return out
