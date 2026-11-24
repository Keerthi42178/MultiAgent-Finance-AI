# FinPilot AI

**AI-Powered Personal Finance & Intelligent Budget Management Platform** — a production-style full-stack app combining a premium fintech UI, a conversational multi-agent finance assistant, five specialized agents coordinated through shared blackboard memory, and an Actor-Critic reinforcement-learning policy for dialogue orchestration.

## Features

- **Premium 3D landing, login & register** — glassmorphism, CSS-3D floating coins/cards, Framer Motion
- **Real authentication** — PBKDF2 password hashing, JWT sessions, per-user data isolation, password reset, audit logs
- **Financial onboarding** — income, commitments, living costs, savings goals → provisional budget
- **Dashboard** — income vs expenses, trends, category pie, weekly spend, budget utilization, bills, goals (all real data, Recharts)
- **Expenses** — full CRUD, filters, search, custom categories
- **CSV bank import** — auto column detection, manual mapping, Indian date formats, debit/credit classification, duplicate detection, skip review, import history
- **Income management** — salary/freelance/business/interest, recurring flags
- **Budget planner** — category limits, essential/flexible, AI suggestions, over-budget alerts
- **Bills & subscriptions** — due dates, recurring, overdue detection, mark-paid, notifications
- **Savings goals** — targets, required monthly contribution, progress, milestones
- **AI assistant** — dedicated chat page + floating widget; intent understanding (Gemini w/ heuristic fallback), agent tools, confirmation gates for writes
- **5 agents** — Expense Analysis, Budget Planning, Bill Reminder, Savings Recommendation, Financial Report
- **Actor-Critic RL** — PyTorch actor-critic policy selects next action/agent; rule baseline fallback; metrics persisted
- **Shared blackboard** — per-user structured memory (profile, dialogue state, agent outputs, preferences) + task-scoped shared memory for multi-agent runs
- **Collaborative multi-agent tasks** — complex requests (e.g. "can I afford an ₹8L car?") are decomposed into dependent subtasks; agents consume each other's outputs via the blackboard and the Actor-Critic policy selects the next eligible agent at every step; the Report Agent aggregates one final answer. Execution trace visible in chat + `GET /api/tasks/{key}`
- **Reports** — monthly report, trends, budget-vs-actual, CSV export, print/PDF
- **Notifications center** — bills, budget, goals, imports; mark-read
- **Settings & privacy** — profile, currency, theme, notification prefs, JSON export, account deletion

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + TypeScript + Vite + Tailwind + Recharts + Framer Motion + Lucide |
| Backend | Python + FastAPI + SQLAlchemy |
| Database | SQLite (dev) / PostgreSQL (prod) via `DATABASE_URL` |
| AI | Google Gemini API (optional; deterministic heuristics fallback) |
| RL | PyTorch Actor-Critic (online REINFORCE-style updates) |

## Project layout

```
FinPilotAI/
  backend/
    main.py                  FastAPI entrypoint
    app/
      agents/                5 specialized agents + base
      api/routes.py          REST API (~40 endpoints)
      core/config.py         env-based settings
      database/db.py         engine/session
      integrations/gemini.py LLM client
      memory/blackboard.py   shared agent memory + task-scoped store
      models/db_models.py    all tables (incl. task_executions)
      orchestration/         orchestrator + decomposer + coordinator
      reinforcement_learning/actor_critic.py
      schemas/               pydantic models
      services/              auth, finance calc, CSV import, notifications
    tests/                   pytest API tests
    sample_statement.csv
    requirements.txt
    .env.example
  frontend/
    src/pages/               Landing, Login, Register, Onboarding, Dashboard,
                             Expenses, Income, Budgets, Bills, Goals, Chat,
                             Reports, Notifications, Settings, Agents, Policy
    src/components/          Scene3D, ChatWidget, ui kit
    src/services/api.ts      REST client (JWT)
```

## Quick start (Windows)

```bat
start.bat
```

Or manually — two terminals:

**Backend**
```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     :: edit SECRET_KEY, optionally GEMINI_API_KEY
uvicorn main:app --reload --port 8000
```

**Frontend**
```bat
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the Vite dev server proxies `/api` to `:8000`.

Optional demo account: set `SEED_DEMO=true` in `backend/.env`, restart backend → login with `demo@finpilot.ai` / `demo1234` (all records flagged `is_demo`).

## Tests

```bat
cd backend
pytest tests/ -v
```

## API overview

All endpoints under `/api`, protected by `Authorization: Bearer <jwt>` except `/api/auth/*`.

- `POST /api/auth/register|login|logout|forgot-password|reset-password`, `GET /api/auth/me`
- `GET|PUT /api/profile`, `GET /api/profile/budget-suggestion`
- `GET /api/dashboard`, `GET /api/reports/monthly|trend`, `GET /api/reports/export.csv`
- `GET|POST|PUT|DELETE /api/expenses`, `GET /api/expenses/spending-by-day`
- `GET|POST|PUT|DELETE /api/income`
- `GET|POST|DELETE /api/categories`
- `GET /api/budgets`, `GET /api/budgets/current`, `PUT /api/budgets/{year}/{month}`
- `GET|POST|PUT|DELETE /api/bills`, `POST /api/bills/{id}/pay`
- `GET|POST|PUT|DELETE /api/goals`, `POST /api/goals/{id}/contribute`
- `POST /api/import/preview`, `POST /api/import/confirm`, `GET /api/import/history`
- `GET /api/notifications`, `POST /api/notifications/read`
- `GET|POST|DELETE /api/chat/sessions[/{id}]`, `POST /api/chat/message`
- `GET /api/agents`, `GET /api/policy`, `GET /api/blackboard`
- `GET /api/tasks`, `GET /api/tasks/{task_key}` — multi-agent task list + execution trace
- `GET /api/settings/export`, `DELETE /api/settings/account`, `GET /api/audit-logs`
- `POST /api/demo/seed`

Interactive docs: http://localhost:8000/docs

## Configuration

See `backend/.env.example`. Never commit real secrets. `GEMINI_API_KEY` is optional — without it the assistant uses deterministic intent parsing and templated answers; all calculations remain real.

## Notes & honest limitations

- The Actor-Critic policy learns online from completed dialogues; it is **not** pre-trained on live users and does not auto-train in the background.
- "Training/evaluation vs rule baseline" metrics are exposed via `/api/policy`; a scripted simulation harness can be added under `backend/tests` for offline evaluation.
- Bill reminders are in-app notifications (plus audit-logged events). Email/push delivery requires configuring an external provider — the app does not claim delivery it cannot perform.
- CSV import is statement-file based; no bank credentials are ever requested.
