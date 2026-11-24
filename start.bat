@echo off
REM FinPilot AI - one-click launcher (Windows)
setlocal
cd /d "%~dp0"

echo === FinPilot AI ===

if not exist backend\.venv (
  echo [1/4] Creating backend virtualenv...
  cd backend
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -r requirements.txt
  cd ..
) else (
  echo [1/4] Backend venv exists.
)

if not exist backend\.env (
  echo [2/4] Creating backend\.env from example...
  copy backend\.env.example backend\.env >nul
  echo      Edit backend\.env and set SECRET_KEY - and GEMINI_API_KEY if used.
)

if not exist frontend\node_modules (
  echo [3/4] Installing frontend dependencies...
  cd frontend
  call npm install
  cd ..
) else (
  echo [3/4] Frontend dependencies installed.
)

echo [4/4] Starting backend on :8000 and frontend on :5173...
start "FinPilot Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\activate.bat && python -m uvicorn main:app --reload --port 8000"
timeout /t 3 /nobreak >nul
start "FinPilot Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
start "" http://localhost:5173
echo.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:5173
echo.
endlocal
