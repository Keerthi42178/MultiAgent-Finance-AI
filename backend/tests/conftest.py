import os
import sys

# Unique SQLite file per pytest session so repeated runs stay isolated
os.environ["DATABASE_URL"] = f"sqlite:///./test_finpilot_{os.getpid()}.db"
os.environ["SECRET_KEY"] = "test-secret"
# keep tests deterministic — never call the real Gemini API
os.environ["GEMINI_API_KEY"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
