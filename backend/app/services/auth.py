"""Authentication: PBKDF2 password hashing + JWT sessions."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.db import get_db
from app.models.db_models import AuditLog, User, UserProfile, ExpenseCategory

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ---------- password hashing (PBKDF2-HMAC-SHA256) ----------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000)
    return f"pbkdf2${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, dk_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ---------- JWT ----------

def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except Exception:
        raise credentials_exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def audit(db: Session, user_id: int | None, action: str, detail: str = "", ip: str = ""):
    db.add(AuditLog(user_id=user_id, action=action, detail=detail[:2000], ip=ip))
    db.commit()


SYSTEM_CATEGORIES = [
    ("Food & Groceries", "expense", "#f59e0b"),
    ("Transportation", "expense", "#3b82f6"),
    ("Shopping", "expense", "#ec4899"),
    ("Housing & Rent", "expense", "#8b5cf6"),
    ("Utilities", "expense", "#06b6d4"),
    ("Healthcare", "expense", "#ef4444"),
    ("Education", "expense", "#6366f1"),
    ("Entertainment", "expense", "#a855f7"),
    ("Travel", "expense", "#14b8a6"),
    ("Personal Care", "expense", "#f97316"),
    ("Subscriptions", "expense", "#84cc16"),
    ("Other", "expense", "#64748b"),
    ("Salary", "income", "#10b981"),
    ("Freelance", "income", "#22c55e"),
    ("Business", "income", "#059669"),
    ("Interest", "income", "#34d399"),
]


def seed_categories(db: Session, user: User):
    existing = {c.name for c in db.query(ExpenseCategory).filter_by(user_id=user.id)}
    for name, kind, color in SYSTEM_CATEGORIES:
        if name not in existing:
            db.add(ExpenseCategory(user_id=user.id, name=name, kind=kind, color=color, is_system=True))
    db.commit()


def create_user(db: Session, email: str, password: str, full_name: str = "") -> User:
    user = User(email=email.lower().strip(), full_name=full_name.strip(), hashed_password=hash_password(password))
    db.add(user)
    db.flush()
    db.add(UserProfile(user_id=user.id))
    seed_categories(db, user)
    db.commit()
    db.refresh(user)
    return user
