"""In-app notification helpers."""
import json
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.models.db_models import Notification, UserProfile
from app.services.finance import upcoming_bills, budget_status, goals_progress, month_bounds


def notify(db: Session, user_id: int, kind: str, title: str, body: str = ""):
    n = Notification(user_id=user_id, kind=kind, title=title, body=body)
    db.add(n)
    return n


def generate_due_notifications(db: Session, user_id: int) -> int:
    """Create notifications for bills due soon / overdue / budget overruns / goal milestones."""
    created = 0
    today = date.today()
    # avoid duplicates: check existing unread with same title today
    def exists(title):
        return db.query(Notification).filter(
            Notification.user_id == user_id, Notification.title == title,
            Notification.is_read == False,  # noqa: E712
            Notification.created_at >= today).first() is not None

    for b in upcoming_bills(db, user_id, days=7):
        label = "Overdue" if b["status"] == "overdue" else "Due soon"
        title = f"{label}: {b['name']} ₹{b['amount']:,.0f} on {b['due_date']}"
        if not exists(title):
            notify(db, user_id, "bill", title, f"Your {b['name']} bill of ₹{b['amount']:,.0f} is {label.lower()}.")
            created += 1

    y, m = today.year, today.month
    bs = budget_status(db, user_id, y, m)
    if bs:
        for c in bs["categories"]:
            if c["over"] and not exists(f"Budget exceeded: {c['category']}"):
                notify(db, user_id, "budget",
                       f"Budget exceeded: {c['category']}",
                       f"Spent ₹{c['spent']:,.0f} of ₹{c['limit']:,.0f} limit.")
                created += 1

    for g in goals_progress(db, user_id):
        if g["percent"] >= 100 and not exists(f"Goal achieved: {g['name']}"):
            notify(db, user_id, "goal", f"Goal achieved: {g['name']}",
                   f"You reached ₹{g['target']:,.0f}.")
            created += 1
        elif g["percent"] >= 75 and not exists(f"Goal milestone: {g['name']}"):
            notify(db, user_id, "goal", f"Goal milestone: {g['name']}",
                   f"{g['percent']}% complete.")
            created += 1

    if created:
        db.commit()
    return created


def list_notifications(db: Session, user_id: int, unread_only=False, limit=100):
    q = db.query(Notification).filter_by(user_id=user_id).order_by(Notification.created_at.desc())
    if unread_only:
        q = q.filter(Notification.is_read == False)  # noqa: E712
    return q.limit(limit).all()


def mark_read(db: Session, user_id: int, notification_id: int | None = None):
    q = db.query(Notification).filter_by(user_id=user_id, is_read=False)
    if notification_id:
        q = q.filter(Notification.id == notification_id)
    n = q.update({"is_read": True})
    db.commit()
    return n


def delete_notification(db: Session, user_id: int, notification_id: int) -> bool:
    n = db.query(Notification).filter_by(id=notification_id, user_id=user_id).first()
    if not n:
        return False
    db.delete(n)
    db.commit()
    return True


def notification_prefs(db: Session, user_id: int) -> dict:
    p = db.query(UserProfile).filter_by(user_id=user_id).first()
    if not p or not p.notification_prefs:
        return {"bills": True, "budget": True, "goals": True, "reports": True}
    try:
        return json.loads(p.notification_prefs)
    except Exception:
        return {}
