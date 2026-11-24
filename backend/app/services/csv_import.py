"""Bank statement CSV import: parsing, column detection, mapping, dedupe."""
import csv
import io
import re
from datetime import datetime
from typing import Optional

import dateutil.parser  # type: ignore
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.db_models import Expense, ImportHistory, Income
from app.services.finance import categorize

DATE_HEADERS = {"date", "transaction date", "txn date", "trans date", "value date", "posting date"}
DESC_HEADERS = {"description", "narration", "particulars", "details", "transaction description", "remarks", "memo"}
DEBIT_HEADERS = {"debit", "withdrawal", "dr", "money out", "paid out"}
CREDIT_HEADERS = {"credit", "deposit", "cr", "money in", "paid in"}
AMOUNT_HEADERS = {"amount", "txn amount", "transaction amount", "value"}
BALANCE_HEADERS = {"balance", "closing balance", "available balance", "ledger balance"}


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", h.strip().lower())


def detect_columns(headers: list[str]) -> dict:
    """Auto-map headers -> canonical fields."""
    mapping = {"date": None, "description": None, "debit": None,
               "credit": None, "amount": None, "balance": None}
    for h in headers:
        n = _norm(h)
        if n in DATE_HEADERS and not mapping["date"]:
            mapping["date"] = h
        elif n in DESC_HEADERS and not mapping["description"]:
            mapping["description"] = h
        elif n in DEBIT_HEADERS and not mapping["debit"]:
            mapping["debit"] = h
        elif n in CREDIT_HEADERS and not mapping["credit"]:
            mapping["credit"] = h
        elif n in AMOUNT_HEADERS and not mapping["amount"]:
            mapping["amount"] = h
        elif n in BALANCE_HEADERS and not mapping["balance"]:
            mapping["balance"] = h
    return mapping


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "").replace("₹", "").replace("INR", "").replace("$", "")
    if s in ("", "-", "--", "n/a", "na"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        f = float(s)
        return -f if neg else f
    except ValueError:
        return None


def _to_date(v) -> Optional[datetime]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # Excel serial
    try:
        f = float(s)
        if 20000 < f < 80000:
            return datetime(1899, 12, 30) + __import__("datetime").timedelta(days=f)
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return dateutil.parser.parse(s, dayfirst=True)
    except Exception:
        return None


def fingerprint(date_s: str, desc: str, amount: float) -> str:
    import hashlib
    return hashlib.sha256(f"{date_s}|{desc.strip().lower()}|{amount:.2f}".encode()).hexdigest()


def preview_csv(content: str) -> dict:
    """Return headers, sample rows and auto-detected mapping."""
    sample = content[:200_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(sample), dialect)
    rows = list(reader)
    if not rows:
        return {"headers": [], "rows": [], "mapping": {}}
    headers = [h.strip() for h in rows[0]]
    data = [[c.strip() for c in r] for r in rows[1:] if any(c.strip() for c in r)][:10]
    return {"headers": headers, "rows": data, "mapping": detect_columns(headers)}


def parse_and_import(db: Session, user_id: int, content: str, filename: str,
                     mapping: Optional[dict] = None,
                     corrections: Optional[list[dict]] = None) -> dict:
    """Parse CSV and import classified transactions.

    corrections: optional list of {row_index, classification: 'expense'|'income'|'skip', category}
    applied after auto-classification so users can fix ambiguous rows.
    """
    try:
        dialect = csv.Sniffer().sniff(content[:200_000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(content), dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        return {"error": "CSV has no data rows"}
    headers = [h.strip() for h in rows[0]]
    if not mapping:
        mapping = detect_columns(headers)

    # resolve column indexes
    idx = {}
    for field, name in mapping.items():
        if name:
            try:
                idx[field] = headers.index(name)
            except ValueError:
                idx[field] = None
        else:
            idx[field] = None
    if idx.get("date") is None:
        return {"error": "Could not detect a date column. Please map columns manually."}
    has_amt = idx.get("amount") is not None or idx.get("debit") is not None or idx.get("credit") is not None
    if not has_amt:
        return {"error": "Could not detect an amount/debit/credit column."}

    corr_map = {c.get("row_index"): c for c in (corrections or [])}

    # existing fingerprints for dedupe
    existing = set()
    for (d, desc, amt) in db.query(Expense.date, Expense.description, Expense.amount).filter_by(user_id=user_id):
        existing.add(fingerprint(d.isoformat(), desc or "", float(amt)))
    for (d, desc, amt) in db.query(Income.date, Income.description, Income.amount).filter_by(user_id=user_id):
        existing.add(fingerprint(d.isoformat(), desc or "", float(amt)))

    hist = ImportHistory(user_id=user_id, filename=filename, total_rows=len(rows) - 1)
    db.add(hist)
    db.flush()

    imported = skipped = duplicates = 0
    skipped_reasons = []
    totals = {"expense": 0.0, "income": 0.0}
    pending = []

    for ri, row in enumerate(rows[1:], start=1):
        def cell(field):
            i = idx.get(field)
            if i is None or i >= len(row):
                return None
            return row[i]

        dt = _to_date(cell("date"))
        desc = (cell("description") or "").strip() or "Bank transaction"
        debit = _to_float(cell("debit"))
        credit = _to_float(cell("credit"))
        amount = _to_float(cell("amount"))
        balance = _to_float(cell("balance"))

        # Determine classification
        cls = None
        amt = None
        if debit is not None and debit != 0:
            cls, amt = "expense", abs(debit)
        elif credit is not None and credit != 0:
            cls, amt = "income", credit
        elif amount is not None and amount != 0:
            # signed amount: negative = expense, positive = income (common convention)
            cls, amt = ("expense", abs(amount)) if amount < 0 else ("income", amount)
        else:
            skipped += 1
            skipped_reasons.append({"row": ri, "reason": "No valid debit/credit/amount value"})
            continue

        # balance-only rows are not transactions
        if amt is None or amt == 0:
            skipped += 1
            skipped_reasons.append({"row": ri, "reason": "Zero amount"})
            continue

        if dt is None:
            skipped += 1
            skipped_reasons.append({"row": ri, "reason": "Unparseable date"})
            continue

        # user correction overrides
        corr = corr_map.get(ri)
        if corr:
            c = corr.get("classification")
            if c in ("expense", "income", "skip"):
                cls = c
            if corr.get("category"):
                pass  # applied below

        if cls == "skip":
            skipped += 1
            skipped_reasons.append({"row": ri, "reason": "Skipped by user"})
            continue

        if cls not in ("expense", "income"):
            # ambiguous (e.g. unsigned amount) -> do not silently import
            skipped += 1
            skipped_reasons.append({"row": ri, "reason": "Ambiguous debit/credit - needs review"})
            continue

        fp = fingerprint(dt.date().isoformat(), desc, round(amt, 2))
        if fp in existing:
            duplicates += 1
            continue
        existing.add(fp)

        category = categorize(desc)
        if corr and corr.get("category"):
            category = corr["category"]
        pending.append((cls, amt, dt.date(), desc, category, ri))

    # second pass: insert
    for cls, amt, d, desc, category, ri in pending:
        if cls == "expense":
            db.add(Expense(user_id=user_id, amount=round(amt, 2), date=d,
                           category=category, description=desc[:500], source="csv_import",
                           import_id=hist.id))
            totals["expense"] += amt
        else:
            src = "Salary" if "salary" in desc.lower() else (
                "Interest" if "interest" in desc.lower() else "Other")
            db.add(Income(user_id=user_id, amount=round(amt, 2), date=d,
                          source=src, description=desc[:500], is_recurring=False))
            totals["income"] += amt
        imported += 1

    hist.imported_rows = imported
    hist.skipped_rows = skipped
    hist.duplicate_rows = duplicates
    import json
    hist.summary = json.dumps({
        "totals": {"expense": round(totals["expense"], 2), "income": round(totals["income"], 2)},
        "skipped": skipped_reasons[:100],
    })
    db.commit()
    return {
        "import_id": hist.id, "total_rows": hist.total_rows,
        "imported": imported, "skipped": skipped, "duplicates": duplicates,
        "expense_total": round(totals["expense"], 2),
        "income_total": round(totals["income"], 2),
        "skipped_reasons": skipped_reasons[:50],
    }
