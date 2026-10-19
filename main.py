import os
import re
import sqlite3
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "scans.db")
HTML_PATH = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="SafeScan AI Full-Stack API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load AI Model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file 'phishing_model.pkl' not found! Run 'py train_model.py' first.")
model = joblib.load(MODEL_PATH)

# 2. Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            verdict TEXT NOT NULL,
            risk_score REAL NOT NULL,
            triggers TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class ScanRequest(BaseModel):
    content: str

# Serve Frontend Dashboard
@app.get("/")
def serve_frontend():
    return FileResponse(HTML_PATH)

# POST: Scan Message with AI + Rule Signals and Store in DB
@app.post("/api/scan")
def scan_content(request: ScanRequest):
    raw_text = request.content.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    cleaned = raw_text.lower()

    # AI Model Base Probability
    probabilities = model.predict_proba([cleaned])[0]
    risk_score = float(probabilities[1]) * 100

    # Extract security risk indicators (explainable AI signals)
    triggers = []
    if re.search(r"http://|bit\.ly|t\.me|tinyurl", cleaned):
        risk_score += 25.0
        triggers.append("Suspicious/Unsecured Link")
    if re.search(r"\b(urgent|immediately|suspended|disconnected|blocked|verify|kyc|otp|cvv)\b", cleaned):
        risk_score += 18.0
        triggers.append("Urgency or Credential Request")
    if re.search(r"\b(won|prize|lottery|reward|refund|earn rs)\b", cleaned):
        risk_score += 18.0
        triggers.append("Financial Lure / Prize Bait")

    risk_score = round(min(99.5, risk_score), 1)
    verdict = "High Risk (Phishing)" if risk_score >= 50.0 else "Safe"
    trigger_text = ", ".join(triggers) if triggers else "Standard Text Pattern"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save into SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scan_history (content, verdict, risk_score, triggers, scanned_at) VALUES (?, ?, ?, ?, ?)",
        (raw_text, verdict, risk_score, trigger_text, timestamp)
    )
    conn.commit()
    conn.close()

    return {
        "content": raw_text,
        "verdict": verdict,
        "risk_score": risk_score,
        "triggers": trigger_text,
        "scanned_at": timestamp
    }

# GET: Fetch History + Dashboard Summary Stats
@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, content, verdict, risk_score, triggers, scanned_at FROM scan_history ORDER BY id DESC LIMIT 15")
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN risk_score >= 50 THEN 1 ELSE 0 END) FROM scan_history")
    total_scans, threats_count = cursor.fetchone()
    conn.close()

    total_scans = total_scans or 0
    threats_count = threats_count or 0
    safe_count = total_scans - threats_count

    return {
        "stats": {
            "total": total_scans,
            "threats": threats_count,
            "safe": safe_count
        },
        "history": [
            {
                "id": r[0],
                "content": r[1],
                "verdict": r[2],
                "risk_score": r[3],
                "triggers": r[4],
                "scanned_at": r[5]
            }
            for r in rows
        ]
    }

# DELETE: Clear Database History
@app.delete("/api/history")
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history")
    conn.commit()
    conn.close()
    return {"message": "History cleared"}