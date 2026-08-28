import sqlite3
import os
from datetime import datetime

# 1. Force a brand new file name to avoid ALL schema conflicts
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "clinical_logs_v3.db")

def init_db():
    """Creates the database and table with the latency_seconds column."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age TEXT,
            sex TEXT,
            blinded_diagnoses TEXT,
            adjusted_diagnoses TEXT,
            arbiter_verdict TEXT,
            human_decision TEXT,
            latency_seconds REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_decision(age, sex, blinded, adjusted, verdict, decision, latency):
    """Inserts a new clinical log into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_logs (
            timestamp, age, sex, blinded_diagnoses, adjusted_diagnoses, 
            arbiter_verdict, human_decision, latency_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), age, sex, blinded, adjusted, verdict, decision, latency))
    conn.commit()
    conn.close()

def fetch_all_logs():
    """Retrieves all logs for the Streamlit dashboard."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows