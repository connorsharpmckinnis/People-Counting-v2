import sqlite3
import json
import uuid
import os
from datetime import datetime

DB_PATH = "jobs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            result TEXT,
            error TEXT,
            filename TEXT,
            config TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_job(job_type: str, filename: str, config: dict) -> str:
    job_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO jobs (id, type, status, created_at, filename, config) VALUES (?, ?, ?, ?, ?, ?)",
        (
            job_id,
            job_type,
            "queued",
            datetime.now().isoformat(),
            filename,
            json.dumps(config)
        )
    )
    conn.commit()
    conn.close()
    return job_id

def get_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    updates = ["status = ?", "updated_at = ?"]
    params = [status, datetime.now().isoformat()]
    
    if result is not None:
        updates.append("result = ?")
        params.append(json.dumps(result))
    
    if error is not None:
        updates.append("error = ?")
        params.append(error)
        
    params.append(job_id)
    
    query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
    c.execute(query, params)
    conn.commit()
    conn.close()
