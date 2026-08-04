"""
Authentication & History Storage Module
SQLite-based user authentication and JSON prediction history.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'users.db')
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'history')

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register(username, password, full_name=""):
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            (username.strip().lower(), hash_password(password), full_name.strip())
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def login(username, password):
    conn = _get_db()
    cursor = conn.execute(
        "SELECT id, full_name FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password))
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return True, {"id": row[0], "username": username.strip().lower(), "full_name": row[1]}
    return False, "Invalid username or password."


def save_prediction(username, prediction_data):
    """Save a prediction result to user's history."""
    history_file = os.path.join(HISTORY_DIR, f"{username}.json")
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)

    prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_data['id'] = len(history) + 1
    history.append(prediction_data)

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def get_history(username):
    """Get all prediction history for a user."""
    history_file = os.path.join(HISTORY_DIR, f"{username}.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


def get_history_stats(username):
    """Get summary stats from user's prediction history."""
    history = get_history(username)
    if not history:
        return {
            'total': 0, 'healthy': 0, 'parkinsons': 0,
            'high_risk': 0, 'medium_risk': 0, 'low_risk': 0,
            'avg_confidence': 0
        }

    total = len(history)
    healthy = sum(1 for h in history if h.get('diagnosis') == 'Healthy')
    parkinsons = total - healthy
    high = sum(1 for h in history if h.get('risk_level') == 'High')
    medium = sum(1 for h in history if h.get('risk_level') == 'Medium')
    low = sum(1 for h in history if h.get('risk_level') == 'Low')
    avg_conf = sum(h.get('confidence', 0) for h in history) / total if total > 0 else 0

    return {
        'total': total, 'healthy': healthy, 'parkinsons': parkinsons,
        'high_risk': high, 'medium_risk': medium, 'low_risk': low,
        'avg_confidence': round(avg_conf, 1)
    }
