"""
logger.py — Intelligent LLM Cost Governor
Handles all logging to SQLite. Metadata is always logged; raw query text
is only stored if config.LOG_RAW_TEXT is True.
"""

import sqlite3
import os
import time
import json

import config


def _get_connection():
    """Create the data directory if needed and return a SQLite connection."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def init_db():
    """Create the logs table if it doesn't already exist. Call this once at startup."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query_type TEXT NOT NULL,
            is_multi_part INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            initial_tier TEXT NOT NULL,
            final_tier TEXT NOT NULL,
            escalated INTEGER NOT NULL,
            escalation_reason TEXT,
            gate_pass INTEGER NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_incurred REAL,
            baseline_cost REAL,
            latency_seconds REAL,
            raw_query_text TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(
    query_type,
    is_multi_part,
    risk_level,
    initial_tier,
    final_tier,
    escalated,
    gate_pass,
    input_tokens,
    output_tokens,
    cost_incurred,
    baseline_cost,
    latency_seconds,
    escalation_reason=None,
    raw_query_text=None,
):
    """
    Log a single query's full journey through the pipeline.

    Raw query text is only persisted if config.LOG_RAW_TEXT is True.
    All other fields (metadata) are always logged regardless of that flag.
    """
    conn = _get_connection()
    cursor = conn.cursor()

    text_to_store = raw_query_text if config.LOG_RAW_TEXT else None

    cursor.execute("""
        INSERT INTO query_logs (
            timestamp, query_type, is_multi_part, risk_level,
            initial_tier, final_tier, escalated, escalation_reason,
            gate_pass, input_tokens, output_tokens, cost_incurred,
            baseline_cost, latency_seconds, raw_query_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        time.time(),
        query_type,
        int(is_multi_part),
        risk_level,
        initial_tier,
        final_tier,
        int(escalated),
        escalation_reason,
        int(gate_pass),
        input_tokens,
        output_tokens,
        cost_incurred,
        baseline_cost,
        latency_seconds,
        text_to_store,
    ))
    conn.commit()
    conn.close()


def get_all_logs():
    """Return all logged queries as a list of dicts, most recent first."""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_summary_stats():
    """
    Return aggregate stats for the dashboard:
    total queries, escalation rate, total cost, total baseline cost,
    cost saved, and per-type gate pass/fail counts.
    """
    logs = get_all_logs()

    if not logs:
        return {
            "total_queries": 0,
            "escalation_rate": 0.0,
            "total_cost": 0.0,
            "total_baseline_cost": 0.0,
            "cost_saved": 0.0,
            "tier_distribution": {},
            "per_type_pass_fail": {},
        }

    total_queries = len(logs)
    escalated_count = sum(1 for l in logs if l["escalated"])
    total_cost = sum(l["cost_incurred"] or 0 for l in logs)
    total_baseline_cost = sum(l["baseline_cost"] or 0 for l in logs)

    tier_distribution = {}
    for l in logs:
        tier = l["final_tier"]
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

    per_type_pass_fail = {}
    for l in logs:
        qtype = l["query_type"]
        if qtype not in per_type_pass_fail:
            per_type_pass_fail[qtype] = {"pass": 0, "fail": 0}
        if l["gate_pass"]:
            per_type_pass_fail[qtype]["pass"] += 1
        else:
            per_type_pass_fail[qtype]["fail"] += 1

    return {
        "total_queries": total_queries,
        "escalation_rate": escalated_count / total_queries if total_queries else 0.0,
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline_cost,
        "cost_saved": total_baseline_cost - total_cost,
        "tier_distribution": tier_distribution,
        "per_type_pass_fail": per_type_pass_fail,
    }