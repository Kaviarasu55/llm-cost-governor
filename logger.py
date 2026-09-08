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
    """
    Create the logs table if it doesn't already exist, and migrate it
    forward if it already exists in the pre-v3 shape (no risk_score /
    gate_status columns). SQLite's ALTER TABLE ADD COLUMN is safe here
    since both new columns are nullable — old rows just get NULL for
    them, and get_summary_stats() falls back to gate_pass for those.
    """
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
            gate_status TEXT,
            risk_score INTEGER,
            gate_reason TEXT,
            gate_signals TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_incurred REAL,
            baseline_cost REAL,
            latency_seconds REAL,
            raw_query_text TEXT
        )
    """)

    # Migration path for a pre-v3/pre-v4 database that already has the
    # table but not the newer columns.
    cursor.execute("PRAGMA table_info(query_logs)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "gate_status" not in existing_columns:
        cursor.execute("ALTER TABLE query_logs ADD COLUMN gate_status TEXT")
    if "risk_score" not in existing_columns:
        cursor.execute("ALTER TABLE query_logs ADD COLUMN risk_score INTEGER")
    if "gate_reason" not in existing_columns:
        cursor.execute("ALTER TABLE query_logs ADD COLUMN gate_reason TEXT")
    if "gate_signals" not in existing_columns:
        cursor.execute("ALTER TABLE query_logs ADD COLUMN gate_signals TEXT")

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
    gate_status=None,
    risk_score=None,
    gate_reason=None,
    gate_signals=None,
):
    """
    Log a single query's full journey through the pipeline.

    gate_status is one of "PASS" / "REVIEW" / "FAIL" (from the v3
    weighted quality gate), or "BYPASSED" for conversational queries
    that never went through the gate at all, or "ERROR" for API
    failures where no response was ever gated. risk_score is the raw
    0+ point total behind that status.

    gate_reason is the full human-readable string from
    quality_gate.evaluate() (e.g. "REVIEW (score=25): instruction_no_steps
    (+25): no step-like structure found") — this used to only get
    captured when a query escalated (buried inside escalation_reason);
    PASS and REVIEW rows never had it logged at all, which made the v4
    dashboard's per-row inspection impossible for exactly the rows the
    calibration pass most needs to look at. Now logged for every row
    that went through the real gate.

    gate_signals is that same evaluation's "signals" list, JSON-encoded
    (list of {"name", "points", "reason"} dicts) — one row per
    triggered check, for building a structured breakdown view rather
    than parsing the reason string.

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
            gate_pass, gate_status, risk_score, gate_reason, gate_signals,
            input_tokens, output_tokens,
            cost_incurred, baseline_cost, latency_seconds, raw_query_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        gate_status,
        risk_score,
        gate_reason,
        gate_signals,
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
    cost saved, per-type gate pass/fail counts (legacy, boolean-based),
    and per-type PASS/REVIEW/FAIL counts (v3, status-based).

    For rows logged before v3 (gate_status is NULL — no risk-score gate
    ever ran on them, or they predate this column), status falls back
    to "PASS" or "FAIL" based on the old boolean gate_pass, since those
    rows never had a REVIEW option. This keeps old and new logs
    comparable in the same table without needing a hard data migration.
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
            "avg_latency_by_tier": {},
            "per_type_pass_fail": {},
            "per_type_status": {},
            "review_count": 0,
            "review_rate": 0.0,
        }

    total_queries = len(logs)
    escalated_count = sum(1 for l in logs if l["escalated"])
    total_cost = sum(l["cost_incurred"] or 0 for l in logs)
    total_baseline_cost = sum(l["baseline_cost"] or 0 for l in logs)

    tier_distribution = {}
    latency_by_tier = {}
    for l in logs:
        tier = l["final_tier"]
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        if l["latency_seconds"] is not None:
            latency_by_tier.setdefault(tier, []).append(l["latency_seconds"])

    avg_latency_by_tier = {
        tier: sum(values) / len(values) for tier, values in latency_by_tier.items()
    }

    per_type_pass_fail = {}
    per_type_status = {}
    review_count = 0

    for l in logs:
        qtype = l["query_type"]

        if qtype not in per_type_pass_fail:
            per_type_pass_fail[qtype] = {"pass": 0, "fail": 0}
        if l["gate_pass"]:
            per_type_pass_fail[qtype]["pass"] += 1
        else:
            per_type_pass_fail[qtype]["fail"] += 1

        status = l.get("gate_status")
        if not status:
            # Legacy row, or a status the gate itself doesn't set
            # (BYPASSED/ERROR are set explicitly by pipeline.py and
            # will already be non-empty here).
            status = "PASS" if l["gate_pass"] else "FAIL"

        if qtype not in per_type_status:
            per_type_status[qtype] = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "BYPASSED": 0, "ERROR": 0}
        per_type_status[qtype][status] = per_type_status[qtype].get(status, 0) + 1

        if status == "REVIEW":
            review_count += 1

    return {
        "total_queries": total_queries,
        "escalation_rate": escalated_count / total_queries if total_queries else 0.0,
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline_cost,
        "cost_saved": total_baseline_cost - total_cost,
        "tier_distribution": tier_distribution,
        "avg_latency_by_tier": avg_latency_by_tier,
        "per_type_pass_fail": per_type_pass_fail,
        "per_type_status": per_type_status,
        "review_count": review_count,
        "review_rate": review_count / total_queries if total_queries else 0.0,
    }