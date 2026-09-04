"""
config.py — Intelligent LLM Cost Governor
Central configuration: model tiers, pricing, logging behavior, API key loading.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------------------------------------------------------
# API KEY
# ----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Make sure you have a .env file with "
        "GROQ_API_KEY=your_key_here in the project root."
    )

# ----------------------------------------------------------------------------
# MODEL TIERS
# Capability ladder: Economy -> Standard -> Specialized
# Cost climbs as capability climbs (verified against Groq's published pricing).
# ----------------------------------------------------------------------------
TIERS = {
    "economy": "openai/gpt-oss-20b",
    "standard": "openai/gpt-oss-120b",
    "specialized": "qwen/qwen3.6-27b",
}

# Escalation order — what to try next if a tier fails the quality gate
ESCALATION_ORDER = ["economy", "standard", "specialized"]

# ----------------------------------------------------------------------------
# PRICING ($ per 1,000,000 tokens) — Groq published rates
# Used for cost tracking and the dashboard's cost-saved-vs-baseline metric.
# Baseline definition: cost if every query had hit the Specialized tier.
# Escalation cost accounting: count REAL cost incurred (economy+specialized
# when escalation happens), not just the final tier's cost.
# ----------------------------------------------------------------------------
PRICING = {
    "economy": {"input": 0.075, "output": 0.30},
    "standard": {"input": 0.15, "output": 0.60},
    "specialized": {"input": 0.60, "output": 3.00},
}

# ----------------------------------------------------------------------------
# PRIVACY / LOGGING
# Metadata is always logged. Raw query text is only logged if this flag is
# explicitly set to True — meant for local debugging only, never in a
# deployed/hosted instance.
# ----------------------------------------------------------------------------
LOG_RAW_TEXT = False

# ----------------------------------------------------------------------------
# STORAGE
# ----------------------------------------------------------------------------
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "logs.db")

# ----------------------------------------------------------------------------
# QUERY TYPES (8 types + MULTI_PART override tag)
# Single-label classification — a query gets exactly one base type,
# optionally flagged with the MULTI_PART override.
# ----------------------------------------------------------------------------
QUERY_TYPES = [
    "SHORT_FACTUAL",
    "EXPLANATION",
    "COMPARISON",
    "CODE_GEN",
    "REASONING_MATH",
    "SUMMARY",
    "CREATIVE",
    "INSTRUCTION_HOWTO",
]

MULTI_PART_OVERRIDE = "MULTI_PART"

CONVERSATIONAL = "CONVERSATIONAL"  # bypass tag — not a real type, handled pre-classifier

# ----------------------------------------------------------------------------
# ROUTING POLICY — which tier each query type starts at
# Conversational bypasses this entirely (handled separately, always "economy").
# ----------------------------------------------------------------------------
DEFAULT_TIER_BY_TYPE = {
    "SHORT_FACTUAL": "economy",
    "EXPLANATION": "economy",
    "COMPARISON": "standard",
    "CODE_GEN": "standard",
    "REASONING_MATH": "standard",
    "SUMMARY": "economy",
    "CREATIVE": "economy",
    "INSTRUCTION_HOWTO": "economy",
}

# ----------------------------------------------------------------------------
# QUALITY RISK LEVELS — detectability of failure, not query difficulty
# Used by the quality gate to decide how strict Layer 3 checks should be.
# ----------------------------------------------------------------------------
RISK_LEVEL_BY_TYPE = {
    "SHORT_FACTUAL": "low",
    "EXPLANATION": "medium",
    "COMPARISON": "high",
    "CODE_GEN": "high",
    "REASONING_MATH": "high",
    "SUMMARY": "medium",
    "CREATIVE": "low",       # Layer 1 only — documented limitation, no depth check
    "INSTRUCTION_HOWTO": "medium",
}

# MULTI_PART override bumps risk one level — handled in router.py logic,
# not hardcoded here since it depends on the base type's current level.
RISK_ESCALATION_ORDER = ["low", "medium", "high"]