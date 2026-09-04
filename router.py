"""
router.py — Intelligent LLM Cost Governor
Routing Policy: takes classifier output and decides which model tier
to call, plus handles escalation to the next tier when a gate fails.
"""

import config


def get_initial_tier(classification: dict) -> str:
    """
    Decide the starting tier for a query based on its classification.

    Conversational queries always go to the cheapest tier (and skip
    the quality gate entirely — that's handled in pipeline.py, not here).
    All other queries use the locked routing policy from config,
    with the MULTI_PART override potentially bumping to a stronger
    starting tier if the base type would otherwise dispatch too low
    for an already risk-elevated query.
    """
    if classification["is_conversational"]:
        return "economy"

    query_type = classification["query_type"]
    tier = config.DEFAULT_TIER_BY_TYPE[query_type]

    # If MULTI_PART bumped risk to "high" but the default tier for this
    # type is still "economy", bump the starting tier to "standard" too —
    # keeps risk level and starting tier consistent with each other.
    if classification["is_multi_part"] and classification["risk_level"] == "high" and tier == "economy":
        tier = "standard"

    return tier


def get_next_tier(current_tier: str) -> str | None:
    """
    Return the next tier up in the escalation order, or None if
    current_tier is already the highest (Specialized) — meaning there's
    nowhere left to escalate to.
    """
    idx = config.ESCALATION_ORDER.index(current_tier)
    if idx < len(config.ESCALATION_ORDER) - 1:
        return config.ESCALATION_ORDER[idx + 1]
    return None


def get_model_id(tier: str) -> str:
    """Return the actual Groq model ID string for a given tier name."""
    return config.TIERS[tier]