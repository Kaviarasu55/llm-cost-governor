"""
classifier.py — Intelligent LLM Cost Governor
Heuristic-based query classification: detects conversational bypass,
one of 8 query types, and the MULTI_PART override tag.

This is deliberately rule-based (not ML) for MVP scope — fast, free,
deterministic, and inspectable, which fits the project's core
differentiator (transparent, explainable decisions). Documented
limitation: heuristics can misclassify edge cases; this is not a
trained classifier.
"""

import re

import config


# ----------------------------------------------------------------------------
# CONVERSATIONAL BYPASS
# ----------------------------------------------------------------------------
CONVERSATIONAL_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|sup)\s*,?\s*(there|everyone|all|friend|buddy|mate|folks|guys|team)?\s*[!.,]*\s*$",
    r"^\s*(thanks|thank you|thx|ty|cheers)\s*,?\s*(a lot|so much|very much|a bunch|a ton)?\s*[!.,]*\s*$",
    r"^\s*(bye|goodbye|see ya|later|see you)\s*,?\s*(later|soon|tomorrow|for now)?\s*[!.,]*\s*$",
    r"^\s*(how are you|what's up|whats up)\s*[?!.,]*\s*$",
    r"^\s*(good morning|good evening|good night|good afternoon)\s*,?\s*(everyone|all|there|folks|team)?\s*[!.,]*\s*$",
    r"^\s*(ok|okay|cool|nice|great|awesome)\s*[!.,]*\s*$",
]


def is_conversational(query: str) -> bool:
    """Check if a query is pure conversational filler (greeting/thanks/etc)."""
    q = query.strip().lower()
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


# ----------------------------------------------------------------------------
# MULTI_PART DETECTION
# ----------------------------------------------------------------------------
def is_multi_part(query: str) -> bool:
    """
    Detect whether a query contains multiple distinct sub-questions.
    Heuristics: multiple question marks, numbered/lettered sub-parts,
    or explicit "and also" / "also," style joins between question-like clauses.
    """
    question_mark_count = query.count("?")
    if question_mark_count >= 2:
        return True

    # Numbered or lettered list markers, e.g. "1)" "2." "a)" within the text
    if re.search(r"(?:^|\s)\(?[1-9a-dA-D][\).]\s", query):
        return True

    # Explicit multi-part connector phrases between question-like segments
    connector_pattern = r"\?\s*(and|also|additionally|plus)\b"
    if re.search(connector_pattern, query, re.IGNORECASE):
        return True

    return False


# ----------------------------------------------------------------------------
# QUERY TYPE CLASSIFICATION
# Each check is a simple keyword/pattern heuristic. Order matters — more
# specific/high-signal checks run first so they aren't shadowed by generic
# ones like EXPLANATION.
# ----------------------------------------------------------------------------

def _is_code_gen(q: str) -> bool:
    keywords = [
        "write code", "implement", "code for",
        "python script", "javascript", "function that", "class that",
        "fix this code", "debug this", "regex for",
        "sql query", "```",
    ]
    if any(k in q for k in keywords):
        return True
    # Generalized "write [me] a/an [1-2 descriptor words] function/program/class/script"
    # Catches "write a Python function", "write a Java class", etc. — cases the
    # old literal "write a function"/"write a class"/"write a program" keywords
    # missed whenever a language or adjective sat between the article and noun.
    if re.search(
        r"\bwrite\s+(?:me\s+)?(?:a|an)\s+(?:\w+\s+){0,2}(?:function|program|script|class)\b",
        q,
    ):
        return True
    return False


def _is_comparison(q: str) -> bool:
    keywords = [
        " vs ", " vs. ", "versus", "compare", "comparison",
        "difference between", "better than", "which is better",
        "pros and cons",
    ]
    return any(k in q for k in keywords)


def _is_reasoning_math(q: str) -> bool:
    keywords = [
        "calculate", "solve", "what is the value of", "how many",
        "prove that", "derive", "equation", "sum of", "product of",
        "if x =", "algebra", "geometry", "probability of",
        "chance of", "odds of", "likelihood of",
    ]
    has_keyword = any(k in q for k in keywords)
    has_math_symbols = bool(re.search(r"\d+\s*[\+\-\*/\^=]\s*\d+", q))
    return has_keyword or has_math_symbols


def _is_summary(q: str) -> bool:
    keywords = [
        "summarize", "summarise", "summary of", "tl;dr", "tldr",
        "give me a brief", "condense", "in short",
    ]
    return any(k in q for k in keywords)


def _is_creative(q: str) -> bool:
    keywords = [
        "creative writing", "compose a", "fictional", "imagine a scenario",
        "write lyrics", "write a script for a",
    ]
    if any(k in q for k in keywords):
        return True
    # Generalized "write [me] a/an [1-2 descriptor words] poem/story/song/haiku/novel"
    # Same fix as CODE_GEN: catches "write a creative poem", "write me a story", etc.
    if re.search(
        r"\bwrite\s+(?:me\s+)?(?:a|an)\s+(?:\w+\s+){0,2}(?:poem|story|song|haiku|novel)\b",
        q,
    ):
        return True
    return False


def _is_instruction_howto(q: str) -> bool:
    keywords = [
        "how do i", "how to", "steps to", "guide me", "walk me through",
        "tutorial", "instructions for", "step by step",
    ]
    return any(k in q for k in keywords)


def _is_short_factual(q: str) -> bool:
    keywords = [
        "what is", "who is", "when did", "where is", "what year",
        "how old", "capital of", "define ",
    ]
    is_short = len(q.split()) <= 12
    return is_short and any(k in q for k in keywords)


def classify_query(query: str):
    """
    Classify a query into one of the 8 base types.

    Returns:
        (query_type: str, multi_part: bool)

    query_type is always one of config.QUERY_TYPES.
    If no specific heuristic matches, defaults to EXPLANATION —
    this is intentional: EXPLANATION is designed to absorb
    ambiguous/underspecified queries per the locked taxonomy.
    """
    q = query.strip().lower()
    multi_part = is_multi_part(query)

    # Order: most specific/high-signal checks first
    if _is_code_gen(q):
        return "CODE_GEN", multi_part
    if _is_comparison(q):
        return "COMPARISON", multi_part
    if _is_reasoning_math(q):
        return "REASONING_MATH", multi_part
    if _is_summary(q):
        return "SUMMARY", multi_part
    if _is_creative(q):
        return "CREATIVE", multi_part
    if _is_instruction_howto(q):
        return "INSTRUCTION_HOWTO", multi_part
    if _is_short_factual(q):
        return "SHORT_FACTUAL", multi_part

    # Default fallback — absorbs ambiguous/underspecified queries
    # and subjective/opinion queries, per locked taxonomy.
    return "EXPLANATION", multi_part


def classify(query: str):
    """
    Full pre-classifier + classifier entry point.

    Returns:
        dict with keys: is_conversational, query_type, is_multi_part, risk_level

    If is_conversational is True, query_type/is_multi_part/risk_level are None
    since conversational queries bypass the analyzer entirely.
    """
    if is_conversational(query):
        return {
            "is_conversational": True,
            "query_type": None,
            "is_multi_part": None,
            "risk_level": None,
        }

    query_type, multi_part = classify_query(query)
    risk_level = config.RISK_LEVEL_BY_TYPE[query_type]

    # MULTI_PART override bumps risk one level
    if multi_part:
        idx = config.RISK_ESCALATION_ORDER.index(risk_level)
        if idx < len(config.RISK_ESCALATION_ORDER) - 1:
            risk_level = config.RISK_ESCALATION_ORDER[idx + 1]

    return {
        "is_conversational": False,
        "query_type": query_type,
        "is_multi_part": multi_part,
        "risk_level": risk_level,
    }