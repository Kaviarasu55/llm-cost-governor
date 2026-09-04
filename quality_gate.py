"""
quality_gate.py — Intelligent LLM Cost Governor
The core differentiator of this project: transparent, inspectable
pass/fail decisions on model responses.

Three layers:
  Layer 1 — universal checks (applies to every response, every type)
  Layer 2 — type-specific checks (per the locked taxonomy)
  Layer 3 — risk-based strictness (stricter thresholds as risk climbs)

IMPORTANT DOCUMENTED LIMITATION: this gate measures a heuristic
failure-risk signal — structural completeness, presence of expected
elements — NOT ground-truth correctness. A structurally complete but
factually wrong answer can still pass. This is intentional MVP scope,
not an oversight, and should be stated plainly in the README.
"""

import re

import config


# ----------------------------------------------------------------------------
# LAYER 1 — UNIVERSAL CHECKS
# Applied to every single response, regardless of type.
# ----------------------------------------------------------------------------
def layer1_check(response_text: str) -> tuple[bool, str]:
    """
    Universal sanity checks every response must pass.
    Returns (passed, reason).
    """
    if not response_text or not response_text.strip():
        return False, "Layer 1: response is empty"

    stripped = response_text.strip()

    if len(stripped) < 10:
        return False, "Layer 1: response is too short to be substantive"

    refusal_markers = [
        "i cannot help with that",
        "i can't help with that",
        "i'm not able to",
        "as an ai language model",
        "i don't have access to",
    ]
    lowered = stripped.lower()
    if any(marker in lowered for marker in refusal_markers):
        return False, "Layer 1: response appears to be a refusal/non-answer"

    return True, "Layer 1: passed"


# ----------------------------------------------------------------------------
# LAYER 2 — TYPE-SPECIFIC CHECKS
# ----------------------------------------------------------------------------

def _extract_comparison_items(query: str) -> list[str]:
    """Best-effort extraction of the items being compared from the query."""
    q = query.lower()

    # Each item capped at 4 words. Previously an unbounded lazy match
    # ([\w\s]+?) with no punctuation before end-of-string would swallow
    # everything to the end of the query — e.g. "...a queue in terms of
    # their underlying use cases and typical performance characteristics"
    # got captured as a single "item", which then could never match any
    # real answer and silently failed the gate at every tier. Bounding
    # the item length keeps extraction to plausible noun phrases.
    item = r"(?:\w+\s+){0,3}\w+"

    # "X vs Y" / "X vs. Y" / "X versus Y"
    match = re.search(rf"({item})\s+(?:vs\.?|versus)\s+({item})(?:[\?\.,]|$)", q)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # "difference between X and Y" / "compare X and Y"
    match = re.search(rf"(?:difference between|compare)\s+({item})\s+and\s+({item})(?:[\?\.,]|$)", q)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return []


def _check_comparison(query: str, response_text: str) -> tuple[bool, str]:
    items = _extract_comparison_items(query)
    if not items:
        # Couldn't confidently extract items — don't fail on something
        # we can't verify; fall through as a pass with a noted caveat.
        return True, "Layer 2 (COMPARISON): could not extract items to verify, skipped"

    lowered_response = response_text.lower()
    missing = [item for item in items if item and item not in lowered_response]

    if missing:
        return False, f"Layer 2 (COMPARISON): response doesn't address: {', '.join(missing)}"
    return True, "Layer 2 (COMPARISON): all compared items addressed"


def _check_code_gen(response_text: str) -> tuple[bool, str]:
    has_code_block = "```" in response_text
    has_code_keywords = bool(re.search(r"\b(def|function|class|import|return|const|let|var)\b", response_text))

    if not (has_code_block or has_code_keywords):
        return False, "Layer 2 (CODE_GEN): no code block or code-like content found"

    # Basic syntax sanity: balanced brackets/parens/braces within any code block
    code_blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", response_text, re.DOTALL)
    for block in code_blocks:
        for open_ch, close_ch in [("(", ")"), ("[", "]"), ("{", "}")]:
            if block.count(open_ch) != block.count(close_ch):
                return False, f"Layer 2 (CODE_GEN): unbalanced '{open_ch}{close_ch}' in code block — basic syntax sanity failed"

    return True, "Layer 2 (CODE_GEN): code present and passes basic syntax sanity"


def _check_reasoning_math(response_text: str) -> tuple[bool, str]:
    # Look for something that reads like a final result: "= number",
    # "answer is X", "result is X", or the response ending in a number.
    has_final_result = bool(re.search(
        r"(=\s*-?\d+(\.\d+)?|answer is|answer:|result is|result:|therefore,?\s*-?\d+)",
        response_text, re.IGNORECASE
    ))
    if not has_final_result:
        return False, "Layer 2 (REASONING_MATH): no final result/answer found, may just restate the problem"
    return True, "Layer 2 (REASONING_MATH): final result present"


def _check_summary(query: str, response_text: str) -> tuple[bool, str]:
    # Meaningful compression as a signal, not an absolute rule.
    # A short input with a short valid summary must still pass.
    if len(query.split()) <= 15:
        return True, "Layer 2 (SUMMARY): input already short, compression check skipped"

    if len(response_text.split()) >= len(query.split()):
        return False, "Layer 2 (SUMMARY): response is not meaningfully shorter than the input"
    return True, "Layer 2 (SUMMARY): response is shorter than input"


def _check_instruction_howto(response_text: str) -> tuple[bool, str]:
    has_numbered_steps = bool(re.search(r"(?:^|\n)\s*\d+[\.\)]\s", response_text))
    has_sequence_words = bool(re.search(
        r"\b(first,|first step|second,|next,|then,|finally,|step 1|step one)\b",
        response_text, re.IGNORECASE
    ))
    if not (has_numbered_steps or has_sequence_words):
        return False, "Layer 2 (INSTRUCTION_HOWTO): no step-like structure found"
    return True, "Layer 2 (INSTRUCTION_HOWTO): step-like structure present"


def _check_explanation(response_text: str) -> tuple[bool, str]:
    # Gate = presence of real explanatory content, NOT sentence count.
    # A correct 1-sentence answer must pass — so we check substance
    # (length above a low floor + not just a restatement), not structure.
    word_count = len(response_text.split())
    if word_count < 5:
        return False, "Layer 2 (EXPLANATION): response too brief to contain real explanatory content"
    return True, "Layer 2 (EXPLANATION): contains explanatory content"


def layer2_check(query: str, response_text: str, query_type: str) -> tuple[bool, str]:
    """Dispatch to the correct type-specific check. CREATIVE and
    SHORT_FACTUAL have no Layer 2 check by design (documented limitation
    for CREATIVE; SHORT_FACTUAL relies on Layer 1 only per locked taxonomy)."""
    if query_type == "COMPARISON":
        return _check_comparison(query, response_text)
    if query_type == "CODE_GEN":
        return _check_code_gen(response_text)
    if query_type == "REASONING_MATH":
        return _check_reasoning_math(response_text)
    if query_type == "SUMMARY":
        return _check_summary(query, response_text)
    if query_type == "INSTRUCTION_HOWTO":
        return _check_instruction_howto(response_text)
    if query_type == "EXPLANATION":
        return _check_explanation(response_text)
    if query_type in ("CREATIVE", "SHORT_FACTUAL"):
        return True, f"Layer 2 ({query_type}): no type-specific check by design"

    return True, f"Layer 2 ({query_type}): no check defined, passed by default"


# ----------------------------------------------------------------------------
# LAYER 3 — RISK-BASED STRICTNESS
# Stricter thresholds as risk climbs. Also handles the MULTI_PART
# override check: each sub-question must have a corresponding
# answered part.
# ----------------------------------------------------------------------------
def _check_multi_part_coverage(query: str, response_text: str) -> tuple[bool, str]:
    """
    Rough coverage check for MULTI_PART queries: count sub-questions
    in the query and check the response has a comparable number of
    distinct addressed segments (paragraphs or numbered points).
    This is a heuristic, not a guarantee of true coverage.
    """
    sub_question_count = query.count("?")
    if sub_question_count < 2:
        sub_question_count = len(re.findall(r"(?:^|\s)\(?[1-9a-dA-D][\).]\s", query))

    if sub_question_count < 2:
        return True, "Layer 3 (MULTI_PART): could not confidently count sub-questions, skipped"

    # Count response segments: paragraphs or numbered list items
    response_segments = max(
        len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", response_text)),
        len([p for p in response_text.split("\n\n") if p.strip()]),
    )

    if response_segments < sub_question_count:
        return False, (
            f"Layer 3 (MULTI_PART): query has ~{sub_question_count} sub-parts, "
            f"response only has ~{response_segments} distinct answered segments"
        )
    return True, "Layer 3 (MULTI_PART): sub-question coverage looks adequate"


def layer3_check(query: str, response_text: str, risk_level: str, is_multi_part: bool) -> tuple[bool, str]:
    """
    Risk-based strictness layer. Higher risk = stricter minimum length
    requirement, on top of whatever Layer 1/2 already checked.
    MULTI_PART coverage check also runs here regardless of risk level.
    """
    if is_multi_part:
        passed, reason = _check_multi_part_coverage(query, response_text)
        if not passed:
            return False, reason

    min_length_by_risk = {
        "low": 0,
        "medium": 15,
        "high": 30,
    }
    min_words = min_length_by_risk.get(risk_level, 0)
    word_count = len(response_text.split())

    if word_count < min_words:
        return False, (
            f"Layer 3 (risk={risk_level}): response has {word_count} words, "
            f"below the {min_words}-word minimum for this risk level"
        )

    return True, f"Layer 3 (risk={risk_level}): strictness check passed"


# ----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------
def evaluate(query: str, response_text: str, query_type: str, risk_level: str, is_multi_part: bool) -> dict:
    """
    Run a response through all applicable gate layers in order.
    Stops at the first failure — the reason tells you exactly which
    layer and why, which is the whole point of this project's
    transparency differentiator.

    Returns:
        {
            "passed": bool,
            "failed_layer": str or None,
            "reason": str
        }
    """
    passed, reason = layer1_check(response_text)
    if not passed:
        return {"passed": False, "failed_layer": "layer1", "reason": reason}

    passed, reason = layer2_check(query, response_text, query_type)
    if not passed:
        return {"passed": False, "failed_layer": "layer2", "reason": reason}

    passed, reason = layer3_check(query, response_text, risk_level, is_multi_part)
    if not passed:
        return {"passed": False, "failed_layer": "layer3", "reason": reason}

    return {"passed": True, "failed_layer": None, "reason": "All applicable layers passed"}