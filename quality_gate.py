"""
quality_gate.py — Intelligent LLM Cost Governor (v3)
The core differentiator of this project: transparent, inspectable
risk-scoring decisions on model responses.

v3 change from v2: bool pass/fail -> weighted risk score.
Same 3-layer structure as before (universal / type-specific / risk-based),
but every check now *contributes points* to a running score instead of
short-circuiting the whole evaluation. Final score maps to one of three
statuses:

    0-19   PASS    — accept the response, nothing logged as a concern
    20-49  REVIEW  — accept the response (does NOT trigger escalation),
                     but flagged in the log for manual review later
    50+    FAIL    — reject, triggers escalation to the next tier

IMPORTANT DOCUMENTED LIMITATION (unchanged from v2): this gate measures
a heuristic failure-risk signal — structural completeness, presence of
expected elements — NOT ground-truth correctness. A structurally
complete but factually wrong answer can still score 0.

WEIGHT PROVENANCE — read before tuning:
  The six weights below came directly from the v3 plan and are treated
  as fixed inputs, not something I invented:
      empty_response          50
      refusal                 40
      missing_code_block      30
      comparison_missing_item 25
      below_min_word_count    15
      truncated               20
  Every other signal (reasoning_no_final_result, instruction_no_steps,
  summary_not_compressed, explanation_too_brief, multi_part_incomplete,
  code_syntax_unbalanced) existed as a v2 bool check but was never
  assigned a point value in the plan. I estimated those by matching
  them to the closest given signal in severity (e.g. "no final result"
  in REASONING_MATH is the math-equivalent of a missing code block, so
  it got the same 30). These six are PROVISIONAL and are exactly what
  the planned 15-20-example calibration pass should correct — see
  SIGNAL_WEIGHTS below, each has an inline note on which are locked
  vs. provisional.

KNOWN EDGE CASE FOUND LIVE (2026-09-06, via the user's own sanity-check
run — quality_gate.evaluate("what is 2+2", "4", "SHORT_FACTUAL", "low",
False)): the universal near-empty check (<10 chars) auto-failed a
CORRECT answer, because it doesn't know about query type. SHORT_FACTUAL
is specifically the type this system is supposed to allow terse valid
answers for (it's "low" risk with a 0-word floor everywhere else) —
this check contradicted that. This bug was already present in v2
(same 10-char floor, same missing type-awareness); v3 made it worse by
turning it into a guaranteed auto-FAIL. Fixed by exempting
SHORT_FACTUAL from the too_short signal specifically — every other
check (refusal, layer 2, min-word-count) still runs normally for it,
they just naturally contribute 0 for a correct short answer since
SHORT_FACTUAL has no type-specific check and a 0-word floor.

OTHER EDGE CASES FOUND WHILE TESTING (flagged, not silently resolved
— these are judgment calls about your given weights, not arithmetic
bugs):
  - A refusal (+40) on a LOW risk-level type never accumulates the
    below_min_word_count signal (low risk = 0-word floor), so it scores
    exactly 40 -> REVIEW, not FAIL. On MEDIUM/HIGH risk this resolves
    itself naturally since refusals are short. Whether a refusal should
    always auto-FAIL regardless of type is a call for your calibration
    pass, not something I should decide by inflating your given weight.
  - missing_code_block (30) and reasoning_no_final_result (30) alone
    don't reach FAIL unless paired with a short-response signal — a
    verbose CODE_GEN answer with zero actual code could land in REVIEW
    instead of escalating. Same reasoning: flagged, not changed.
"""

import re
from collections import namedtuple

Signal = namedtuple("Signal", ["name", "points", "reason"])

# ----------------------------------------------------------------------------
# SIGNAL WEIGHTS
# ----------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    # --- locked: given directly in the v3 plan ---
    "empty_response": 50,
    "refusal": 40,
    "missing_code_block": 30,
    "comparison_missing_item": 25,
    "below_min_word_count": 15,
    "truncated": 20,
    # --- provisional: estimated by analogy, needs calibration ---
    "too_short": 50,              # near-empty (<10 chars) — treated same as empty_response;
                                   # e.g. "Idk." conveys effectively zero value regardless of type
    "reasoning_no_final_result": 30,   # analogue of missing_code_block
    "instruction_no_steps": 25,        # analogue of comparison_missing_item
    "summary_not_compressed": 20,      # analogue of truncated (non-conformant, not broken)
    "explanation_too_brief": 15,       # analogue of below_min_word_count
    "multi_part_incomplete": 35,       # scaled up: usually means >1 sub-question dropped
    "code_syntax_unbalanced": 20,      # code exists but is broken, less severe than missing entirely
}

REVIEW_THRESHOLD = 20
FAIL_THRESHOLD = 50

REFUSAL_MARKERS = [
    "i cannot help with that",
    "i can't help with that",
    "i'm not able to",
    "as an ai language model",
    "i don't have access to",
]


# ----------------------------------------------------------------------------
# LAYER 1 — UNIVERSAL CHECKS
# ----------------------------------------------------------------------------
def _check_refusal(stripped_text: str) -> Signal | None:
    lowered = stripped_text.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return Signal("refusal", SIGNAL_WEIGHTS["refusal"],
                       "response appears to be a refusal/non-answer")
    return None


def _check_truncated(stripped_text: str) -> Signal | None:
    """
    Heuristic: a response that ends mid-thought (dangling connector word,
    trailing comma/colon, no closing punctuation) after being long enough
    for truncation to plausibly matter. Deliberately conservative — short
    complete answers with no terminal punctuation are common and should
    NOT be flagged, so this only fires on longer responses.
    """
    words = stripped_text.split()
    if len(words) < 20:
        return None

    if stripped_text.endswith("```"):
        return None
    if stripped_text[-1] in '.!?"\'`)]}':
        return None

    dangling_words = {
        "and", "or", "but", "because", "so", "the", "a", "an", "to", "of",
        "with", "is", "are", "in", "for", "as", "that", "which",
    }
    last_word = re.sub(r"[^\w]", "", words[-1].lower())
    ends_dangling = stripped_text[-1] in ",:" or last_word in dangling_words

    if ends_dangling:
        return Signal("truncated", SIGNAL_WEIGHTS["truncated"],
                       "response appears cut off mid-thought (ends on a dangling word/connector)")
    return None


# ----------------------------------------------------------------------------
# LAYER 2 — TYPE-SPECIFIC CHECKS
# ----------------------------------------------------------------------------
_COMPARISON_LEADIN_RE = re.compile(
    r"^(?:please\s+)?(?:what(?:'s| is) the difference between|difference between|differences between|compare)\s+",
    re.IGNORECASE,
)


def _extract_comparison_items(query: str) -> list[str]:
    """
    Best-effort extraction of the items being compared from the query.

    BUG FOUND WHILE TESTING v3 (inherited from v2, not new): the item
    regex's {0,3} word cap doesn't stop it from swallowing a leading
    verb like "compare" when the query is phrased "Compare X vs Y" —
    e.g. "Compare TCP vs UDP" extracted item[0] as "compare tcp"
    instead of "tcp", so a response that correctly discussed TCP still
    failed the gate because it never contained the literal substring
    "compare tcp". Fixed by stripping the known lead-in phrase before
    running the vs/versus match, so the item regex only ever sees the
    actual noun phrase.
    """
    original_q = query.strip().lower()
    item = r"(?:\w+\s+){0,3}\w+"

    # vs/versus path: strip the lead-in first so "compare" doesn't get
    # captured as part of the item.
    stripped_q = _COMPARISON_LEADIN_RE.sub("", original_q)
    match = re.search(rf"({item})\s+(?:vs\.?|versus)\s+({item})(?:[\?\.,]|$)", stripped_q)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # "difference between X and Y" / "compare X and Y" path: the cue
    # phrase is part of the pattern itself, so match against the
    # original (un-stripped) query.
    match = re.search(rf"(?:difference between|compare)\s+({item})\s+and\s+({item})(?:[\?\.,]|$)", original_q)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return []


def _check_comparison(query: str, response_text: str) -> Signal | None:
    items = _extract_comparison_items(query)
    if not items:
        return None  # can't confidently verify — don't penalize what we can't check

    lowered_response = response_text.lower()
    missing = [item for item in items if item and item not in lowered_response]

    if missing:
        return Signal("comparison_missing_item", SIGNAL_WEIGHTS["comparison_missing_item"],
                       f"response doesn't address: {', '.join(missing)}")
    return None


def _check_code_gen(response_text: str) -> Signal | None:
    has_code_block = "```" in response_text
    has_code_keywords = bool(re.search(r"\b(def|function|class|import|return|const|let|var)\b", response_text))

    if not (has_code_block or has_code_keywords):
        return Signal("missing_code_block", SIGNAL_WEIGHTS["missing_code_block"],
                       "no code block or code-like content found")

    code_blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", response_text, re.DOTALL)
    for block in code_blocks:
        for open_ch, close_ch in [("(", ")"), ("[", "]"), ("{", "}")]:
            if block.count(open_ch) != block.count(close_ch):
                return Signal("code_syntax_unbalanced", SIGNAL_WEIGHTS["code_syntax_unbalanced"],
                              f"unbalanced '{open_ch}{close_ch}' in code block")
    return None


def _check_reasoning_math(response_text: str) -> Signal | None:
    has_final_result = bool(re.search(
        r"(=\s*-?\d+(\.\d+)?|answer is|answer:|result is|result:|therefore,?\s*-?\d+)",
        response_text, re.IGNORECASE
    ))
    if not has_final_result:
        return Signal("reasoning_no_final_result", SIGNAL_WEIGHTS["reasoning_no_final_result"],
                       "no final result/answer found, may just restate the problem")
    return None


def _check_summary(query: str, response_text: str) -> Signal | None:
    if len(query.split()) <= 15:
        return None  # input already short, compression check not meaningful

    if len(response_text.split()) >= len(query.split()):
        return Signal("summary_not_compressed", SIGNAL_WEIGHTS["summary_not_compressed"],
                       "response is not meaningfully shorter than the input")
    return None


def _check_instruction_howto(response_text: str) -> Signal | None:
    has_numbered_steps = bool(re.search(r"(?:^|\n)\s*\d+[\.\)]\s", response_text))
    # BUG FOUND WHILE TESTING v3 (inherited from v2, not new): the
    # trailing \b after alternatives ending in a comma (e.g. "first,")
    # never matches, because \b requires a word/non-word transition and
    # both the comma and the space after it are non-word characters.
    # "First, place the egg..." — exactly the phrasing this check exists
    # to catch — silently failed to match and fell through to the
    # numbered-list check only. Fixed by dropping the trailing \b so the
    # comma/space-terminated alternatives can match.
    has_sequence_words = bool(re.search(
        r"\b(first,|first step|second,|next,|then,|finally,|step 1|step one)",
        response_text, re.IGNORECASE
    ))
    if not (has_numbered_steps or has_sequence_words):
        return Signal("instruction_no_steps", SIGNAL_WEIGHTS["instruction_no_steps"],
                       "no step-like structure found")
    return None


def _check_explanation(response_text: str) -> Signal | None:
    word_count = len(response_text.split())
    if word_count < 5:
        return Signal("explanation_too_brief", SIGNAL_WEIGHTS["explanation_too_brief"],
                       "response too brief to contain real explanatory content")
    return None


def _layer2_signal(query: str, response_text: str, query_type: str) -> Signal | None:
    """CREATIVE and SHORT_FACTUAL have no Layer 2 check by design (same
    documented limitation as v2)."""
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
    return None  # CREATIVE, SHORT_FACTUAL, or unrecognized type


# ----------------------------------------------------------------------------
# LAYER 3 — RISK-BASED STRICTNESS + MULTI_PART COVERAGE
# ----------------------------------------------------------------------------
def _check_multi_part_coverage(query: str, response_text: str) -> Signal | None:
    sub_question_count = query.count("?")
    if sub_question_count < 2:
        sub_question_count = len(re.findall(r"(?:^|\s)\(?[1-9a-dA-D][\).]\s", query))

    if sub_question_count < 2:
        return None  # couldn't confidently count sub-questions

    # BUG FOUND WHILE TESTING v3/v4 LIVE (2026-09-06): a real multi-part
    # query ("capital of Italy? ...Spain? ...Portugal?") got a fully
    # correct response — one fact per line, separated by single '\n',
    # e.g. "The capital of Italy is Rome.\nThe capital of Spain is
    # Madrid.\n..." — but this only checked for numbered-list markers
    # ("1. ") or double-newline ("\n\n") paragraph breaks. Neither
    # matches single-newline-per-fact formatting, which is extremely
    # common for exactly this kind of short multi-part answer, so it
    # counted response_segments = 1 and flagged a complete answer as
    # incomplete. Fixed by also counting non-empty single-newline lines
    # as a third signal, taking the max across all three.
    response_segments = max(
        len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", response_text)),
        len([p for p in response_text.split("\n\n") if p.strip()]),
        len([line for line in response_text.split("\n") if line.strip()]),
    )

    if response_segments < sub_question_count:
        return Signal("multi_part_incomplete", SIGNAL_WEIGHTS["multi_part_incomplete"],
                       f"query has ~{sub_question_count} sub-parts, response only has "
                       f"~{response_segments} distinct answered segments")
    return None


def _check_min_word_count(response_text: str, risk_level: str) -> Signal | None:
    min_length_by_risk = {"low": 0, "medium": 15, "high": 30}
    min_words = min_length_by_risk.get(risk_level, 0)
    word_count = len(response_text.split())

    if word_count < min_words:
        return Signal("below_min_word_count", SIGNAL_WEIGHTS["below_min_word_count"],
                       f"{word_count} words, below the {min_words}-word minimum for risk={risk_level}")
    return None


# ----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------
def evaluate(query: str, response_text: str, query_type: str, risk_level: str, is_multi_part: bool) -> dict:
    """
    Run a response through all applicable checks, accumulating a risk
    score instead of stopping at the first failure. Every triggered
    signal is kept, so the log/dashboard can show the full picture of
    what went wrong, not just the first thing.

    Returns:
        {
            "passed": bool,          # False only on FAIL — kept for
                                      # pipeline.py's existing escalation
                                      # check (PASS and REVIEW both count
                                      # as passed=True; only FAIL escalates)
            "status": "PASS" | "REVIEW" | "FAIL",
            "risk_score": int,
            "signals": [{"name": str, "points": int, "reason": str}, ...],
            "reason": str,           # human-readable summary
            "failed_layer": None,    # kept as a key for backward compat;
                                      # layers no longer short-circuit
                                      # independently so this is always None
        }
    """
    response_text = response_text or ""
    stripped = response_text.strip()
    signals: list[Signal] = []

    if not stripped:
        signals.append(Signal("empty_response", SIGNAL_WEIGHTS["empty_response"], "response is empty"))
    elif len(stripped) < 10 and query_type != "SHORT_FACTUAL":
        # SHORT_FACTUAL is exempt: this is the one type explicitly meant
        # to have terse, valid one-word/one-number answers (e.g. "4",
        # "Paris"). Every other check still runs normally below for it —
        # this exemption only skips the blunt length-based signal.
        signals.append(Signal("too_short", SIGNAL_WEIGHTS["too_short"],
                               "response is near-empty / not substantive"))
    else:
        refusal_sig = _check_refusal(stripped)
        if refusal_sig:
            signals.append(refusal_sig)

        truncated_sig = _check_truncated(stripped)
        if truncated_sig:
            signals.append(truncated_sig)

        type_sig = _layer2_signal(query, response_text, query_type)
        if type_sig:
            signals.append(type_sig)

        if is_multi_part:
            mp_sig = _check_multi_part_coverage(query, response_text)
            if mp_sig:
                signals.append(mp_sig)

        wc_sig = _check_min_word_count(response_text, risk_level)
        if wc_sig:
            signals.append(wc_sig)

    total_score = sum(s.points for s in signals)

    if total_score >= FAIL_THRESHOLD:
        status = "FAIL"
    elif total_score >= REVIEW_THRESHOLD:
        status = "REVIEW"
    else:
        status = "PASS"

    passed = status != "FAIL"

    if signals:
        detail = "; ".join(f"{s.name}(+{s.points}): {s.reason}" for s in signals)
        reason = f"{status} (score={total_score}): {detail}"
    else:
        reason = f"{status} (score=0): no risk signals triggered"

    return {
        "passed": passed,
        "status": status,
        "risk_score": total_score,
        "signals": [s._asdict() for s in signals],
        "reason": reason,
        "failed_layer": None,
    }