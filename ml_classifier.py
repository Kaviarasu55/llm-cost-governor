"""
ml_classifier.py — Intelligent LLM Cost Governor, Phase 2

ML-based query type classification (TF-IDF + Logistic Regression),
running alongside the rule-based classifier.py — not replacing it.
Matches classifier.py's classify_query() interface so either one can
feed the router behind a common contract.

multi_part detection stays rule-based (classifier.is_multi_part) —
the dataset only has 7 true multi_part examples out of 208, nowhere
near enough to train a separate classifier for it.

Requires ml_classifier.pkl to exist — run train_classifier.py first.
"""

import joblib

import classifier  # reuse existing multi_part heuristic

MODEL_PATH = "ml_classifier.pkl"

_pipeline = None  # lazy-loaded singleton, loaded once per process


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def classify_query_ml(query: str):
    """
    Classify a query into one of the 8 base types using the trained
    ML pipeline.

    Returns:
        (query_type: str, multi_part: bool)

    Same return shape as classifier.classify_query(), so this is a
    drop-in alternative behind the same interface.
    """
    pipeline = _get_pipeline()
    # BUG FOUND WHILE TESTING v3 LIVE: sklearn's .predict() returns numpy
    # array elements, not native Python strings — query_type came back as
    # np.str_('INSTRUCTION_HOWTO') instead of a plain str. It's a subclass
    # of str so dict lookups and == comparisons happened to work fine
    # everywhere downstream, but it's a landmine for things like
    # json.dumps() or certain SQLite paths later. Cast to a real str here,
    # at the one place it enters the system, rather than downstream.
    query_type = str(pipeline.predict([query])[0])
    multi_part = classifier.is_multi_part(query)
    return query_type, multi_part


def classify(query: str):
    """
    Full pre-classifier + classifier entry point (ML backend).

    Returns the exact same dict shape as classifier.classify() —
    {is_conversational, query_type, is_multi_part, risk_level} — so
    pipeline.py can swap between rule-based and ML backends with no
    other code changes. Conversational bypass and risk-level/MULTI_PART
    logic are reused from classifier.py rather than duplicated, so both
    backends apply identical rules downstream of type detection —
    the only thing that actually differs is how query_type is predicted.
    """
    if classifier.is_conversational(query):
        return {
            "is_conversational": True,
            "query_type": None,
            "is_multi_part": None,
            "risk_level": None,
        }

    query_type, multi_part = classify_query_ml(query)
    return classifier.build_result(query_type, multi_part)