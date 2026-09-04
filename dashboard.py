"""
dashboard.py — Intelligent LLM Cost Governor
Streamlit UI: a query box to run queries through the pipeline, plus
an aggregate stats dashboard (volume, tier distribution, escalation
rate, cost-saved-vs-baseline, per-type gate pass/fail).

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd

import pipeline
import logger
import config

# Ensure the logs table exists before anything else runs
logger.init_db()

st.set_page_config(page_title="LLM Cost Governor", layout="wide")

st.title("🧭 Intelligent LLM Cost Governor")
st.caption(
    "Routes queries to the cheapest capable model tier, escalating only "
    "when a transparent quality gate fails. Every decision is logged and inspectable."
)

# ----------------------------------------------------------------------------
# QUERY SECTION
# ----------------------------------------------------------------------------
st.header("Run a Query")

query_text = st.text_area("Enter a query", height=100, placeholder="e.g. What is the difference between Python and JavaScript?")

if st.button("Run", type="primary"):
    if not query_text.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Routing and processing..."):
            result = pipeline.run_query(query_text)

        if result["error"]:
            st.error(f"Something went wrong: {result['error']}")
        else:
            st.success("Response received")
            st.markdown(result["response_text"])

            st.divider()
            st.subheader("Routing details")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Query type", result["query_type"])
            col2.metric("Final tier", result["final_tier"])
            col3.metric("Escalated?", "Yes" if result["escalated"] else "No")
            col4.metric("Gate passed?", "Yes" if result["gate_pass"] else "No")

            col5, col6, col7 = st.columns(3)
            col5.metric("Cost", f"${result['cost']:.6f}")
            col6.metric("Baseline cost (Specialized tier)", f"${result['baseline_cost']:.6f}")
            saved = result["baseline_cost"] - result["cost"]
            col7.metric("Saved on this query", f"${saved:.6f}")

            if result["is_multi_part"]:
                st.caption("🔀 Detected as a multi-part query")
            if result["risk_level"]:
                st.caption(f"⚠️ Quality risk level: {result['risk_level']}")

st.divider()

# ----------------------------------------------------------------------------
# AGGREGATE STATS SECTION
# ----------------------------------------------------------------------------
st.header("Dashboard")

stats = logger.get_summary_stats()

if stats["total_queries"] == 0:
    st.info("No queries logged yet — run a query above to see stats here.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total queries", stats["total_queries"])
    col2.metric("Escalation rate", f"{stats['escalation_rate'] * 100:.1f}%")
    col3.metric("Total cost", f"${stats['total_cost']:.4f}")
    col4.metric(
        "Cost saved vs. always-Specialized baseline",
        f"${stats['cost_saved']:.4f}",
        help="Baseline = cost if every query had gone straight to the Specialized tier."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Tier distribution")
        if stats["tier_distribution"]:
            tier_df = pd.DataFrame(
                list(stats["tier_distribution"].items()),
                columns=["Tier", "Count"]
            ).set_index("Tier")
            st.bar_chart(tier_df)

    with right:
        st.subheader("Per-type gate pass/fail")
        if stats["per_type_pass_fail"]:
            rows = []
            for qtype, counts in stats["per_type_pass_fail"].items():
                rows.append({"Type": qtype, "Pass": counts["pass"], "Fail": counts["fail"]})
            pass_fail_df = pd.DataFrame(rows).set_index("Type")
            st.bar_chart(pass_fail_df)

    st.divider()
    st.subheader("Recent queries")
    logs = logger.get_all_logs()
    if logs:
        display_df = pd.DataFrame(logs)[
            ["timestamp", "query_type", "initial_tier", "final_tier",
             "escalated", "gate_pass", "cost_incurred", "latency_seconds"]
        ].head(20)
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s")
        st.dataframe(display_df, use_container_width=True)

# ----------------------------------------------------------------------------
# LIMITATIONS NOTE
# ----------------------------------------------------------------------------
st.divider()
with st.expander("Documented limitations"):
    st.markdown("""
    - The quality gate measures a **heuristic failure-risk signal**, not ground-truth
      correctness. A structurally complete but factually wrong answer can still pass.
    - Query classification is **rule-based** (keyword/pattern matching), not ML-trained —
      it can misclassify edge cases that don't match expected phrasing.
    - CREATIVE queries have no depth check beyond Layer 1 — there's no way to verify
      creative quality heuristically.
    - COMPARISON, SUMMARY, and MULTI_PART checks rely on best-effort text extraction
      and may skip verification when items/sub-questions can't be confidently parsed.
    - Logs are **ephemeral** on a hosted deployment (e.g. Streamlit Community Cloud) —
      they reset on restart/sleep/redeploy. This is a documented tradeoff, not a bug.
    """)