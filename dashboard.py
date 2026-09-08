"""
dashboard.py — Intelligent LLM Cost Governor
Streamlit UI: a query box to run queries through the pipeline, plus
an aggregate stats dashboard (volume, tier distribution, escalation
rate, cost-saved-vs-baseline, per-type gate status, latency-by-tier)
and a filterable log explorer with per-row signal-level drill-down —
built for the v3 calibration pass (spot-checking REVIEW/FAIL rows).

Run with: streamlit run dashboard.py
"""

import json

import streamlit as st
import pandas as pd

import pipeline
import logger
import config

# Ensure the logs table exists before anything else runs
logger.init_db()

st.set_page_config(page_title="LLM Cost Governor", layout="wide")

st.title("Intelligent LLM Cost Governor")
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
            col4.metric(
                "Gate status",
                result["gate_status"] or ("Pass" if result["gate_pass"] else "Fail"),
                help="PASS/REVIEW/FAIL come from the v3 weighted risk gate. "
                     "REVIEW means the response was accepted (no escalation) "
                     "but flagged as borderline for manual spot-checking later. "
                     "BYPASSED = conversational query, never gated. "
                     "ERROR = API failure, gate never ran."
            )

            if result["risk_score"] is not None:
                st.caption(f"Risk score: {result['risk_score']}")
            if result.get("gate_reason"):
                st.caption(f"Gate reason: {result['gate_reason']}")

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

    st.metric(
        "Flagged for review",
        f"{stats['review_count']} ({stats['review_rate'] * 100:.1f}%)",
        help="Responses that were accepted (not escalated) but scored in the "
             "20-49 risk-score band — borderline cases worth a manual spot-check "
             "when calibrating the gate's weights."
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
        st.subheader("Per-type gate status (PASS / REVIEW / FAIL)")
        if stats["per_type_status"]:
            rows = []
            for qtype, counts in stats["per_type_status"].items():
                rows.append({
                    "Type": qtype,
                    "Pass": counts.get("PASS", 0),
                    "Review": counts.get("REVIEW", 0),
                    "Fail": counts.get("FAIL", 0),
                })
            status_df = pd.DataFrame(rows).set_index("Type")
            st.bar_chart(status_df)

    st.subheader("Average latency by tier")
    if stats["avg_latency_by_tier"]:
        latency_df = pd.DataFrame(
            list(stats["avg_latency_by_tier"].items()),
            columns=["Tier", "Avg latency (s)"]
        ).set_index("Tier")
        st.bar_chart(latency_df)

    st.divider()

    # ------------------------------------------------------------------------
    # LOG EXPLORER — filterable table + per-row drill-down. Built for the
    # calibration pass (v3 plan's "15-20 manually-judged real logged
    # responses"): rather than scrolling a flat dataframe, filter down to
    # e.g. "COMPARISON x REVIEW" and inspect exactly which signals fired.
    # ------------------------------------------------------------------------
    st.subheader("Log explorer")
    logs = logger.get_all_logs()

    if not logs:
        st.info("No queries logged yet.")
    else:
        all_types = sorted({l["query_type"] for l in logs})
        all_statuses = ["PASS", "REVIEW", "FAIL", "BYPASSED", "ERROR"]

        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        with filter_col1:
            type_filter = st.multiselect("Query type", options=all_types, default=[])
        with filter_col2:
            status_filter = st.multiselect(
                "Gate status", options=all_statuses, default=["REVIEW", "FAIL"],
                help="Defaults to REVIEW + FAIL — the rows worth a manual look "
                     "when calibrating gate weights. Clear to see everything."
            )
        with filter_col3:
            row_limit = st.number_input("Max rows", min_value=5, max_value=500, value=50, step=5)

        full_df = pd.DataFrame(logs)
        # Normalize missing gate_status on legacy rows the same way
        # get_summary_stats() does, so filtering behaves consistently.
        full_df["gate_status"] = full_df.apply(
            lambda row: row["gate_status"] or ("PASS" if row["gate_pass"] else "FAIL"),
            axis=1,
        )

        filtered_df = full_df
        if type_filter:
            filtered_df = filtered_df[filtered_df["query_type"].isin(type_filter)]
        if status_filter:
            filtered_df = filtered_df[filtered_df["gate_status"].isin(status_filter)]

        st.caption(f"{len(filtered_df)} of {len(full_df)} logged queries match the current filter.")

        display_df = filtered_df[
            ["id", "timestamp", "query_type", "initial_tier", "final_tier",
             "escalated", "gate_status", "risk_score", "gate_reason",
             "cost_incurred", "latency_seconds"]
        ].head(int(row_limit)).copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # --- Per-row drill-down: signal-level breakdown for one query ---
        if not filtered_df.empty:
            st.markdown("**Inspect a single query**")
            options = filtered_df["id"].head(int(row_limit)).tolist()

            def _label(row_id):
                row = filtered_df[filtered_df["id"] == row_id].iloc[0]
                return f"#{row_id} — {row['query_type']} — {row['gate_status']} (score={row['risk_score']})"

            selected_id = st.selectbox("Select a logged query by ID", options=options, format_func=_label)

            if selected_id is not None:
                row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
                st.write(f"**Reason:** {row['gate_reason'] or '(none recorded)'}")

                signals_raw = row.get("gate_signals")
                if signals_raw:
                    try:
                        signals = json.loads(signals_raw)
                    except (TypeError, json.JSONDecodeError):
                        signals = []
                    if signals:
                        st.table(pd.DataFrame(signals)[["name", "points", "reason"]])
                    else:
                        st.caption("No individual risk signals were triggered.")
                else:
                    st.caption("No signal breakdown recorded for this row.")

                if row.get("raw_query_text"):
                    st.text_area("Original query", row["raw_query_text"], height=80, disabled=True)

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