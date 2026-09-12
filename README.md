# Intelligent LLM Cost Governor

🔗 **Live Demo:** [https://llm-cost-governor.streamlit.app/](https://llm-cost-governor.streamlit.app/)

Routes each incoming query to the **cheapest model tier capable of answering it well**, escalating to a stronger (pricier) tier only when a transparent, inspectable quality gate says the response isn't good enough. Every routing decision, gate score, and escalation is logged and explainable — nothing is a black box.

Built on [Groq](https://groq.com/) with three model tiers:

| Tier | Model | Use case |
|---|---|---|
| Economy | `openai/gpt-oss-20b` | Short factual, explanation, summary, creative, how-to |
| Standard | `openai/gpt-oss-120b` | Comparison, code gen, reasoning/math |
| Specialized | `qwen/qwen3.6-27b` | Escalation target when a cheaper tier fails the gate |

## How it works

```
query → classify → route to tier → call model → quality gate → escalate if FAIL → log
```

1. **Classify** — every query is tagged with one of 8 types (`SHORT_FACTUAL`, `EXPLANATION`, `COMPARISON`, `CODE_GEN`, `REASONING_MATH`, `SUMMARY`, `CREATIVE`, `INSTRUCTION_HOWTO`), plus a `MULTI_PART` flag for queries with multiple sub-questions. Pure conversational filler ("hi", "thanks") bypasses the whole pipeline and goes straight to the Economy tier for free.

   Two interchangeable classifier backends live behind the same interface (`classify()` → `{is_conversational, query_type, is_multi_part, risk_level}`):
   - **Rule-based** (`classifier.py`) — regex/keyword heuristics. Fully transparent and deterministic, but only ~32% type accuracy on the held-out test set.
   - **ML** (`ml_classifier.py`) — TF-IDF (1-2 grams) + LinearSVC, trained in `train_classifier.py`. ~78% type accuracy / ~92% tier accuracy on the same holdout. **This is the production default** (`config.CLASSIFIER_BACKEND = "ml"`).

2. **Route** — `router.py` maps query type → starting tier per a locked policy in `config.py`, with a `MULTI_PART` override that bumps the starting tier up if risk was elevated to `high` but the default tier was still Economy.

3. **Quality gate** (`quality_gate.py`, v3) — instead of a bool pass/fail, every response accumulates a weighted risk score from three layers of checks (universal: empty/refusal/truncated; type-specific: e.g. missing code block, no steps, no final numeric result; risk-based: word-count floor, multi-part coverage). The score maps to:
   - **0–19 → PASS** — accepted, no concern
   - **20–49 → REVIEW** — accepted, but flagged for manual spot-checking later
   - **50+ → FAIL** — rejected, triggers escalation to the next tier

   This measures a *heuristic failure-risk signal* (structural completeness), **not ground-truth correctness** — a fluent but factually wrong answer can still score 0. That's a documented limitation, not a bug.

4. **Log** — every query's full journey (type, tiers, escalation, gate score/reason/signals, tokens, cost, latency) is written to SQLite. Cost is tracked against a baseline (what the query would've cost at the Specialized tier every time) so savings are measurable.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

The trained ML classifier (`ml_classifier.pkl`) is already included. To retrain it from `llm_query_classifier_dataset.csv`:

```bash
python train_classifier.py
```

## Usage

Launch the dashboard (query box + live routing/cost stats + filterable log explorer):

```bash
streamlit run dashboard.py
```

Or drive it programmatically:

```python
import pipeline
result = pipeline.run_query("What is the difference between TCP and UDP?")
```

## Evaluation

Head-to-head accuracy of both classifier backends on a separate, naturally-phrased holdout set (`holdout_test_set.csv`) they were never trained/tuned on:

```bash
python evaluate_holdout.py
```

## Project structure

```
config.py              # model tiers, pricing, routing policy, risk levels
classifier.py           # rule-based classifier + shared risk/multi-part logic
ml_classifier.py        # ML classifier (TF-IDF + LinearSVC), same interface
train_classifier.py     # trains and persists ml_classifier.pkl
evaluate_holdout.py     # rule vs ML accuracy comparison
router.py               # classification -> starting tier, escalation order
model_client.py         # Groq API wrapper (cost/latency/retries)
quality_gate.py         # v3 weighted risk-scoring gate
pipeline.py             # glues everything together end-to-end
logger.py               # SQLite logging + aggregate stats
dashboard.py            # Streamlit UI
```

## Known limitations

- The quality gate is a heuristic proxy for failure risk, not a correctness checker.
- `CREATIVE` queries have no depth check beyond the universal layer — no heuristic exists yet for creative quality.
- `COMPARISON`, `SUMMARY`, and `MULTI_PART` checks rely on best-effort text extraction and may skip verification when items/sub-questions can't be confidently parsed.
- Several gate signal weights are provisional (estimated by analogy to the six weights given in the original plan) and are pending a manual calibration pass against real logged responses.
- Logs are ephemeral on hosted/ sleeping deployments (e.g. Streamlit Community Cloud) — this is a documented tradeoff, not a bug.
