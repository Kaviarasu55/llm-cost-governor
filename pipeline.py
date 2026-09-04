"""
pipeline.py — Intelligent LLM Cost Governor
The glue file: classifier -> router -> model_client -> quality_gate ->
escalation-on-fail -> logger. This is the first true end-to-end run
of the whole system.
"""

import classifier
import router
import model_client
import quality_gate
import logger


def run_query(query_text: str) -> dict:
    """
    Run a single query through the full pipeline.

    Returns a dict with everything the caller (dashboard, CLI, etc.)
    needs to display:
        {
            "response_text": str,
            "query_type": str,
            "is_multi_part": bool,
            "risk_level": str or None,
            "initial_tier": str,
            "final_tier": str,
            "escalated": bool,
            "gate_pass": bool,
            "cost": float,
            "baseline_cost": float,
            "latency_seconds": float,
            "error": str or None,
        }
    """
    classification = classifier.classify(query_text)

    # --- Conversational bypass: cheapest tier, no gate at all ---
    if classification["is_conversational"]:
        result = model_client.call_model("economy", query_text)

        if not result["success"]:
            _log_and_return_error(query_text, "CONVERSATIONAL", False, None, result)
            return _build_output(
                response_text="Sorry, something went wrong processing that.",
                query_type="CONVERSATIONAL", is_multi_part=False, risk_level=None,
                initial_tier="economy", final_tier="economy", escalated=False,
                gate_pass=False, cost=0.0, baseline_cost=0.0,
                latency_seconds=result["latency_seconds"], error=result["error"],
            )

        baseline_cost = model_client.calculate_baseline_cost(
            result["input_tokens"], result["output_tokens"]
        )
        logger.log_query(
            query_type="CONVERSATIONAL", is_multi_part=False, risk_level="none",
            initial_tier="economy", final_tier="economy", escalated=False,
            gate_pass=True, input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"], cost_incurred=result["cost"],
            baseline_cost=baseline_cost, latency_seconds=result["latency_seconds"],
            raw_query_text=query_text,
        )
        return _build_output(
            response_text=result["response_text"], query_type="CONVERSATIONAL",
            is_multi_part=False, risk_level=None, initial_tier="economy",
            final_tier="economy", escalated=False, gate_pass=True,
            cost=result["cost"], baseline_cost=baseline_cost,
            latency_seconds=result["latency_seconds"], error=None,
        )

    # --- Normal path: classify -> route -> call -> gate -> escalate if needed ---
    query_type = classification["query_type"]
    is_multi_part = classification["is_multi_part"]
    risk_level = classification["risk_level"]

    initial_tier = router.get_initial_tier(classification)
    current_tier = initial_tier
    escalated = False
    escalation_reason = None
    total_cost = 0.0
    total_latency = 0.0
    last_result = None
    gate_result = None

    while True:
        call_result = model_client.call_model(current_tier, query_text)
        total_latency += call_result["latency_seconds"]

        if not call_result["success"]:
            # API failure — try escalating (a bigger model might succeed
            # where a smaller one errored), otherwise give up.
            next_tier = router.get_next_tier(current_tier)
            if next_tier is None:
                _log_failure(
                    query_type, is_multi_part, risk_level, initial_tier,
                    current_tier, escalated, f"API error, no further tier: {call_result['error']}",
                    total_cost, total_latency, query_text,
                )
                return _build_output(
                    response_text="Sorry, something went wrong processing that.",
                    query_type=query_type, is_multi_part=is_multi_part, risk_level=risk_level,
                    initial_tier=initial_tier, final_tier=current_tier, escalated=escalated,
                    gate_pass=False, cost=total_cost, baseline_cost=0.0,
                    latency_seconds=total_latency, error=call_result["error"],
                )
            escalated = True
            escalation_reason = f"API error at {current_tier}: {call_result['error']}"
            current_tier = next_tier
            continue

        total_cost += call_result["cost"]
        last_result = call_result

        gate_result = quality_gate.evaluate(
            query_text, call_result["response_text"], query_type, risk_level, is_multi_part
        )

        if gate_result["passed"]:
            break

        # Gate failed — escalate if possible, otherwise return the best
        # answer we have (from the highest tier) even though it failed.
        next_tier = router.get_next_tier(current_tier)
        if next_tier is None:
            break

        escalated = True
        escalation_reason = f"Gate failed at {current_tier}: {gate_result['reason']}"
        current_tier = next_tier

    baseline_cost = model_client.calculate_baseline_cost(
        last_result["input_tokens"], last_result["output_tokens"]
    )

    logger.log_query(
        query_type=query_type, is_multi_part=is_multi_part, risk_level=risk_level,
        initial_tier=initial_tier, final_tier=current_tier, escalated=escalated,
        escalation_reason=escalation_reason, gate_pass=gate_result["passed"],
        input_tokens=last_result["input_tokens"], output_tokens=last_result["output_tokens"],
        cost_incurred=total_cost, baseline_cost=baseline_cost,
        latency_seconds=total_latency, raw_query_text=query_text,
    )

    return _build_output(
        response_text=last_result["response_text"], query_type=query_type,
        is_multi_part=is_multi_part, risk_level=risk_level, initial_tier=initial_tier,
        final_tier=current_tier, escalated=escalated, gate_pass=gate_result["passed"],
        cost=total_cost, baseline_cost=baseline_cost, latency_seconds=total_latency,
        error=None,
    )


def _log_failure(query_type, is_multi_part, risk_level, initial_tier, final_tier,
                  escalated, escalation_reason, cost, latency, query_text):
    logger.log_query(
        query_type=query_type, is_multi_part=is_multi_part, risk_level=risk_level or "none",
        initial_tier=initial_tier, final_tier=final_tier, escalated=escalated,
        escalation_reason=escalation_reason, gate_pass=False,
        input_tokens=None, output_tokens=None, cost_incurred=cost,
        baseline_cost=0.0, latency_seconds=latency, raw_query_text=query_text,
    )


def _log_and_return_error(query_text, query_type, escalated, risk_level, result):
    logger.log_query(
        query_type=query_type, is_multi_part=False, risk_level=risk_level or "none",
        initial_tier="economy", final_tier="economy", escalated=escalated,
        escalation_reason=result["error"], gate_pass=False,
        input_tokens=None, output_tokens=None, cost_incurred=0.0,
        baseline_cost=0.0, latency_seconds=result["latency_seconds"],
        raw_query_text=query_text,
    )


def _build_output(response_text, query_type, is_multi_part, risk_level, initial_tier,
                   final_tier, escalated, gate_pass, cost, baseline_cost,
                   latency_seconds, error):
    return {
        "response_text": response_text,
        "query_type": query_type,
        "is_multi_part": is_multi_part,
        "risk_level": risk_level,
        "initial_tier": initial_tier,
        "final_tier": final_tier,
        "escalated": escalated,
        "gate_pass": gate_pass,
        "cost": cost,
        "baseline_cost": baseline_cost,
        "latency_seconds": latency_seconds,
        "error": error,
    }