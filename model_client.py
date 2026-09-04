"""
model_client.py — Intelligent LLM Cost Governor
Wraps all Groq API calls. Every call returns a consistent result dict
regardless of tier, including token counts, cost, latency, and
success/error status — this is what quality_gate.py and pipeline.py
build on top of.
"""

import time

from groq import Groq, APIError, APIConnectionError, APITimeoutError, RateLimitError

import config
import router


client = Groq(api_key=config.GROQ_API_KEY)


def _calculate_cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the dollar cost of a call based on config.PRICING (per 1M tokens)."""
    rates = config.PRICING[tier]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


def call_model(tier: str, query: str, max_retries: int = 2) -> dict:
    """
    Call the model for a given tier with the user's query.

    Returns a dict with:
        success (bool)
        response_text (str or None)
        input_tokens (int or None)
        output_tokens (int or None)
        cost (float or None)
        latency_seconds (float)
        error (str or None) — human-readable reason if success is False
        tier (str)
        model_id (str)

    Retries transient errors (timeout, connection, rate limit) up to
    max_retries times with a short backoff before giving up. On final
    failure, returns success=False rather than raising, so the pipeline
    can decide what to do (e.g. log and surface an error to the user)
    instead of crashing the whole run.
    """
    model_id = router.get_model_id(tier)
    attempt = 0
    last_error = None

    while attempt <= max_retries:
        start_time = time.time()
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": query}],
                reasoning_format="parsed",
            )
            latency = time.time() - start_time

            response_text = completion.choices[0].message.content
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            cost = _calculate_cost(tier, input_tokens, output_tokens)

            return {
                "success": True,
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency_seconds": latency,
                "error": None,
                "tier": tier,
                "model_id": model_id,
            }

        except RateLimitError as e:
            last_error = f"Rate limit hit: {e}"
            attempt += 1
            time.sleep(2 * attempt)  # simple backoff: 2s, 4s

        except (APITimeoutError, APIConnectionError) as e:
            last_error = f"Connection/timeout issue: {e}"
            attempt += 1
            time.sleep(1 * attempt)

        except APIError as e:
            # Non-retryable API error (e.g. bad request, invalid model) —
            # fail immediately, no point retrying.
            last_error = f"Groq API error: {e}"
            break

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            break

    latency = time.time() - start_time
    return {
        "success": False,
        "response_text": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
        "latency_seconds": latency,
        "error": last_error,
        "tier": tier,
        "model_id": model_id,
    }


def calculate_baseline_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate what this query would have cost if it had gone straight
    to the Specialized tier — this is the dashboard's savings baseline,
    per the locked definition.
    """
    return _calculate_cost("specialized", input_tokens, output_tokens)