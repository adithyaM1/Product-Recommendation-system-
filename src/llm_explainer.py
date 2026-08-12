"""
llm_explainer.py
-----------------
Optional enhancement layer. The recommender's core logic (recommender.py)
never needs an LLM -- it always produces a solid, rule-based reason.

If the user sets ANTHROPIC_API_KEY, this module asks Claude to rewrite that
rule-based reason as one friendly sentence. If the key is missing, the
`anthropic` package isn't installed, or the API call fails for any reason,
we silently fall back to the original rule-based reason so the agent never
breaks because of the network or a bad key.
"""

import os

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


def enhance_reason_with_llm(user_name: str, product_name: str, base_reason: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return base_reason

    try:
        import anthropic  # imported lazily so the package is only required if used

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Rewrite the following product recommendation reason as ONE short, "
            "friendly sentence (max 25 words). Keep every factual detail "
            f"(matched interests, category, budget fit). User: {user_name}. "
            f"Product: {product_name}.\n\nReason: {base_reason}"
        )
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        return text or base_reason
    except Exception:
        # Network issues, bad key, rate limits, missing package, etc.
        # The agent must keep working without the LLM, so we fail quietly.
        return base_reason
