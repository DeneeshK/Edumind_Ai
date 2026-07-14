"""Unit tests for LLM token + cost accounting (clients/groq_client + config).

Covers:
  (a) usage extraction records token metrics with a mocked Groq response;
  (b) cost computation math against a fake price entry;
  (c) unpriced (zero) models skip cost recording.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
from clients import groq_client
from core.metrics import metrics


pytestmark = pytest.mark.unit


def _counter_value(counter, **labels) -> float:
    """Read the current value of a labeled prometheus_client Counter child."""
    return counter.labels(**labels)._value.get()


def _response_with_usage(text: str, prompt_tokens: int, completion_tokens: int):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.mark.asyncio
async def test_generate_records_token_metrics_from_usage(monkeypatch):
    model = "unit-test-token-model"
    caller = "unit_test_caller"

    before_prompt = _counter_value(
        metrics.llm_tokens, model=model, caller=caller, direction="prompt"
    )
    before_completion = _counter_value(
        metrics.llm_tokens, model=model, caller=caller, direction="completion"
    )

    monkeypatch.setattr(groq_client, "get_client", lambda: object())

    async def fake_with_retry(fn, *args, **kwargs):
        return _response_with_usage("hi", prompt_tokens=120, completion_tokens=30)

    monkeypatch.setattr(groq_client, "_with_retry", fake_with_retry)

    result = await groq_client.generate(
        messages=[{"role": "user", "content": "hello"}],
        model=model,
        _caller=caller,
    )

    assert result == "hi"
    after_prompt = _counter_value(
        metrics.llm_tokens, model=model, caller=caller, direction="prompt"
    )
    after_completion = _counter_value(
        metrics.llm_tokens, model=model, caller=caller, direction="completion"
    )
    assert after_prompt - before_prompt == 120
    assert after_completion - before_completion == 30


def test_compute_llm_cost_uses_price_table(monkeypatch):
    # Fake price: $2 / 1M input tokens, $6 / 1M output tokens.
    monkeypatch.setitem(config.GROQ_MODEL_PRICES, "fake-priced-model", (2.0, 6.0))

    cost = config.compute_llm_cost_usd(
        "fake-priced-model", prompt_tokens=1_000_000, completion_tokens=500_000
    )
    # 1M input * $2/1M + 0.5M output * $6/1M = 2.0 + 3.0 = 5.0
    assert cost == pytest.approx(5.0)


def test_compute_llm_cost_skips_zero_priced_and_unknown_models():
    # Configured models ship as placeholder zeros → cost accounting disabled.
    assert config.compute_llm_cost_usd("openai/gpt-oss-120b", 1000, 1000) is None
    # Unknown model → None (never invent a price).
    assert config.compute_llm_cost_usd("no-such-model", 1000, 1000) is None


def test_record_usage_skips_cost_for_zero_priced_model():
    model = "openai/gpt-oss-120b"  # placeholder (0, 0) price
    caller = "unit_cost_skip_caller"
    before = _counter_value(metrics.llm_cost_usd, model=model, caller=caller)

    returned = groq_client._record_usage(model, caller, 1000, 1000)

    after = _counter_value(metrics.llm_cost_usd, model=model, caller=caller)
    assert returned is None
    assert after == before  # no cost recorded for an unpriced model
