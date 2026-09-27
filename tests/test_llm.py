"""Tests for the OpenRouter venue's one retry.

The retry existed for statuses only — 429, 500, 502, 503 — which is a shape a
dropped connection never has. On 2026-09-21 a MiniBench sweep lost a gemini run
to `RemoteProtocolError: Server disconnected without sending a response.` and
forecast the question on four models instead of five. The ensemble median is
this design's measured edge, so a run lost to the wire is not a free loss.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from bot.venues.llm import LlmError, OpenRouterClient

COMPLETION = {
    "choices": [{"message": {"content": "Probability: 30%"}}],
    "usage": {"cost": 0.01},
}


class _Wire:
    """Scripted responses, in order, counting attempts."""

    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if callable(step):
            return step(request)
        return step

    def client(self) -> OpenRouterClient:
        return OpenRouterClient("key", client=httpx.Client(transport=httpx.MockTransport(self)))


def _dropped(_: httpx.Request) -> httpx.Response:
    raise httpx.RemoteProtocolError("Server disconnected without sending a response.")


def test_a_completion_lost_on_the_wire_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire([_dropped, httpx.Response(200, json=COMPLETION)])
    assert wire.client().complete("x-ai/grok-4.6", "prompt") == "Probability: 30%"
    assert wire.attempts == 2


def test_a_second_drop_is_raised_not_retried_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire([_dropped])
    with pytest.raises(httpx.RemoteProtocolError):
        wire.client().complete("x-ai/grok-4.6", "prompt")
    assert wire.attempts == 2


def test_a_drop_then_a_429_spends_no_second_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """One retry per call, whatever mix of failures it meets: a call that keeps
    buying retries is how a bot sleeps through its submission window."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire([_dropped, httpx.Response(429, text="slow down")])
    with pytest.raises(LlmError, match="HTTP 429"):
        wire.client().complete("google/gemini-3.8-flash", "prompt")
    assert wire.attempts == 2


def test_the_status_retry_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire([httpx.Response(503, text="high demand"), httpx.Response(200, json=COMPLETION)])
    assert wire.client().complete("google/gemini-3.8-flash", "prompt") == "Probability: 30%"
    assert wire.attempts == 2


# ------------------------------------------------ the provider's own words
#
# 2026-09-27: twelve warmup runs had been logged as "empty completion". Probed,
# they were Google's "high demand" sent as HTTP 200 with the error inside the
# choice. And every quota 429 was cut off in the log before the part naming
# the quota, behind ~300 characters of OpenRouter's envelope.

HIGH_DEMAND_IN_BAND = {
    "choices": [
        {
            "message": {"content": ""},
            "finish_reason": "error",
            "error": {
                "code": 502,
                "message": "This model is currently experiencing high demand.",
                "metadata": {"error_type": "provider_unavailable"},
            },
        }
    ],
    "usage": {"cost": 0, "is_byok": True},
}

GOOGLE_RAW_429 = (
    '{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, '
    "please check your plan and billing details. For more information on this error, head "
    "to: https://ai.google.dev/gemini-api/docs/rate-limits.\\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_paid_tier_requests, limit: 150, "
    'model: gemini-3.8-flash",\n    "status": "RESOURCE_EXHAUSTED"\n  }\n}\n'
)

GOOGLE_QUOTA_429 = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {
            "raw": GOOGLE_RAW_429,
            "provider_name": "Google AI Studio",
            "is_byok": True,
        },
    }
}


def test_an_in_band_provider_error_is_named_not_called_empty() -> None:
    wire = _Wire([httpx.Response(200, json=HIGH_DEMAND_IN_BAND)])
    with pytest.raises(LlmError) as raised:
        wire.client().complete("google/gemini-3.8-flash", "prompt")
    message = str(raised.value)
    assert "high demand" in message and "502" in message
    assert "empty completion" not in message


def test_a_completion_cut_off_at_max_tokens_is_raised_not_parsed() -> None:
    """Cut off mid-thought, with a bare percentage in it that the binary parser
    would otherwise have taken for the answer."""
    truncated = {
        "choices": [
            {
                "message": {"content": "Base rate is about 20% in October. But wait! If"},
                "finish_reason": "length",
            }
        ]
    }
    wire = _Wire([httpx.Response(200, json=truncated)])
    with pytest.raises(LlmError, match="max_tokens"):
        wire.client().complete("google/gemini-3.5-flash", "prompt")


def test_a_quota_429_leads_with_the_quota_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire([httpx.Response(429, json=GOOGLE_QUOTA_429)])
    with pytest.raises(LlmError) as raised:
        wire.client().complete("google/gemini-3.8-flash", "prompt")
    message = str(raised.value)
    assert message.startswith("HTTP 429 from Google AI Studio: Quota exceeded for metric")
    # The pipeline keeps 240 characters of a run's error; the quota must be in them.
    assert "model: gemini-3.8-flash" in message[:240]
    assert "You exceeded your current quota" in message  # the rest is kept, after it
