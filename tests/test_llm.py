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
