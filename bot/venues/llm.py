"""OpenRouter chat-completions client — the forecaster's only route to any LLM.

WHY OPENROUTER AND NOT A PER-PROVIDER CLIENT
--------------------------------------------

Metaculus distributes its donated Anthropic/OpenAI/Google credits as OpenRouter
keys (`docs/METHOD.md`), and a personal OpenRouter key exposes the
same models through the same API. One code path serves both funding scenarios,
and switching between them is an `.env` change, not a code change. Appending
`:online` to a model name turns on that provider's native web search, which the
Fall 2025 bot-advice post rates as more reliable than custom search pipelines.

HOW IT COULD LIE
----------------

- **A silent empty completion becomes a dropped forecast run.** `complete`
  raises on any non-2xx, on a missing/empty message, and on the API's in-band
  `error` object (OpenRouter can return HTTP 200 with an error body). The
  caller decides whether a failed run is fatal; this module never converts
  failure into empty-string success.
- **The provider's reason gets lost in the wrapper.** Google's "high demand"
  arrives as HTTP 200 with the error inside the choice, and was logged as
  "empty completion" 12 times during the 2026-09 warmup. Its quota 429s name the
  exhausted quota only after ~300 characters of OpenRouter envelope, past where
  the log cut them off. Both now read as the provider's own words.
- **A completion cut off at max_tokens reads like an answer.** Gemini 3.5 Flash
  spent 7,680 of 8,000 tokens thinking on 2026-09-27 and stopped mid-sentence,
  and the binary parser falls back to the last bare "NN%" anywhere in the text.
  A run that stopped on `finish_reason: length` is raised, not parsed.
- **Cost runs away invisibly.** `key_limits()` exposes OpenRouter's
  `limit_remaining` so the orchestrator can log spend every run — the resources
  page's own recommendation. Budget enforcement lives with the caller.
- **A mistyped model id fails per-call, late and expensively.** `models()`
  returns the live id list so the orchestrator can validate its roster once at
  startup and fail before the first question, not after the fifth.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

API = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=15.0)  # reasoning models are slow
USER_AGENT = "metaculus-bot/0.2 (devinjones-bot; +https://github.com/devinjones521/metaculus-bot)"

# One polite retry on transient statuses, then raise. A loop that retries
# forever would silently eat the submission window.
RETRYABLE = {429, 500, 502, 503}
RETRY_PAUSE_SECONDS = 20.0

# A connection that dies before the response has no status code to match on, so
# it never reached the retry above: on 2026-09-21 one such drop went straight
# out as a failed run and thinned an ensemble to 4/5.
TRANSPORT_RETRY_PAUSE_SECONDS = 5.0


class LlmError(RuntimeError):
    """Raised for any completion that did not produce usable text."""


class OpenRouterClient:
    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        if not api_key.strip():
            raise ValueError("empty OpenRouter key; set OPENROUTER_API_KEY in .env")
        # Running totals for this client's lifetime, fed by the per-call usage
        # block OpenRouter returns when asked. The 2026-08-18 rehearsal burned
        # ~2x the estimated budget before any per-call number existed; cost is
        # now a measurement, not an estimate.
        self.total_cost_usd: float = 0.0
        self.cost_by_model: dict[str, float] = {}
        self._own_client = client is None
        self._http = client or httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "User-Agent": USER_AGENT,
            },
        )

    def close(self) -> None:
        if self._own_client:
            self._http.close()

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 8000,
    ) -> str:
        """One chat completion. Returns the text or raises LlmError — never ''."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to include the exact dollar cost in the response.
            "usage": {"include": True},
        }
        response = self._post_with_one_retry("/chat/completions", body)
        payload = response.json()
        if payload.get("error"):
            raise LlmError(f"{model}: {payload['error']}")
        cost = (payload.get("usage") or {}).get("cost")
        if isinstance(cost, int | float):
            self.total_cost_usd += float(cost)
            self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + float(cost)
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"{model}: malformed completion payload") from exc
        if choice.get("error"):
            error = choice["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            code = error.get("code", "?") if isinstance(error, dict) else "?"
            raise LlmError(f"{model}: provider error {code} in the completion: {detail}")
        if choice.get("finish_reason") == "length":
            raise LlmError(f"{model}: stopped at max_tokens={max_tokens} before finishing")
        if not text or not str(text).strip():
            raise LlmError(
                f"{model}: empty completion (finish_reason={choice.get('finish_reason')})"
            )
        return str(text)

    def models(self) -> set[str]:
        """Live model ids, for validating the roster once at startup."""
        response = self._http.get(f"{API}/models")
        response.raise_for_status()
        data = response.json().get("data", [])
        return {str(entry.get("id")) for entry in data if entry.get("id")}

    def key_limits(self) -> dict[str, Any]:
        """OpenRouter's view of this key: {'limit': ..., 'limit_remaining': ...}."""
        response = self._http.get(f"{API}/key")
        response.raise_for_status()
        data = response.json().get("data", {})
        return {
            "limit": data.get("limit"),
            "limit_remaining": data.get("limit_remaining"),
        }

    def _post_with_one_retry(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """Exactly one retry, whether the status said "later" or the connection
        died with nothing on it.

        Resending a completion can pay twice for one answer, since a request
        that died on the wire may have been served. That is the cheaper half of
        the trade: a run that never happens costs the ensemble a whole model,
        and the ensemble median is this design's measured edge.
        """
        try:
            response = self._http.post(f"{API}{path}", json=body)
        except httpx.TransportError:
            time.sleep(TRANSPORT_RETRY_PAUSE_SECONDS)
            response = self._http.post(f"{API}{path}", json=body)
        else:
            if response.status_code in RETRYABLE:
                retry_after = response.headers.get("Retry-After", "")
                delay = (
                    float(retry_after)
                    if retry_after.replace(".", "", 1).isdigit()
                    else RETRY_PAUSE_SECONDS
                )
                time.sleep(min(delay, 120.0))
                response = self._http.post(f"{API}{path}", json=body)
        if response.status_code != httpx.codes.OK:
            raise LlmError(describe_http_error(response))
        return response


def describe_http_error(response: httpx.Response) -> str:
    """A failed call in the provider's own words, with the part that names a
    quota first, since the caller keeps only the first few hundred characters.

    OpenRouter wraps a provider's error as `error.metadata.raw`, a JSON string
    holding the provider's own `error.message`.
    """
    head = f"HTTP {response.status_code}"
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return f"{head}: {response.text[:300]}"
    if not isinstance(error, dict):
        return f"{head}: {response.text[:300]}"
    metadata = error.get("metadata") or {}
    message = str(error.get("message") or "")
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    if isinstance(raw, str) and raw.strip():
        try:
            inner = json.loads(raw)
        except ValueError:
            inner = None
        inner_error = inner.get("error") if isinstance(inner, dict) else None
        if isinstance(inner_error, dict) and inner_error.get("message"):
            message = str(inner_error["message"])
        else:
            message = raw
    message = " ".join(message.split())
    quota = message.find("Quota exceeded for metric")
    if quota > 0:
        message = message[quota:] + " — " + message[:quota].strip()
    provider = metadata.get("provider_name") if isinstance(metadata, dict) else None
    return f"{head}{f' from {provider}' if provider else ''}: {message[:400]}"
