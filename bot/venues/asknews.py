"""AskNews search — the second research source, free for registered bot makers.

WHY A SECOND SOURCE
-------------------

The strongest predictor of tournament score in the Fall 2025 survey was the
NUMBER of distinct research sources (r = 0.42) — not which one. AskNews is the
tournament's sponsored option: 1k calls/month per registered maker, and its
output shape (dated article summaries, not an LLM narrative) complements the
model-native web search the forecast models already run. Registration is an
account-owner action (AskNews account under the bot's email + a message to their team).

HOW IT COULD LIE
----------------

- **Absent credentials must degrade research, not kill the run.** `search`
  returns None when anything fails; the orchestrator logs the degradation and
  proceeds on the remaining sources. A missing bonus source is not a reason to
  miss a submission window.
- **Endpoint drift.** These paths follow docs.asknews.app as read on
  2026-08-18 and are exercised for real only once credentials exist. The first
  live run must eyeball one article list before the output is trusted
  (rule 6: prove the check can fire).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import httpx

TOKEN_URL = "https://auth.asknews.app/oauth2/token"
SEARCH_URL = "https://api.asknews.app/v1/news/search"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)
USER_AGENT = "metaculus-bot/0.2 (devinjones-bot; +https://github.com/devinjones521/metaculus-bot)"


@dataclass(frozen=True)
class Article:
    title: str
    summary: str
    source: str
    published: str

    def render(self) -> str:
        return f"[{self.published}] {self.title} ({self.source})\n{self.summary}"


class AskNewsClient:
    """Client-credentials OAuth2, then bearer search calls."""

    def __init__(
        self, client_id: str, client_secret: str, *, client: httpx.Client | None = None
    ) -> None:
        if not client_id.strip() or not client_secret.strip():
            raise ValueError("empty AskNews credentials")
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self._own_client = client is None
        self._http = client or httpx.Client(
            timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        self._token: str | None = None
        self._token_expiry: dt.datetime | None = None

    def close(self) -> None:
        if self._own_client:
            self._http.close()

    def _bearer(self) -> str:
        now = dt.datetime.now(dt.UTC)
        if self._token and self._token_expiry and now < self._token_expiry:
            return self._token
        response = self._http.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "news",
            },
        )
        response.raise_for_status()
        payload = response.json()
        self._token = str(payload["access_token"])
        lifetime = int(payload.get("expires_in", 3600))
        self._token_expiry = now + dt.timedelta(seconds=max(lifetime - 60, 60))
        return self._token

    def search(self, query: str, *, n_articles: int = 8) -> list[Article] | None:
        """Recent articles for `query`, or None on ANY failure.

        None, not [] — an outage recorded as "no news exists" would quietly bias
        every forecast toward the status quo (the Vitl lesson, again).
        """
        try:
            token = self._bearer()
            response = self._http.get(
                SEARCH_URL,
                params={
                    "query": query,
                    "n_articles": n_articles,
                    "return_type": "dicts",
                    "method": "kw",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != httpx.codes.OK:
                return None
            payload = response.json()
            raw_articles = payload.get("as_dicts") or payload.get("articles") or []
            out: list[Article] = []
            for raw in raw_articles:
                if not isinstance(raw, dict):
                    continue
                out.append(
                    Article(
                        title=str(raw.get("eng_title") or raw.get("title") or ""),
                        summary=str(raw.get("summary") or ""),
                        source=str(raw.get("source_id") or raw.get("domain_url") or ""),
                        published=str(raw.get("pub_date") or ""),
                    )
                )
            return out
        except Exception:  # noqa: BLE001 - degrade to "source unavailable"
            return None
