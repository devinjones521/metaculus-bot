"""Metaculus bots-only forecasting tournaments — read questions, submit forecasts.

THE VENUE
---------

Metaculus pays bots to forecast, on purpose: the tournaments exist to benchmark
AI forecasting against humans, so the prize money is the research instrument.
Measured 2026-08-18: 165 bots on the Summer 2026 leaderboard, a $50k seasonal
pool, and an unbroken bi-weekly $1k MiniBench. See `docs/METHOD.md`.

WHAT CAN AND CANNOT HAPPEN THROUGH THIS FILE
--------------------------------------------

Reads: tournament question lists, question detail. Writes: probability forecasts
and reasoning comments on the bot account (`devinjones-bot`). **No money can move
through this API.** Prize payout, if it ever happens, is a manual step on the
account owner's side. There is no payment, order, or transfer surface here to guard, because none
exists.

HOW IT COULD LIE
----------------

- **A submission that fails quietly is a forecast that never happened.** On a
  scored tournament, an absent forecast loses points exactly like a wrong one,
  and a bot that thinks it forecast 60 questions but submitted 40 reports a
  clean run — the failure that looks like success. Every write in this module
  therefore raises on any non-2xx response: the opposite convention from the
  recorders, whose *reads* degrade to None on purpose.
- **Skipped pages look like a small tournament.** `open_questions` follows the
  server's `next` cursor to the end; a one-page read of a 500-question season
  would silently forecast 100 questions and ignore 400.
- **An unknown question type must not be guessed at.** Parsing keeps the type
  string as the server sent it; the payload builders accept only the shapes they
  understand and raise on anything else. Guessing a payload shape for a new type
  would submit a syntactically valid, semantically wrong forecast.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

API = "https://www.metaculus.com/api"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=15.0)
USER_AGENT = "metaculus-bot/0.2 (devinjones-bot; +https://github.com/devinjones521/metaculus-bot)"

# Deliberately tighter than whatever the server accepts today. Tail overconfidence
# is exactly what a Brier score punishes hardest, and nothing in this project's
# evidence base justifies pricing anything below 1% or above 99%.
BINARY_FLOOR = 0.01
BINARY_CEIL = 0.99

# The continuous-question CDF the API expects: 201 points, inclusive bounds.
CDF_POINTS = 201

# Pagination pacing and the cap on honouring a server-sent Retry-After.
PAGE_PAUSE_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 120.0


@dataclass(frozen=True)
class Question:
    """One forecastable question, flattened from a post.

    `qtype` is the server's string, unmodified ("binary", "multiple_choice",
    "numeric", "discrete", "date", or whatever arrives next). Numeric-family
    scaling fields are None wherever the server did not send them.
    """

    post_id: int
    question_id: int
    title: str
    qtype: str
    description: str
    resolution_criteria: str
    fine_print: str
    options: tuple[str, ...]
    open_time: str
    close_time: str
    unit: str
    range_min: float | None
    range_max: float | None
    open_lower_bound: bool | None
    open_upper_bound: bool | None
    zero_point: float | None
    # Discrete questions carry their own grid size; numeric uses 201.
    outcome_count: int | None = None
    # True when the authenticated account already has a forecast standing on
    # this question. CAUTION, learned live 2026-08-18: the LIST endpoint never
    # populates my_forecasts, so this is only trustworthy when parsed from a
    # post DETAIL response — from a listing it is always False, and a guard
    # built on it alone passes by being vacuous. Use forecast_standing().
    already_forecast: bool = False
    # The QUESTION's own status. An open group post can contain resolved
    # sub-questions; forecasting one gets a 405 (observed live 2026-08-18,
    # bot-testing-area, two resolved members of an open group).
    status: str = ""

    @property
    def cdf_points(self) -> int:
        return (self.outcome_count + 1) if self.outcome_count else CDF_POINTS


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _question_from_raw(post: Mapping[str, Any], raw: Mapping[str, Any]) -> Question | None:
    question_id = raw.get("id")
    if question_id is None:
        return None
    scaling = raw.get("scaling") or {}
    options = raw.get("options") or ()
    return Question(
        post_id=int(post.get("id", 0)),
        question_id=int(question_id),
        title=str(raw.get("title") or post.get("title") or ""),
        qtype=str(raw.get("type") or ""),
        description=str(raw.get("description") or ""),
        resolution_criteria=str(raw.get("resolution_criteria") or ""),
        fine_print=str(raw.get("fine_print") or ""),
        options=tuple(str(option) for option in options),
        open_time=str(raw.get("open_time") or ""),
        close_time=str(raw.get("scheduled_close_time") or ""),
        unit=str(raw.get("unit") or ""),
        range_min=_float_or_none(scaling.get("range_min")),
        range_max=_float_or_none(scaling.get("range_max")),
        open_lower_bound=raw.get("open_lower_bound"),
        open_upper_bound=raw.get("open_upper_bound"),
        zero_point=_float_or_none(scaling.get("zero_point")),
        outcome_count=(
            int(scaling["inbound_outcome_count"]) if scaling.get("inbound_outcome_count") else None
        ),
        already_forecast=bool((raw.get("my_forecasts") or {}).get("latest")),
        status=str(raw.get("status") or ""),
    )


def only_open(questions: list[Question]) -> list[Question]:
    """Keep questions whose OWN status is open. The post-level `statuses=open`
    filter is not enough: an open group post can carry resolved members."""
    return [question for question in questions if question.status == "open"]


def parse_posts(payload: Mapping[str, Any]) -> list[Question]:
    """Flatten a /api/posts/ page into Questions.

    Posts with no question payload (notebooks, announcements) are skipped —
    they are not forecastable, so skipping is correct rather than lossy. Group
    posts contribute one Question per sub-question.
    """
    out: list[Question] = []
    for post in payload.get("results", []):
        raw_single = post.get("question")
        if isinstance(raw_single, Mapping):
            question = _question_from_raw(post, raw_single)
            if question is not None:
                out.append(question)
            continue
        group = post.get("group_of_questions")
        if isinstance(group, Mapping):
            for raw in group.get("questions", []):
                if isinstance(raw, Mapping):
                    question = _question_from_raw(post, raw)
                    if question is not None:
                        out.append(question)
    return out


def next_page_url(payload: Mapping[str, Any]) -> str | None:
    """Where to paginate next, or None when done.

    Observed live 2026-08-18: with `statuses=open` and nothing open, the server
    returns an EMPTY results page whose `next` still points at the following
    offset, and does so at every offset — an infinite chain of empty pages. A
    client that trusts `next` alone walks that chain until the rate limiter
    stops it (ours reached offset 800 before the 429 did). An empty page
    therefore ends pagination, whatever `next` claims.
    """
    if not payload.get("results"):
        return None
    url = payload.get("next")
    return str(url) if url else None


def binary_payload(question_id: int, p: float) -> dict[str, Any]:
    """Forecast payload for one binary question. Raises rather than clamps:
    silently moving a probability is a forecast nobody made."""
    if not (BINARY_FLOOR <= p <= BINARY_CEIL):
        raise ValueError(f"binary probability {p} outside [{BINARY_FLOOR}, {BINARY_CEIL}]")
    return {
        "question": question_id,
        "probability_yes": round(float(p), 6),
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }


def multiple_choice_payload(
    question_id: int, probabilities: Mapping[str, float], options: Sequence[str]
) -> dict[str, Any]:
    """Forecast payload for a multiple-choice question.

    The keys must match the question's options exactly — a missing or extra key
    means the caller is answering a different question than the one asked.
    """
    if set(probabilities) != set(options):
        missing = set(options) - set(probabilities)
        extra = set(probabilities) - set(options)
        raise ValueError(f"options mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    values = list(probabilities.values())
    if any(not (0.001 <= value <= 0.999) for value in values):
        raise ValueError(f"multiple-choice probabilities outside [0.001, 0.999]: {values}")
    total = sum(values)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"multiple-choice probabilities sum to {total}, not 1")
    return {
        "question": question_id,
        "probability_yes": None,
        "probability_yes_per_category": {k: round(float(v), 6) for k, v in probabilities.items()},
        "continuous_cdf": None,
    }


def numeric_payload(
    question_id: int, cdf: Sequence[float], expected_points: int = CDF_POINTS
) -> dict[str, Any]:
    """Forecast payload for a continuous (numeric/discrete/date) question.

    The API wants a CDF over the question's scaled range: 201 points for
    numeric, outcome_count+1 for discrete. The server rejects steps below
    0.01/(n-1) ("values must be in strictly increasing order" cost one Q2 2025
    maker ~30% of their numeric submissions), so that floor is enforced here
    too — a payload this function accepts must be one the server accepts.
    """
    values = [float(value) for value in cdf]
    if len(values) != expected_points:
        raise ValueError(f"cdf has {len(values)} points, expected {expected_points}")
    if any(not (0.0 <= value <= 1.0) for value in values):
        raise ValueError("cdf values must lie in [0, 1]")
    min_step = 0.01 / (expected_points - 1) * 0.98
    if any(later - earlier < min_step for earlier, later in zip(values, values[1:], strict=False)):
        raise ValueError(f"cdf steps must increase by at least {min_step:.2e}")
    return {
        "question": question_id,
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": values,
    }


class MetaculusClient:
    """Authenticated client for the bot account. Token from `.env`, never source."""

    def __init__(self, token: str, *, client: httpx.Client | None = None) -> None:
        if not token.strip():
            raise ValueError("empty Metaculus token; set METACULUS_TOKEN in .env")
        self._own_client = client is None
        self._http = client or httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Token {token.strip()}",
                "User-Agent": USER_AGENT,
            },
        )

    def close(self) -> None:
        if self._own_client:
            self._http.close()

    def open_questions(self, tournament: str) -> list[Question]:
        """Every currently-open question in a tournament, across all pages."""
        questions: list[Question] = []
        url: str | None = f"{API}/posts/"
        params: dict[str, Any] | None = {
            "tournaments": tournament,
            "statuses": "open",
            "limit": 100,
            "with_cp": "false",
        }
        while url:
            response = self._request("GET", url, params=params)
            payload = response.json()
            questions.extend(parse_posts(payload))
            url = next_page_url(payload)
            params = None  # the `next` URL already carries the query string
            if url:
                time.sleep(PAGE_PAUSE_SECONDS)
        return only_open(questions)

    def forecast_standing(self, post_id: int) -> set[int]:
        """Question ids under this post where THIS account has a forecast.

        Reads the post DETAIL endpoint, because the list endpoint never
        populates my_forecasts (verified live 2026-08-18): a dedup guard fed
        from listings would always pass and always be wrong.
        """
        response = self._request("GET", f"{API}/posts/{post_id}/")
        post = response.json()
        standing: set[int] = set()
        raws: list[Mapping[str, Any]] = []
        if isinstance(post.get("question"), Mapping):
            raws.append(post["question"])
        group = post.get("group_of_questions")
        if isinstance(group, Mapping):
            raws.extend(raw for raw in group.get("questions", []) if isinstance(raw, Mapping))
        for raw in raws:
            if (raw.get("my_forecasts") or {}).get("latest") and raw.get("id") is not None:
                standing.add(int(raw["id"]))
        return standing

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One request, with a single polite retry on 429.

        Exactly one retry, honouring Retry-After up to a cap: a loop that retries
        forever turns a server-side problem into a silent hang, and a bot that
        hangs during a submission window misses it without reporting anything.
        """
        response = self._http.request(method, url, **kwargs)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 30.0
            time.sleep(min(delay, MAX_RETRY_AFTER_SECONDS))
            response = self._http.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def submit(self, payloads: Sequence[Mapping[str, Any]]) -> None:
        """Submit one or more forecasts. Raises on any non-2xx: an unsubmitted
        forecast must never look like a submitted one."""
        if not payloads:
            return
        response = self._http.post(f"{API}/questions/forecast/", json=list(payloads))
        response.raise_for_status()

    def comment(self, post_id: int, text: str) -> None:
        """Attach the bot's reasoning to a post as a private comment. Tournament
        rules ask for reasoning; private keeps it out of other bots' retrieval."""
        response = self._http.post(
            f"{API}/comments/create/",
            json={
                "on_post": post_id,
                "text": text,
                "is_private": True,
                "included_forecast": True,
            },
        )
        response.raise_for_status()
