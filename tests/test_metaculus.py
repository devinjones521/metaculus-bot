"""Tests for the Metaculus venue client.

Each test is named for the specific failure it forbids. The theme, as everywhere
in this repo, is the failure that looks like success: a forecast that silently
went to the wrong question, got clamped to a probability nobody chose, or never
went anywhere at all while the run reported clean.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from bot.venues.metaculus import (
    CDF_POINTS,
    MetaculusClient,
    binary_payload,
    multiple_choice_payload,
    next_page_url,
    numeric_payload,
    only_open,
    parse_posts,
)

BINARY_POST = {
    "id": 45163,
    "title": "Will the 2026 Atlantic hurricane season have at least 3 named storms?",
    "question": {
        "id": 44120,
        "type": "binary",
        "title": "Will the 2026 Atlantic hurricane season have at least 3 named storms?",
        "description": "Some background.",
        "resolution_criteria": "Resolves YES if NHC names a third storm.",
        "fine_print": "Small print.",
        "open_time": "2026-08-10T14:00:00Z",
        "scheduled_close_time": "2026-08-22T14:00:00Z",
    },
}

MC_POST = {
    "id": 45200,
    "title": "What state will have the hottest temperature?",
    "question": {
        "id": 44160,
        "type": "multiple_choice",
        "options": ["Texas", "Arizona", "Nevada", "Other"],
        "scheduled_close_time": "2026-08-31T14:00:00Z",
    },
}

NUMERIC_POST = {
    "id": 45210,
    "title": "What will Brent close at?",
    "question": {
        "id": 44170,
        "type": "numeric",
        "unit": "$/bbl",
        "open_upper_bound": True,
        "open_lower_bound": False,
        "scaling": {"range_min": 55.0, "range_max": 130.0, "zero_point": None},
    },
}

GROUP_POST = {
    "id": 45220,
    "title": "Group: named storms by month",
    "group_of_questions": {
        "questions": [
            {"id": 44180, "type": "binary", "title": "By September?"},
            {"id": 44181, "type": "binary", "title": "By October?"},
        ]
    },
}

NOTEBOOK_POST = {"id": 45230, "title": "Tournament announcement", "notebook": {"id": 9}}


def test_parse_flattens_single_group_and_skips_notebooks() -> None:
    """A notebook parsed as a question would get a forecast POSTed at nothing;
    a group post parsed as one question would drop its siblings silently."""
    payload = {"results": [BINARY_POST, MC_POST, NUMERIC_POST, GROUP_POST, NOTEBOOK_POST]}
    questions = parse_posts(payload)
    assert [q.question_id for q in questions] == [44120, 44160, 44170, 44180, 44181]
    assert all(q.post_id for q in questions)


def test_parse_keeps_server_type_string_verbatim() -> None:
    """An unknown type must arrive unmodified so the orchestrator can refuse it,
    rather than being coerced into a family it does not belong to."""
    post = {"id": 1, "question": {"id": 2, "type": "date"}}
    (question,) = parse_posts({"results": [post]})
    assert question.qtype == "date"


def test_parse_reads_numeric_scaling() -> None:
    (question,) = parse_posts({"results": [NUMERIC_POST]})
    assert question.range_min == 55.0
    assert question.range_max == 130.0
    assert question.open_upper_bound is True
    assert question.open_lower_bound is False
    assert question.unit == "$/bbl"


def test_binary_payload_shape() -> None:
    payload = binary_payload(44120, 0.58)
    assert payload == {
        "question": 44120,
        "probability_yes": 0.58,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }


@pytest.mark.parametrize("p", [0.0, 0.005, 0.995, 1.0, -0.2, 1.7])
def test_binary_payload_refuses_tails_rather_than_clamping(p: float) -> None:
    """A clamped probability is a forecast nobody made. Refusing is loud;
    clamping would quietly rewrite the bot's judgment at the wire."""
    with pytest.raises(ValueError):
        binary_payload(44120, p)


def test_mc_payload_requires_exact_option_match() -> None:
    """A missing or extra key means the caller answered a different question —
    the server may even accept it, which is exactly why it must not leave here."""
    options = ["Texas", "Arizona", "Nevada", "Other"]
    good = {"Texas": 0.2, "Arizona": 0.4, "Nevada": 0.15, "Other": 0.25}
    assert multiple_choice_payload(44160, good, options)["probability_yes_per_category"] == {
        "Texas": 0.2,
        "Arizona": 0.4,
        "Nevada": 0.15,
        "Other": 0.25,
    }
    with pytest.raises(ValueError):
        multiple_choice_payload(44160, {k: v for k, v in good.items() if k != "Other"}, options)
    with pytest.raises(ValueError):
        multiple_choice_payload(44160, {**good, "Utah": 0.0}, options)


def test_mc_payload_requires_probabilities_that_sum_to_one() -> None:
    options = ["A", "B"]
    with pytest.raises(ValueError):
        multiple_choice_payload(1, {"A": 0.5, "B": 0.4}, options)


def test_numeric_payload_validates_cdf() -> None:
    rising = [i / (CDF_POINTS - 1) for i in range(CDF_POINTS)]
    assert numeric_payload(44170, rising)["continuous_cdf"] == rising
    with pytest.raises(ValueError):
        numeric_payload(44170, rising[:-1])  # wrong length
    broken = list(rising)
    broken[100], broken[101] = broken[101], broken[100]
    with pytest.raises(ValueError):
        numeric_payload(44170, broken)  # decreasing step
    with pytest.raises(ValueError):
        numeric_payload(44170, [v * 1.5 for v in rising])  # exceeds 1


def test_resolved_members_of_an_open_group_are_filtered() -> None:
    """Observed live 2026-08-18: post 43325 was open while two of its four
    sub-questions were resolved; forecasting those returned HTTP 405. The
    post-level statuses=open filter cannot see this — the question's own
    status decides."""
    group = {
        "id": 43325,
        "title": "Net worth thresholds",
        "group_of_questions": {
            "questions": [
                {"id": 43330, "type": "binary", "status": "open"},
                {"id": 43327, "type": "binary", "status": "resolved"},
                {"id": 43329, "type": "binary", "status": "open"},
                {"id": 43328, "type": "binary", "status": "resolved"},
            ]
        },
    }
    questions = only_open(parse_posts({"results": [group]}))
    assert [q.question_id for q in questions] == [43330, 43329]


def test_empty_page_ends_pagination_even_when_next_is_set() -> None:
    """Observed live 2026-08-18: an empty `statuses=open` page still carries a
    `next` URL, at every offset, forever. Following it walked to offset 800 and
    into the rate limiter. Empty results end the walk, whatever `next` says."""
    assert next_page_url({"results": [], "next": "https://x/api/posts/?offset=3"}) is None


def test_populated_page_paginates_and_last_page_stops() -> None:
    assert next_page_url({"results": [{"id": 1}], "next": "https://x/next"}) == "https://x/next"
    assert next_page_url({"results": [{"id": 1}], "next": None}) is None


def test_empty_page_parses_to_no_questions() -> None:
    """Prove the null can fire: an empty tournament reads as zero questions,
    not as a crash and not as fabricated rows."""
    assert parse_posts({"results": []}) == []


# ---------------------------------------------------------------------------
# The wire dying mid-request. On 2026-09-21 four MiniBench questions hit
# `RemoteProtocolError: Server disconnected without sending a response`, which
# carries no status code and so walked straight past the Retry-After path. One
# of them (q45942) died on the COMMENT, after its forecast had landed: a
# blind resend of a write is the one thing that must never happen, and doing
# nothing left a forecast on the board with no reasoning attached.


def _dropped(_: httpx.Request) -> httpx.Response:
    raise httpx.RemoteProtocolError("Server disconnected without sending a response.")


class _Wire:
    """A scripted transport: one handler per path, called in order, recording
    every attempt so a test can count resends rather than infer them."""

    def __init__(self, routes: dict[str, list[Any]]) -> None:
        self.routes = {path: list(responses) for path, responses in routes.items()}
        self.attempts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.attempts.append(f"{request.method} {path}")
        queue = self.routes.get(path)
        if not queue:
            raise AssertionError(f"unscripted call: {request.method} {path}")
        step = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(step):
            return step(request)
        return step

    def count(self, method: str, path: str) -> int:
        return self.attempts.count(f"{method} {path}")

    def client(self) -> MetaculusClient:
        return MetaculusClient("token", client=httpx.Client(transport=httpx.MockTransport(self)))


def _post_with_standing(question_id: int, standing: bool) -> httpx.Response:
    latest = {"start_time": 1789989804.0} if standing else None
    return httpx.Response(
        200, json={"id": 45757, "question": {"id": question_id, "my_forecasts": {"latest": latest}}}
    )


def test_a_submission_lost_on_the_wire_is_not_resent_once_the_server_shows_it_landed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The forecast is in, only the response was lost. A resend here would put a
    second forecast on a question this project allows exactly one on."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire(
        {
            "/api/questions/forecast/": [_dropped],
            "/api/posts/45757/": [_post_with_standing(45942, standing=True)],
        }
    )
    wire.client().submit([binary_payload(45942, 0.3)], post_id=45757)
    assert wire.count("POST", "/api/questions/forecast/") == 1
    assert wire.count("GET", "/api/posts/45757/") == 1


def test_a_submission_lost_on_the_wire_is_resent_when_nothing_landed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire(
        {
            "/api/questions/forecast/": [_dropped, httpx.Response(201, json={})],
            "/api/posts/45757/": [_post_with_standing(45942, standing=False)],
        }
    )
    wire.client().submit([binary_payload(45942, 0.3)], post_id=45757)
    assert wire.count("POST", "/api/questions/forecast/") == 2


def test_a_submission_is_never_resent_blind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a post id the server cannot be asked what it holds, so the error
    is raised. A dead question is recoverable next sweep; a double forecast is not."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire({"/api/questions/forecast/": [_dropped]})
    with pytest.raises(httpx.RemoteProtocolError):
        wire.client().submit([binary_payload(45942, 0.3)])
    assert wire.count("POST", "/api/questions/forecast/") == 1


def test_a_second_drop_raises_rather_than_looping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire(
        {
            "/api/questions/forecast/": [_dropped],
            "/api/posts/45757/": [_post_with_standing(45942, standing=False)],
        }
    )
    with pytest.raises(httpx.RemoteProtocolError):
        wire.client().submit([binary_payload(45942, 0.3)], post_id=45757)
    assert wire.count("POST", "/api/questions/forecast/") == 2


def test_a_comment_lost_on_the_wire_is_resent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicate private comment costs nothing. A forecast with no reasoning
    is not prize-eligible and no later sweep returns to it."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire({"/api/comments/create/": [_dropped, httpx.Response(201, json={"id": 1})]})
    wire.client().comment(45757, "the reasoning")
    assert wire.count("POST", "/api/comments/create/") == 2


def test_a_dropped_read_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads are idempotent, so this one is free: a dropped listing used to kill
    a whole sweep rather than one question."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    wire = _Wire(
        {"/api/posts/": [_dropped, httpx.Response(200, json={"results": [], "next": None})]}
    )
    assert wire.client().open_questions("minibench") == []
    assert wire.count("GET", "/api/posts/") == 2
