"""Results by mail: every resolution the bot answered, its score, and the running total.

Asked for on 2026-09-27. The payloads below are trimmed from the live API's
answers for the 2026-08-24 MiniBench, the only cycle this account has scores
in: post 45276 (binary, answered), 45282 (discrete, answered) and 45305
(resolved, unanswered, and its outcome hidden from this account).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import httpx
import pytest

from bot import results as br
from bot.venues.metaculus import MetaculusClient, Resolution, parse_resolutions

NOW = dt.datetime(2026, 10, 2, 12, 0, tzinfo=dt.UTC)
PROJECT = {"id": 33125, "slug": "minibench", "type": "question_series"}


def _post(post_id: int, question: dict[str, Any], project: dict[str, Any] = PROJECT) -> dict:
    return {
        "id": post_id,
        "title": question.get("title", f"Post {post_id}"),
        "projects": {"default_project": project},
        "question": question,
    }


def _binary(
    qid: int,
    *,
    resolution: str | None = "no",
    p_yes: float | None = 0.32,
    spot_peer: float | None = 62.04477485472652,
    status: str = "resolved",
    resolved_at: str = "2026-10-02T09:00:00Z",
) -> dict[str, Any]:
    answered = p_yes is not None
    scores = {"spot_peer_score": spot_peer, "spot_baseline_score": 40.0} if spot_peer else {}
    return {
        "id": qid,
        "title": f"Will thing {qid} happen?",
        "type": "binary",
        "status": status,
        "resolution": resolution,
        "resolution_set_time": resolved_at,
        "options": None,
        "scaling": {"range_min": None, "range_max": None, "zero_point": None},
        "my_forecasts": {
            "latest": {"forecast_values": [1 - p_yes, p_yes]} if answered else {},
            "score_data": scores,
            "history": [],
        },
    }


# The shape of 45282 on 2026-09-27: discrete, resolved 7719, the bot's median at
# 0.4857 on a 7187.5..8212.5 axis.
DISCRETE_45282 = {
    "id": 45472,
    "title": "What will the S&P 500 index close at on September 4?",
    "type": "discrete",
    "status": "resolved",
    "resolution": "7719.0",
    "resolution_set_time": "2026-09-05T18:52:49.386450Z",
    "options": None,
    "scaling": {"range_min": 7187.5, "range_max": 8212.5, "zero_point": None},
    "my_forecasts": {
        "latest": {"forecast_values": [0.001, 0.5, 0.999], "centers": [0.4856595541120198]},
        "score_data": {"spot_peer_score": 1.7927962125813284, "spot_baseline_score": 35.3},
        "history": [{}],
    },
}


# ------------------------------------------------------------ the venue parse


def test_an_answered_binary_reads_its_outcome_forecast_and_score() -> None:
    (r,) = parse_resolutions(_post(45276, _binary(45466)))
    assert (r.post_id, r.question_id, r.qtype, r.resolution) == (45276, 45466, "binary", "no")
    assert r.answered and r.forecast == pytest.approx(0.32)  # forecast_values is [no, yes]
    assert r.spot_peer == pytest.approx(62.04, abs=0.01)
    assert (r.project_id, r.project) == (33125, "minibench")


def test_a_numeric_forecast_is_read_back_as_its_median_in_real_units() -> None:
    (r,) = parse_resolutions(_post(45282, DISCRETE_45282))
    assert r.forecast == pytest.approx(7187.5 + 0.4856595541120198 * 1025.0)
    assert r.resolution == "7719.0"


def test_an_unanswered_question_reads_as_unanswered_with_its_outcome_unknown() -> None:
    """45305 had ~150 forecasters and a real outcome; this account sees None."""
    (r,) = parse_resolutions(_post(45305, _binary(45495, resolution=None, p_yes=None)))
    assert not r.answered and r.forecast is None and r.resolution is None


def test_only_resolved_questions_are_parsed_including_inside_groups() -> None:
    group = {
        "id": 43325,
        "title": "Net worth thresholds",
        "projects": {"default_project": PROJECT},
        "group_of_questions": {
            "questions": [_binary(1, status="open"), _binary(2), _binary(3, status="closed")]
        },
    }
    assert [r.question_id for r in parse_resolutions(group)] == [2]
    assert parse_resolutions(_post(9, _binary(10, status="closed"))) == []


def test_the_resolved_listing_stops_at_an_empty_page_and_never_runs_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listing's `next` link never ends (see next_page_url)."""
    import time

    monkeypatch.setattr(time, "sleep", lambda _: None)
    pages = [
        {
            "results": [{"id": 1}, {"id": 2}],
            "next": "https://www.metaculus.com/api/posts/?offset=2",
        },
        {"results": [], "next": "https://www.metaculus.com/api/posts/?offset=4"},
    ]
    seen: list[str] = []

    def wire(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=pages[min(len(seen) - 1, 1)])

    client = MetaculusClient("token", client=httpx.Client(transport=httpx.MockTransport(wire)))
    assert client.resolved_post_ids("minibench") == [1, 2]
    assert len(seen) == 2
    assert "statuses=resolved" in seen[0] and "tournaments=minibench" in seen[0]

    endless = {"results": [{"id": 7}], "next": "https://www.metaculus.com/api/posts/?offset=9"}
    runaway = MetaculusClient(
        "token",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=endless))
        ),
    )
    assert len(runaway.resolved_post_ids("minibench")) == 20  # MAX_RESOLVED_PAGES, then stop


# ------------------------------------------------------------ the check


class FakeResults:
    """The two reads the check needs, and a count of how often it made them."""

    def __init__(self, posts: dict[int, dict[str, Any]]) -> None:
        self.posts = posts
        self.listings = 0
        self.details: list[int] = []

    def resolved_post_ids(self, tournament: str) -> list[int]:
        self.listings += 1
        return list(self.posts)

    def resolutions(self, post_id: int) -> list[Resolution]:
        self.details.append(post_id)
        return parse_resolutions(self.posts[post_id])


class Outbox:
    def __init__(self, fail: int = 0) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, subject: str, body: str) -> None:
        if self.fail:
            self.fail -= 1
            raise OSError("smtp.gmail.com: connection timed out")
        self.sent.append((subject, body))


def _check(client: FakeResults, outbox: Outbox, tmp_path: Path, now: dt.datetime) -> list[str]:
    return br.check_results("minibench", client, outbox, now=now, state_dir=tmp_path)


def test_a_resolution_is_mailed_once_with_its_outcome_answer_score_and_total(
    tmp_path: Path,
) -> None:
    client = FakeResults(
        {
            45276: _post(45276, _binary(45466)),
            45282: _post(45282, DISCRETE_45282),
            45305: _post(45305, _binary(45495, resolution=None, p_yes=None)),
        }
    )
    outbox = Outbox()

    assert _check(client, outbox, tmp_path, NOW) == [outbox.sent[0][0]]

    subject, body = outbox.sent[0]
    assert (
        subject
        == "devinjones-bot: 3 results in minibench, +31.9 spot peer on average over 2, 1 MISSED"
    )
    assert "Outcome: no. The bot said 32% yes." in body
    assert "Spot peer: +62.0" in body
    assert "Outcome: 7,719. The bot's median: 7,685." in body
    assert "https://www.metaculus.com/questions/45276/" in body
    assert "MISSED" in body and "Will thing 45495 happen?" in body
    # Every number carries its N and a standard error (rule 5).
    assert "So far in minibench: 2 questions scored, total +63.8, average +31.9" in body
    assert "standard error 30.1" in body
    assert "1 missed" in body

    # Nothing new: no mail, and within the hour, not even a read.
    later = NOW + dt.timedelta(hours=2)
    assert _check(client, outbox, tmp_path, later) == []
    assert _check(client, outbox, tmp_path, later + dt.timedelta(minutes=20)) == []
    assert len(outbox.sent) == 1 and client.listings == 2
    assert sorted(client.details) == [45276, 45282, 45305]  # each post read once


def test_a_long_run_of_misses_is_counted_in_full_but_named_only_twenty_times(
    tmp_path: Path,
) -> None:
    """The 08-24 cycle, read live on 2026-09-27: 10 answered, 50 missed."""
    posts = {pid: _post(pid, _binary(pid + 1000, resolution=None, p_yes=None)) for pid in range(50)}
    outbox = Outbox()
    _check(FakeResults(posts), outbox, tmp_path, NOW)
    subject, body = outbox.sent[0]
    assert subject.endswith("50 MISSED")
    assert body.count("https://www.metaculus.com/questions/") == 20
    assert "... and 30 more." in body and "50 missed" in body


def test_a_score_that_has_not_landed_waits_and_then_arrives_in_its_own_mail(
    tmp_path: Path,
) -> None:
    pending = _post(45276, _binary(45466, spot_peer=None, resolved_at="2026-10-02T11:30:00Z"))
    client = FakeResults({45276: pending})
    outbox = Outbox()

    assert _check(client, outbox, tmp_path, NOW) == []  # resolved, score not yet posted

    client.posts[45276] = _post(45276, _binary(45466, spot_peer=12.5))
    assert len(_check(client, outbox, tmp_path, NOW + dt.timedelta(hours=1))) == 1
    assert "Spot peer: +12.5" in outbox.sent[0][1]


def test_a_score_that_never_lands_is_reported_as_unscored_not_dropped(tmp_path: Path) -> None:
    stuck = _post(45276, _binary(45466, spot_peer=None, resolved_at="2026-09-29T09:00:00Z"))
    outbox = Outbox()
    assert len(_check(FakeResults({45276: stuck}), outbox, tmp_path, NOW)) == 1
    subject, body = outbox.sent[0]
    assert "No score from Metaculus 2 days after it resolved" in body
    assert "no scored questions yet" in body and "1 annulled or unscored" in body


def test_an_annulled_question_is_said_not_to_count(tmp_path: Path) -> None:
    annulled = _post(45276, _binary(45466, resolution="annulled", spot_peer=None))
    outbox = Outbox()
    _check(FakeResults({45276: annulled}), outbox, tmp_path, NOW)
    assert "Annulled: it doesn't count." in outbox.sent[0][1]


def test_a_mail_that_fails_is_sent_again_at_the_next_check(tmp_path: Path) -> None:
    """Results arrive once. One the mail server refused must not be marked as told."""
    client = FakeResults({45276: _post(45276, _binary(45466))})
    outbox = Outbox(fail=1)

    assert _check(client, outbox, tmp_path, NOW) == []
    assert _check(client, outbox, tmp_path, NOW + dt.timedelta(minutes=20)) == []  # hourly
    assert len(_check(client, outbox, tmp_path, NOW + dt.timedelta(hours=1))) == 1
    assert "Spot peer: +62.0" in outbox.sent[0][1]


def test_each_cycle_keeps_its_own_total(tmp_path: Path) -> None:
    """MiniBench reuses its slug cycle after cycle; one average across two
    cycles would describe neither."""
    old = {"id": 1, "slug": "minibench-2026-08-24"}
    client = FakeResults({45276: _post(45276, _binary(45466, spot_peer=50.0), project=old)})
    outbox = Outbox()
    _check(client, outbox, tmp_path, NOW)

    client.posts = {45900: _post(45900, _binary(46000, spot_peer=-10.0))}
    _check(client, outbox, tmp_path, NOW + dt.timedelta(hours=1))

    body = outbox.sent[1][1]
    assert "So far in minibench: 1 question scored, total -10.0" in body
    assert "minibench-2026-08-24" not in body  # the old cycle is not in this mail
