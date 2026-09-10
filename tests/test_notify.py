"""Openings and answers by mail: tell the owner what the bot did, and never cry wolf.

Asked for on 2026-09-10: "when tournaments start, when I answer and what I
answered". The failure modes are an inbox that is told nothing, and one that
is told the same thing so often it stops reading.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from bot import notify as bn
from bot.forecast import ForecastResult

NOW = dt.datetime(2026, 9, 21, 0, 20, tzinfo=dt.UTC)


def _result(qid: int, **overrides: Any) -> ForecastResult:
    base: dict[str, Any] = {
        "question_id": qid,
        "post_id": qid + 1000,
        "title": f"Will thing {qid} happen?",
        "qtype": "binary",
        "submitted": True,
        "runs_ok": 5,
        "answer": "30%",
    }
    base.update(overrides)
    return ForecastResult(**base)


class Outbox:
    def __init__(self, fail: int = 0) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, subject: str, body: str) -> None:
        if self.fail:
            self.fail -= 1
            raise OSError("smtp.gmail.com: connection timed out")
        self.sent.append((subject, body))


# ---------------------------------------------------------------- answers


def test_the_answer_mail_names_each_question_its_answer_and_its_link() -> None:
    thinned = _result(2, runs_ok=4, runs_failed=1, run_errors=["google/x: HTTP 503"])
    mail = bn.answers_email("minibench", [_result(1), thinned], NOW)
    assert mail is not None
    subject, body = mail
    assert subject == "devinjones-bot answered 2 questions in minibench"
    assert "Will thing 1 happen?" in body and "Answer: 30%" in body
    assert "https://www.metaculus.com/questions/1001/" in body
    assert "4/5 models answered" in body and "HTTP 503" in body  # a thinned ensemble says so


def test_a_sweep_that_did_nothing_mails_nothing() -> None:
    skipped = _result(1, submitted=False, error="skipped: already forecast")
    assert bn.answers_email("minibench", [skipped], NOW) is None
    assert bn.answers_email("minibench", [], NOW) is None


def test_a_failed_question_is_mailed_with_its_reason() -> None:
    dead = _result(3, submitted=False, runs_ok=0, error="RuntimeError: all 5 runs failed — 402")
    mail = bn.answers_email("minibench", [dead], NOW)
    assert mail is not None
    subject, body = mail
    assert "1 FAILED" in subject
    assert "all 5 runs failed" in body


# ---------------------------------------------------------------- openings


def test_the_first_sweep_after_a_deploy_records_a_baseline_and_says_nothing(
    tmp_path: Path,
) -> None:
    """The Fall practice question was open before the deploy; it is not news."""
    outbox = Outbox()
    practice = _result(45707, submitted=False, error="skipped: already forecast")
    bn.notify_sweep("fall", [practice], outbox, now=NOW, state_dir=tmp_path)
    assert outbox.sent == []
    assert bn.read_state(bn.state_path("fall", tmp_path)) == {"seen": [45707], "last_new": None}


def test_new_questions_after_a_quiet_spell_announce_the_tournament(tmp_path: Path) -> None:
    outbox = Outbox()
    bn.notify_sweep("minibench", [], outbox, now=NOW, state_dir=tmp_path)  # baseline: empty
    bn.notify_sweep("minibench", [_result(1), _result(2)], outbox, now=NOW, state_dir=tmp_path)
    subjects = [subject for subject, _ in outbox.sent]
    assert subjects[0] == "devinjones-bot: minibench is open, 2 new questions"
    assert subjects[1] == "devinjones-bot answered 2 questions in minibench"


def test_new_questions_inside_the_day_do_not_announce_again(tmp_path: Path) -> None:
    outbox = Outbox()
    bn.notify_sweep("minibench", [], outbox, now=NOW, state_dir=tmp_path)
    bn.notify_sweep("minibench", [_result(1)], outbox, now=NOW, state_dir=tmp_path)
    later = NOW + dt.timedelta(hours=3)
    bn.notify_sweep("minibench", [_result(1), _result(2)], outbox, now=later, state_dir=tmp_path)
    openings = [s for s, _ in outbox.sent if " is open" in s]
    assert len(openings) == 1  # the stream within a cycle is one opening, not sixty


def test_the_next_cycle_is_announced_after_a_quiet_week(tmp_path: Path) -> None:
    outbox = Outbox()
    bn.notify_sweep("minibench", [], outbox, now=NOW, state_dir=tmp_path)
    bn.notify_sweep("minibench", [_result(1)], outbox, now=NOW, state_dir=tmp_path)
    next_cycle = NOW + dt.timedelta(days=14)
    bn.notify_sweep("minibench", [_result(9)], outbox, now=next_cycle, state_dir=tmp_path)
    openings = [body for s, body in outbox.sent if " is open" in s]
    assert len(openings) == 2
    assert "the first in 14 days" in openings[1]


def test_each_tournament_keeps_its_own_memory(tmp_path: Path) -> None:
    """Two pollers writing one file would erase each other's baseline."""
    outbox = Outbox()
    bn.notify_sweep("minibench", [], outbox, now=NOW, state_dir=tmp_path)
    bn.notify_sweep("fall", [_result(7)], outbox, now=NOW, state_dir=tmp_path)
    assert bn.read_state(bn.state_path("minibench", tmp_path)) == {"seen": [], "last_new": None}
    assert bn.read_state(bn.state_path("fall", tmp_path)) == {"seen": [7], "last_new": None}


def test_a_failed_opening_mail_is_retried_on_the_next_sweep(tmp_path: Path) -> None:
    outbox = Outbox(fail=1)
    bn.notify_sweep("minibench", [], outbox, now=NOW, state_dir=tmp_path)
    fresh = [_result(1, submitted=False, error="skipped: already forecast")]
    bn.notify_sweep("minibench", fresh, outbox, now=NOW, state_dir=tmp_path)  # send fails
    assert outbox.sent == []
    bn.notify_sweep("minibench", fresh, outbox, now=NOW, state_dir=tmp_path)
    assert [s for s, _ in outbox.sent] == ["devinjones-bot: minibench is open, 1 new question"]


def test_a_garbled_state_file_is_a_baseline_not_a_crash(tmp_path: Path) -> None:
    path = bn.state_path("minibench", tmp_path)
    path.write_text("{not json", encoding="utf-8")
    outbox = Outbox()
    bn.notify_sweep("minibench", [_result(1)], outbox, now=NOW, state_dir=tmp_path)
    assert [s for s, _ in outbox.sent if " is open" in s] == []
