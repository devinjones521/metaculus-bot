"""Funding alerts: one email per change of level, and never a false alarm.

The owner means to run the whole season on donated credit. The failure these
tests guard against is the pollers parking at the floor for days with nobody
told, and the opposite one: an inbox that learns to ignore the bot.
"""

from __future__ import annotations

from pathlib import Path

from bot import alerts as ba
from bot.poll import affordable_questions

WALLETS = ba.funding_wallets(min_credit=8.5, cost_per_question=1.15)
# donated:  empty below 8.50 + 1.15 = 9.65;  low below 8.50 + 25 x 1.15 = 37.25
# personal: empty below 0.05 + 0.03 = 0.08;  low below 0.05 + 25 x 0.03 = 0.80


class Outbox:
    """A mail server that records what it accepted, and can refuse the first N."""

    def __init__(self, fail: int = 0) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, subject: str, body: str) -> None:
        if self.fail:
            self.fail -= 1
            raise OSError("smtp.gmail.com: connection timed out")
        self.sent.append((subject, body))


def test_levels() -> None:
    donated = WALLETS["donated"]
    assert ba.level(None, donated) is None  # unknown is not empty
    assert ba.level(99.07, donated) == ba.OK
    assert ba.level(30.0, donated) == ba.LOW
    assert ba.level(9.0, donated) == ba.OUT


def test_empty_means_exactly_what_makes_the_poller_park() -> None:
    """If these two ever disagree, the bot parks without an email, or the owner is
    told it stopped while it is still forecasting."""
    donated = WALLETS["donated"]
    for balance in (5.04, 9.0, 9.64, 9.66, 10.0, 12.0, 37.0):
        cap = affordable_questions(balance, 8.5, 1.15)
        parks = cap is not None and cap < 1
        assert (ba.level(balance, donated) == ba.OUT) == parks, balance


def test_a_low_balance_mails_once_and_the_next_ticks_are_quiet(tmp_path: Path) -> None:
    outbox = Outbox()
    for _ in range(3):
        ba.check_funding({"donated": 30.0}, WALLETS, outbox, state_path=tmp_path / "a.json")
    ((subject, _body),) = outbox.sent
    assert "LOW" in subject and "$30.00" in subject
    assert "~18 questions left" in subject  # (30.00 - 8.50) // 1.15


def test_every_change_of_level_mails_including_the_top_up(tmp_path: Path) -> None:
    outbox = Outbox()
    for balance in (50.0, 30.0, 30.0, 9.0, 9.0, 99.0):
        ba.check_funding({"donated": balance}, WALLETS, outbox, state_path=tmp_path / "a.json")
    subjects = [subject for subject, _ in outbox.sent]
    assert len(subjects) == 3
    assert "LOW" in subjects[0]
    assert "EMPTY" in subjects[1]
    assert "funded again" in subjects[2]


def test_two_pollers_on_one_key_send_one_mail_between_them(tmp_path: Path) -> None:
    """Both pollers read the same key and share the state file, as does a restart."""
    minibench, fall = Outbox(), Outbox()
    ba.check_funding({"donated": 9.0}, WALLETS, minibench, state_path=tmp_path / "a.json")
    ba.check_funding({"donated": 9.0}, WALLETS, fall, state_path=tmp_path / "a.json")
    assert len(minibench.sent) == 1
    assert fall.sent == []


def test_a_failed_send_is_retried_next_tick_not_forgotten(tmp_path: Path) -> None:
    outbox = Outbox(fail=1)
    ba.check_funding({"donated": 9.0}, WALLETS, outbox, state_path=tmp_path / "a.json")
    assert outbox.sent == []
    ba.check_funding({"donated": 9.0}, WALLETS, outbox, state_path=tmp_path / "a.json")
    assert len(outbox.sent) == 1


def test_an_unreadable_balance_never_mails(tmp_path: Path) -> None:
    """One bad OpenRouter response must not say 'funded again', or 'empty'."""
    outbox = Outbox()
    ba.check_funding({"donated": 30.0}, WALLETS, outbox, state_path=tmp_path / "a.json")
    ba.check_funding(
        {"donated": None, "personal": None}, WALLETS, outbox, state_path=tmp_path / "a.json"
    )
    assert len(outbox.sent) == 1


def test_the_two_keys_are_judged_separately(tmp_path: Path) -> None:
    outbox = Outbox()
    ba.check_funding(
        {"donated": 99.0, "personal": 0.5}, WALLETS, outbox, state_path=tmp_path / "a.json"
    )
    ((subject, body),) = outbox.sent
    assert "personal key (Grok)" in subject and "LOW" in subject
    assert "four models" in body  # what empty means for THIS key: not a stop


def test_the_empty_donated_mail_says_forecasting_has_stopped() -> None:
    subject, body = ba.Alert(WALLETS["donated"], ba.LOW, ba.OUT, 9.0).render()
    assert "EMPTY" in subject
    assert "NO forecasts" in body


def test_a_garbled_state_file_reads_as_nothing_mailed(tmp_path: Path) -> None:
    state = tmp_path / "a.json"
    state.write_text("{not json", encoding="utf-8")
    assert ba.read_state(state) == {}
