"""Health verdicts and the box wire format, tested without a box.

The expensive failure this command exists for is `silent`: a unit that reports
`active` while doing nothing. Every severity is pinned here, plus the one that is
new to this repo, `parked`, which must never be mistaken for `silent`.
"""

from __future__ import annotations

import datetime as dt

from bot.health import (
    CRASHLOOP,
    FLAPPING,
    OK,
    OVERDUE,
    PARKED,
    POLLERS,
    SILENT,
    UNKNOWN,
    compare_trees,
    judge,
)
from bot.venues.box import parse

NOW = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)
POLLER = POLLERS[0]


def _beat(minutes_ago: float, state: str = "sweeping", credit: float | None = 60.0) -> dict:
    ts = (NOW - dt.timedelta(minutes=minutes_ago)).isoformat()
    return {"ts": ts, "tournament": POLLER.tournament, "state": state, "credit": credit}


def _judge(beat: dict | None, active: str | None = "active", restarts: int = 0) -> str:
    return judge(POLLER, active, restarts, beat, NOW.timestamp()).severity


class TestJudge:
    def test_a_fresh_tick_is_ok(self) -> None:
        assert _judge(_beat(5)) == OK

    def test_active_but_not_ticking_is_silent(self) -> None:
        """The failure that hid for two days in free-money."""
        assert _judge(_beat(120)) == SILENT

    def test_active_with_no_heartbeat_file_is_silent(self) -> None:
        assert _judge(None) == SILENT

    def test_a_garbled_timestamp_is_silent_not_ok(self) -> None:
        assert _judge({"ts": "not a time", "state": "sweeping"}) == SILENT

    def test_parked_at_the_floor_is_not_silent(self) -> None:
        """Parked writes one log row then nothing for days; the heartbeat keeps
        moving, and that is the only thing that separates it from dead."""
        assert _judge(_beat(5, state="parked", credit=7.0)) == PARKED

    def test_parked_but_not_ticking_is_still_silent(self) -> None:
        """Parked is a state of a LIVE loop. A stale parked beat is a dead loop."""
        assert _judge(_beat(200, state="parked")) == SILENT

    def test_inside_grace_is_overdue_not_silent(self) -> None:
        assert _judge(_beat(45)) == OVERDUE

    def test_a_stopped_unit_is_a_crashloop(self) -> None:
        assert _judge(_beat(5), active="failed", restarts=4) == CRASHLOOP

    def test_ticking_after_restarts_is_flapping(self) -> None:
        assert _judge(_beat(5), restarts=2) == FLAPPING

    def test_a_unit_the_box_does_not_have_is_unknown(self) -> None:
        assert _judge(_beat(5), active=None) == UNKNOWN


class TestParse:
    def test_round_trips_every_record_kind(self) -> None:
        text = (
            "unit\tmetaculus-poll@minibench\tactive\t0\n"
            'beat\tminibench\t{"ts": "2026-09-21T11:55:00+00:00", "state": "sweeping"}\n'
            "beat\tfall-futureeval-2026\t\n"
            'poll\t{"submitted": 2, "tournament": "minibench"}\n'
            "submitted\t7\n"
            "hash\tbot/poll.py\tabc123\n"
            "now\t1789992000\n"
        )
        state = parse(text)
        assert state.units["metaculus-poll@minibench"] == ("active", 0)
        assert state.beats["minibench"] == {
            "ts": "2026-09-21T11:55:00+00:00",
            "state": "sweeping",
        }
        assert state.beats["fall-futureeval-2026"] is None  # no file yet: None, not {}
        assert state.last_poll == {"submitted": 2, "tournament": "minibench"}
        assert state.submitted == 7
        assert state.hashes == {"bot/poll.py": "abc123"}
        assert state.box_epoch == 1789992000.0

    def test_a_half_written_heartbeat_is_none_not_a_crash(self) -> None:
        state = parse('beat\tminibench\t{"ts": "2026-09\n')
        assert state.beats["minibench"] is None


def test_drift_reports_missing_and_stale_but_not_extra() -> None:
    drift = compare_trees({"a.py": "1", "b.py": "2"}, {"b.py": "X", "old.py": "9"})
    assert drift.missing == ["a.py"]
    assert drift.differing == ["b.py"]
    assert drift.extra == ["old.py"]
    assert not drift.ok
