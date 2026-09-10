"""The poll loop, tested on the two things that actually cost money.

The 2026-08-24 MiniBench entry scored the best per-question rate in a field of
182 and earned nothing, because it ran twice in one hour and covered 10 of 58
questions. Two failure modes produced that, and both are tested here rather
than reasoned about:

  1. the loop stopping (or never continuing) while questions were still opening;
  2. the loop spending the key down past the ~$8 max_tokens floor, at which
     point every ensemble run 402s and the failures look like model failures.

`test_main_...` drives the real entry point, because the argument parsing and
the deadline arithmetic live there and nowhere else — the wiring between tested
pieces is this project's most repeated defect.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

from bot import poll as bp

START = dt.datetime(2026, 9, 7, 18, 0, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _no_writes_to_real_data(tmp_path: Any, monkeypatch: Any) -> None:
    """Every test's heartbeat and poll log go to a temp dir, whether it asks or not.

    Found 2026-09-10: once the loop gained a heartbeat, every older test that ran
    poll() wrote a fake-clock heartbeat into the repo's real data/metaculus/. A
    per-test patch is a thing the next test forgets; this is not.
    """
    monkeypatch.setattr(bp, "HEARTBEAT_DIR", tmp_path)
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")


class FakeSession:
    """Only what poll() touches: a balance and something to hand to a sweep."""

    def __init__(self, credits: list[float | None] | float | None = 100.0) -> None:
        self._credits = credits if isinstance(credits, list) else [credits]
        self.closed = False

    def credits_remaining(self) -> float | None:
        return self._credits[0] if len(self._credits) == 1 else self._credits.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    """A clock that only moves when the loop sleeps, or when a sweep runs."""

    def __init__(self, start: dt.datetime = START) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += dt.timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)


def _summary(submitted: int = 1) -> Any:
    from bot.forecast import SweepSummary

    return SweepSummary(
        seen=submitted, submitted=submitted, failed=0, cost_usd=1.0, credits_remaining=50.0
    )


def test_it_keeps_sweeping_until_the_deadline(monkeypatch: Any) -> None:
    """The whole point: one sweep is not an entry, it is a three-hour slice."""
    clock = FakeClock()
    calls: list[Any] = []
    monkeypatch.setattr(bp, "sweep_and_log", lambda *a, **k: (calls.append(k), _summary())[1])

    outcome = bp.poll(
        FakeSession(),
        "minibench",
        until=START + dt.timedelta(hours=2),
        interval_seconds=20 * 60,
        now=clock,
        sleep=clock.sleep,
    )

    assert len(calls) == 6  # 2 hours at 20-minute ticks
    assert outcome.sweeps == 6
    assert outcome.submitted == 6
    assert outcome.stopped_because == "deadline reached"


def test_the_credit_floor_stops_the_run_before_it_spends(tmp_path: Any, monkeypatch: Any) -> None:
    """$5.04 on the key is where the 08-24 run died mid-sweep, emitting five
    model failures per question. It must now stop, and say so, having spent
    nothing."""
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    swept: list[Any] = []
    monkeypatch.setattr(bp, "sweep_and_log", lambda *a, **k: (swept.append(k), _summary())[1])
    clock = FakeClock()

    outcome = bp.poll(
        FakeSession(credits=5.04),
        "minibench",
        until=START + dt.timedelta(hours=8),
        interval_seconds=20 * 60,
        now=clock,
        sleep=clock.sleep,
    )

    assert swept == []  # not one dollar spent into the wall
    assert outcome.sweeps == 0
    assert "credit floor" in outcome.stopped_because
    row = json.loads((tmp_path / "polls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["credits_remaining"] == 5.04
    assert "credit floor" in row["stopped"]


def test_each_sweep_is_capped_by_what_the_key_can_afford(monkeypatch: Any) -> None:
    """The cap is the guard that turns 'ran out mid-question' into 'stopped
    with credit to spare'."""
    seen: list[int | None] = []
    monkeypatch.setattr(
        bp, "sweep_and_log", lambda *a, **k: (seen.append(k["limit"]), _summary())[1]
    )
    clock = FakeClock()

    bp.poll(
        FakeSession(credits=[20.0, 12.0]),
        "minibench",
        until=START + dt.timedelta(minutes=40),
        interval_seconds=20 * 60,
        min_credit=8.5,
        cost_per_question=1.15,
        now=clock,
        sleep=clock.sleep,
    )

    # (20.00 - 8.50) / 1.15 = 10 questions;  (12.00 - 8.50) / 1.15 = 3
    assert seen == [10, 3]


def test_an_unreadable_balance_does_not_stop_a_three_day_run(monkeypatch: Any) -> None:
    """None means UNKNOWN. Treating one bad lookup as 'no money' would end the
    run on a network blip, which is the failure this module exists to prevent."""
    seen: list[int | None] = []
    monkeypatch.setattr(
        bp, "sweep_and_log", lambda *a, **k: (seen.append(k["limit"]), _summary())[1]
    )
    clock = FakeClock()

    outcome = bp.poll(
        FakeSession(credits=None),
        "minibench",
        until=START + dt.timedelta(minutes=40),
        interval_seconds=20 * 60,
        now=clock,
        sleep=clock.sleep,
    )

    assert seen == [None, None]
    assert outcome.stopped_because == "deadline reached"


def test_one_failed_sweep_never_ends_the_poll_and_is_recorded(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A silent gap in a three-day log is indistinguishable from a loop that
    was never scheduled. The row is the evidence that it was alive."""
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    attempts = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("connection reset by peer")
        return _summary()

    monkeypatch.setattr(bp, "sweep_and_log", flaky)
    clock = FakeClock()

    outcome = bp.poll(
        FakeSession(),
        "minibench",
        until=START + dt.timedelta(minutes=60),
        interval_seconds=20 * 60,
        now=clock,
        sleep=clock.sleep,
    )

    assert attempts["n"] == 3  # it kept going after the failure
    assert outcome.submitted == 2  # the two that worked
    row = json.loads((tmp_path / "polls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["error"] == "RuntimeError: connection reset by peer"
    assert row["failed"] == 1


def test_a_slow_sweep_does_not_push_the_next_one_further_out(monkeypatch: Any) -> None:
    """Fixed cadence, not fixed rest. A sweep that overruns the interval starts
    the next immediately; sleeping a further 20 minutes on top would drift
    further behind the question stream with every tick."""
    clock = FakeClock()

    def slow(*args: Any, **kwargs: Any) -> Any:
        clock.advance(25 * 60)  # the sweep itself outlasts the interval
        return _summary()

    monkeypatch.setattr(bp, "sweep_and_log", slow)

    bp.poll(
        FakeSession(),
        "minibench",
        until=START + dt.timedelta(hours=3),
        interval_seconds=20 * 60,
        now=clock,
        sleep=clock.sleep,
    )

    assert clock.slept == []  # already behind schedule; no rest was taken


def _rows(path: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_parked_poller_resumes_on_its_own_when_credit_arrives(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The service case. Metaculus tops the donated key up automatically on good
    MiniBench performance; a bot that exited at the floor would be dead when the
    money arrived, and one that restarted under systemd would look like a crash."""
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    monkeypatch.setattr(bp, "HEARTBEAT_DIR", tmp_path)
    swept: list[Any] = []
    monkeypatch.setattr(bp, "sweep_and_log", lambda *a, **k: (swept.append(k), _summary())[1])
    clock = FakeClock()

    outcome = bp.poll(
        FakeSession(credits=[5.0, 5.0, 50.0]),  # floor, floor, top-up, then 50 stays
        "minibench",
        until=START + dt.timedelta(minutes=80),  # ticks at 0, 20, 40, 60
        interval_seconds=20 * 60,
        wait_at_floor=True,
        now=clock,
        sleep=clock.sleep,
    )

    assert len(swept) == 2  # nothing while parked, then both remaining ticks
    assert outcome.stopped_because == "deadline reached"
    rows = _rows(tmp_path / "polls.jsonl")
    assert ["parked" in r for r in rows] == [True, False]
    assert "resumed" in rows[1] and rows[1]["credits_remaining"] == 50.0


def test_a_parked_episode_is_logged_once_not_every_tick(tmp_path: Any, monkeypatch: Any) -> None:
    """72 identical rows a day would bury the one that says when it started."""
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    monkeypatch.setattr(bp, "HEARTBEAT_DIR", tmp_path)
    monkeypatch.setattr(bp, "sweep_and_log", lambda *a, **k: _summary())
    clock = FakeClock()

    outcome = bp.poll(
        FakeSession(credits=5.0),
        "minibench",
        until=START + dt.timedelta(hours=2),
        interval_seconds=20 * 60,
        wait_at_floor=True,
        now=clock,
        sleep=clock.sleep,
    )

    assert outcome.sweeps == 0
    assert outcome.stopped_because == "deadline reached"  # parked, never exited
    assert len(_rows(tmp_path / "polls.jsonl")) == 1


def test_the_heartbeat_moves_every_tick_even_while_parked(tmp_path: Any, monkeypatch: Any) -> None:
    """A parked bot writes one log row and then nothing for days, which is what a
    dead one looks like. The heartbeat is what tells them apart."""
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    monkeypatch.setattr(bp, "HEARTBEAT_DIR", tmp_path)
    clock = FakeClock()

    bp.poll(
        FakeSession(credits=5.0),
        "minibench",
        until=START + dt.timedelta(hours=2),
        interval_seconds=20 * 60,
        wait_at_floor=True,
        now=clock,
        sleep=clock.sleep,
    )

    beat = json.loads((tmp_path / "heartbeat-minibench.json").read_text(encoding="utf-8"))
    assert beat["state"] == "parked"
    assert beat["ts"] == (START + dt.timedelta(minutes=100)).isoformat()  # the LAST tick
    assert beat["credit"] == 5.0


def test_main_passes_wait_at_floor_through_to_the_loop(monkeypatch: Any) -> None:
    """The flag lives in main(); a flag parsed and never forwarded is the wiring
    defect this project keeps finding between tested pieces."""
    seen: dict[str, Any] = {}

    def fake_poll(session: Any, tournament: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return bp.PollOutcome(0, 0, "deadline reached")

    monkeypatch.setattr(bp, "build_session", lambda dry_run: FakeSession())
    monkeypatch.setattr(bp, "poll", fake_poll)

    assert bp.main(["--hours", "1", "--wait-at-floor"]) == 0
    assert seen["wait_at_floor"] is True
    assert bp.main(["--hours", "1"]) == 0
    assert seen["wait_at_floor"] is False


def test_affordable_questions_distinguishes_zero_from_unknown() -> None:
    assert bp.affordable_questions(None, 8.5, 1.15) is None  # unknown
    assert bp.affordable_questions(9.0, 8.5, 1.15) == 0  # known, and it is zero
    assert bp.affordable_questions(150.0, 8.5, 1.15) == 123


@pytest.mark.parametrize(
    "argv",
    [
        ["--tournament", "minibench"],  # neither deadline given
        ["--until", "2026-09-10T06:56Z", "--hours", "4"],  # both given
        ["--hours", "-1"],  # deadline in the past
        ["--hours", "4", "--interval-minutes", "0"],
    ],
)
def test_main_refuses_a_poll_that_would_never_end_or_never_tick(argv: list[str]) -> None:
    assert bp.main(argv) == 2


def test_main_runs_the_real_loop_and_logs_every_sweep(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    """Driven through main(), because deadline parsing, the budget banner and
    the loop wiring live there and nowhere else."""
    from test_forecast import FakeClient, _question

    from bot import forecast as fc

    class FakeLlmClient:
        def __init__(self, key: str, **kwargs: Any) -> None:
            self.total_cost_usd = 0.0
            self.cost_by_model = {}

        def models(self) -> set[str]:
            return {"model-a", "model-r"}

        def complete(self, model: str, prompt: str, *, temperature: float = 1.0) -> str:
            self.total_cost_usd += 0.5
            self.cost_by_model["model-a"] = self.total_cost_usd
            if "research assistant" in prompt or "left gaps" in prompt:
                return "Nothing decisive. REMAINING GAPS: none."
            return "Reasoning...\nProbability: 30%"

        def key_limits(self) -> dict[str, Any]:
            return {"limit_remaining": 100.0}

        def close(self) -> None:
            return None

    class ClosableClient(FakeClient):
        def close(self) -> None:
            return None

    monkeypatch.setenv("METAC_FORECAST_MODELS", "model-a")
    monkeypatch.setenv("METAC_RESEARCH_MODEL", "model-r")
    monkeypatch.setattr(fc, "metaculus_token", lambda: "token")
    monkeypatch.setattr(fc, "openrouter_key", lambda: "key")
    monkeypatch.setattr(fc, "asknews_credentials", lambda: None)
    monkeypatch.setattr(fc, "MetaculusClient", lambda token: ClosableClient([_question()]))
    monkeypatch.setattr("bot.venues.llm.OpenRouterClient", FakeLlmClient)
    monkeypatch.setattr(fc, "FORECAST_LOG", tmp_path / "forecasts.jsonl")
    monkeypatch.setattr(fc, "POLL_LOG", tmp_path / "polls.jsonl")
    monkeypatch.setattr(bp, "POLL_LOG", tmp_path / "polls.jsonl")
    monkeypatch.setattr(bp.time, "sleep", lambda seconds: None)

    assert bp.main(["--tournament", "t", "--hours", "24", "--max-sweeps", "2"]) == 0

    polls = [
        json.loads(line)
        for line in (tmp_path / "polls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(polls) == 2, "every sweep writes its own row, or the log cannot prove liveness"
    # The second question is skipped as already forecast, so only sweep one submits.
    assert [p["submitted"] for p in polls] == [1, 0]
    # Cost is per sweep, not the client's running total: the second sweep spent
    # nothing, and a cumulative field here would report the first sweep's spend
    # again forever.
    assert polls[1]["llm_cost_usd"] == 0.0
    assert polls[1]["llm_cost_cumulative_usd"] == polls[0]["llm_cost_cumulative_usd"]
    assert "sweep limit reached" in capsys.readouterr().out
