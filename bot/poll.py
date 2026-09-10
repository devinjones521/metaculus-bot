"""Forecast the WHOLE cycle, not just the first morning.

    uv run python -m bot.poll --tournament minibench --until 2026-09-10T06:56Z
    uv run python -m bot.poll --tournament minibench --hours 12 --dry-run

WHY THIS EXISTS
---------------

Measured on the 2026-08-24 MiniBench cycle, and it is the most expensive
operational lesson the project has: `devinjones-bot` scored **+23.9 spot-peer
per question, the highest per-question rate in a field of 182**, and earned
**$0**. It covered 10 of the cycle's 58 questions.

MiniBench questions live for about **three hours each** and are released in a
rolling stream across the whole forecasting window — 36 questions in the first
17.8 hours of the 2026-09-07 cycle, roughly two an hour. `bot.forecast` sweeps
whatever is open at the moment it runs. Run once, it sees one three-hour slice
and reports a clean, complete, entirely successful sweep of it. The 08-24 entry
ran twice, both inside one hour on the first morning; 46 questions opened and
closed with nothing running.

The free-money project log blamed the OpenRouter budget floor for that stop. The floor was
real and it killed question twelve — but no amount of credit would have bought
the other 46, because nothing was polling. This module is the fix for the half
that money cannot buy.

Prize money makes the gap concrete. The pool is allocated in proportion to
**score squared**, with entries under $50 cut and redistributed upward, so
coverage compounds and there is a cliff: at our measured rate, 10 questions of
58 pays $0, 28 is a coin flip, and 58 pays about $197. Half a cycle is not half
the money — for a long stretch of the curve it is none of it.

HOW IT COULD LIE
----------------

- **A poll that sees nothing looks exactly like a poll that is broken.** Every
  sweep appends its own row to `data/metaculus/polls.jsonl`, and a sweep that
  raises appends a row carrying the exception text rather than vanishing. Rule
  6 applies to a quiet three-day log: prove the thing can fire.
- **Stopping silently is the same failure as never starting.** The loop exits
  only for a named reason, prints it, and returns it in `PollOutcome`, so
  "finished" can never be confused with "died on hour nine".
- **Spending into the wall wastes the last of the key.** Below roughly $8 the
  `max_tokens=8000` reservation makes every ensemble run 402, which turns a
  credit problem into what looks like five model failures. The floor is checked
  before each sweep and again as a per-sweep question cap, so the run stops
  while it still has forecasts left in it rather than emitting failures.
- **An unreadable balance is not a zero balance.** `credits_remaining()`
  returns None when OpenRouter will not say, and None means proceed with a
  warning — treating a transient lookup failure as "no money" would stop a
  three-day run dead on one bad response.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from bot.alerts import check_funding, funding_wallets
from bot.config import alert_mail_settings
from bot.forecast import (
    DATA_DIR,
    POLL_LOG,
    Session,
    SetupError,
    append_jsonl,
    build_session,
    sweep_and_log,
    use_utf8_stdio,
)

# Questions live ~3h, so any interval well under that sees every question with
# most of its life left. 20 minutes costs one listing call per idle tick.
DEFAULT_INTERVAL_MINUTES = 20.0

# Below this the max_tokens=8000 reservation 402s every ensemble run. Measured
# 2026-08-24: the sweep died with $6.16 and again with $5.95 on the key.
DEFAULT_MIN_CREDIT_USD = 8.50

# Measured 2026-08-24: $12.41 of OpenRouter bought 11 submitted questions.
DEFAULT_COST_PER_QUESTION_USD = 1.15

# One file per tournament, because the box runs one poller per tournament.
HEARTBEAT_DIR = DATA_DIR


@dataclass
class PollOutcome:
    sweeps: int
    submitted: int
    stopped_because: str


def write_heartbeat(tournament: str, state: str, credit: float | None, at: dt.datetime) -> None:
    """Overwrite this tournament's heartbeat file, once per tick, in every state.

    `polls.jsonl` cannot prove liveness on its own: a bot parked at the credit
    floor writes one row and then, correctly, nothing for days — which is
    exactly what a dead one looks like. The heartbeat's age is what `bot.health`
    judges; the log is what a post-mortem reads.

    Written to a temp file and renamed, so a reader never sees half a file.
    """
    path = HEARTBEAT_DIR / f"heartbeat-{tournament}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"ts": at.isoformat(), "tournament": tournament, "state": state, "credit": credit}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_deadline(text: str) -> dt.datetime:
    """ISO 8601, trailing Z allowed. Naive input is read as UTC."""
    stamp = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.UTC)


def affordable_questions(
    credit: float | None, min_credit: float, cost_per_question: float
) -> int | None:
    """How many questions this sweep may attempt. None means 'no known limit'.

    Zero is a real answer and means stop — not 'unlimited'. Keeping that
    distinct from None is the whole point of the return type.
    """
    if credit is None:
        return None
    return int((credit - min_credit) // cost_per_question)


def poll(
    session: Session,
    tournament: str,
    *,
    until: dt.datetime,
    interval_seconds: float,
    min_credit: float = DEFAULT_MIN_CREDIT_USD,
    cost_per_question: float = DEFAULT_COST_PER_QUESTION_USD,
    dry_run: bool = False,
    max_sweeps: int | None = None,
    wait_at_floor: bool = False,
    alert: Callable[[dict[str, float | None]], list[str]] | None = None,
    now: Callable[[], dt.datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> PollOutcome:
    """Sweep `tournament` every interval until the deadline, credit, or Ctrl-C.

    `wait_at_floor` is for the unattended service. Metaculus tops the donated
    key up automatically on good MiniBench performance, so under a supervisor
    "stop at the floor" is wrong twice over: `Restart=always` turns the clean
    exit into a restart loop that looks like a crash, and anything else leaves
    the bot dead when the money arrives. Parked, it re-reads the balance every
    interval and resumes on its own.
    """
    # Both resolved here rather than in the signature: a default bound at
    # definition time captures the real `time.sleep` object, and a test that
    # patches the module attribute then sleeps for twenty real minutes.
    now = now or (lambda: dt.datetime.now(dt.UTC))
    sleep = sleep or time.sleep
    sweeps = 0
    submitted = 0
    parked = False

    def row(**fields: object) -> dict[str, object]:
        base: dict[str, object] = {
            "ts": now().isoformat(),
            "tournament": tournament,
            "questions_seen": 0,
            "submitted": 0,
            "failed": 0,
            "dry_run": dry_run,
        }
        return {**base, **fields}

    while True:
        if now() >= until:
            return PollOutcome(sweeps, submitted, "deadline reached")
        if max_sweeps is not None and sweeps >= max_sweeps:
            return PollOutcome(sweeps, submitted, "sweep limit reached")

        tick_started = now()
        credit = session.credits_remaining()
        if alert is not None:
            # Every tick, parked or not: parked is exactly when the owner must hear.
            balances = {"donated": credit, "personal": session.personal_credits_remaining()}
            for subject in alert(balances):
                print(f"alert emailed: {subject}")
                append_jsonl(POLL_LOG, row(credits_remaining=credit, alert=subject))
        cap = affordable_questions(credit, min_credit, cost_per_question)
        if cap is not None and cap < 1:
            floor = min_credit + cost_per_question
            if not wait_at_floor:
                reason = (
                    f"credit floor: ${credit:.2f} left, under the ${floor:.2f} one more "
                    "question needs. Top up the OpenRouter key and restart."
                )
                print(reason)
                append_jsonl(POLL_LOG, row(credits_remaining=credit, stopped=reason))
                return PollOutcome(sweeps, submitted, reason)
            if not parked:
                # One row per episode, not per tick: 72 identical rows a day
                # would bury the one that says when it started.
                note = (
                    f"parked at the credit floor: ${credit:.2f} left, under the ${floor:.2f} "
                    "one more question needs. Re-checking every tick; resumes on top-up."
                )
                print(note)
                append_jsonl(POLL_LOG, row(credits_remaining=credit, parked=note))
                parked = True
            write_heartbeat(tournament, "parked", credit, tick_started)
        else:
            if parked:
                note = f"credit restored: ${credit:.2f} on the key — resuming."
                print(note)
                append_jsonl(POLL_LOG, row(credits_remaining=credit, resumed=note))
                parked = False
            if credit is None:
                print("note: OpenRouter balance unreadable this tick — sweeping without a cap.")
            write_heartbeat(tournament, "sweeping", credit, tick_started)
            sweeps += 1
            try:
                summary = sweep_and_log(
                    session, tournament, dry_run=dry_run, limit=cap, resubmit=False
                )
                submitted += summary.submitted
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - contain, record, keep polling
                # One bad sweep must not end a three-day run, and it must not be
                # invisible either: the row is what tells a post-mortem that the
                # loop was alive and failing rather than never scheduled.
                detail = f"{type(exc).__name__}: {exc}"
                print(f"  SWEEP FAILED: {detail}")
                append_jsonl(
                    POLL_LOG,
                    {
                        **row(credits_remaining=credit, failed=1, error=detail),
                        "ts": tick_started.isoformat(),
                    },
                )

        # Fixed cadence, not fixed rest: a sweep that overruns the interval
        # starts the next one immediately instead of drifting further behind
        # the question stream with every tick.
        next_at = tick_started + dt.timedelta(seconds=interval_seconds)
        if next_at >= until:
            return PollOutcome(sweeps, submitted, "deadline reached")
        gap = (next_at - now()).total_seconds()
        if gap > 0:
            sleep(gap)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tournament", default="minibench")
    parser.add_argument("--until", help="stop at this ISO 8601 instant, e.g. 2026-09-10T06:56Z")
    parser.add_argument("--hours", type=float, help="stop this many hours from now")
    parser.add_argument("--interval-minutes", type=float, default=DEFAULT_INTERVAL_MINUTES)
    parser.add_argument("--min-credit", type=float, default=DEFAULT_MIN_CREDIT_USD)
    parser.add_argument("--cost-per-question", type=float, default=DEFAULT_COST_PER_QUESTION_USD)
    parser.add_argument("--max-sweeps", type=int, default=None)
    parser.add_argument(
        "--wait-at-floor",
        action="store_true",
        help="at the credit floor, park and resume on top-up instead of exiting (for services)",
    )
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not submit")
    args = parser.parse_args(argv)

    if bool(args.until) == bool(args.hours):
        print("give exactly one of --until or --hours: an endless poll is not a plan.")
        return 2
    started = dt.datetime.now(dt.UTC)
    until = (
        _parse_deadline(args.until)
        if args.until
        else started + dt.timedelta(hours=float(args.hours))
    )
    if until <= started:
        print(f"deadline {until.isoformat()} is not in the future.")
        return 2
    if args.interval_minutes <= 0:
        print("--interval-minutes must be positive.")
        return 2

    try:
        session = build_session(dry_run=args.dry_run)
    except SetupError as exc:
        print(str(exc))
        return 2

    hours = (until - started).total_seconds() / 3600
    credit = session.credits_remaining()
    cap = affordable_questions(credit, args.min_credit, args.cost_per_question)
    print(
        f"polling {args.tournament} every {args.interval_minutes:g}m until "
        f"{until.isoformat(timespec='minutes')} ({hours:.1f}h, "
        f"~{int(hours * 60 / args.interval_minutes)} sweeps)"
    )
    print(
        f"OpenRouter: ${credit:.2f} on the key"
        if credit is not None
        else "OpenRouter: balance unreadable"
    )
    if cap is not None:
        # max(cap, 0): a shortfall is not a negative quantity of questions, and
        # "allows ~-4 questions" is the kind of nonsense number that teaches an
        # operator to stop reading the banner.
        print(
            f"budget allows ~{max(cap, 0)} more question(s) above the ${args.min_credit:.2f} floor "
            f"at ${args.cost_per_question:.2f} each"
        )

    try:
        outcome = poll(
            session,
            args.tournament,
            until=until,
            interval_seconds=args.interval_minutes * 60,
            min_credit=args.min_credit,
            cost_per_question=args.cost_per_question,
            dry_run=args.dry_run,
            max_sweeps=args.max_sweeps,
            wait_at_floor=args.wait_at_floor,
            alert=mail_alerter(args.min_credit, args.cost_per_question),
        )
    except KeyboardInterrupt:
        print("\ninterrupted — stopping after the current sweep.")
        return 130
    finally:
        session.close()

    print(
        f"stopped: {outcome.stopped_because}. "
        f"{outcome.sweeps} sweep(s), {outcome.submitted} forecast(s) submitted."
    )
    return 0


def mail_alerter(
    min_credit: float, cost_per_question: float
) -> Callable[[dict[str, float | None]], list[str]] | None:
    """The funding-alert hook for poll(), or None, said out loud, when mail is not set up.

    Built from the same floor and rate the loop uses, so "empty" in the email is
    exactly "parked" in the loop.
    """
    settings = alert_mail_settings()
    if settings is None:
        print("note: ALERT_SMTP_USER / ALERT_SMTP_PASSWORD absent — funding alerts are OFF.")
        return None
    from bot.venues.mail import send_mail

    wallets = funding_wallets(min_credit, cost_per_question)
    send = functools.partial(send_mail, settings)

    def alert(balances: dict[str, float | None]) -> list[str]:
        return check_funding(balances, wallets, send)

    print(f"funding alerts ON: mail to {settings.to} on each change of level.")
    return alert


if __name__ == "__main__":
    sys.exit(main())
