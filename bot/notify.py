"""Tell the owner what the bot is doing: when a tournament opens, and every answer it gives.

The poller calls `notify_sweep` after each sweep; there is no CLI.

WHY THIS EXISTS
---------------

Asked for on 2026-09-10: the owner wants to know when a tournament starts,
when the bot answers, and what it answered, from the inbox and without
running anything. The funding alerts (`bot.alerts`) say whether the bot CAN
forecast; these say what it DID.

WHAT GETS MAILED
----------------

  opening   the first new questions a tournament shows after at least
            QUIET_HOURS with none: a MiniBench cycle starting, or the Fall
            season's first questions.
  answers   one mail per sweep that submitted or failed anything: each
            question, its link, the answer in words, and how many of the
            ensemble's runs contributed.

A sweep that did nothing sends nothing. A mail every 20 minutes saying
"nothing happened" is how an inbox learns to ignore the bot.

HOW IT COULD LIE
----------------

- **The first sweep after a deploy sees questions that were already open.**
  Without memory it would announce a tournament that started weeks ago. With
  no state file, a sweep records what is open as a baseline and says nothing.
- **Two pollers sharing one file would clobber each other's memory.** Each
  tournament keeps its own state file.
- **An opening mail that fails to send must not be forgotten.** The memory
  moves on only after the mail server accepts it, so the next sweep tries
  again. Answer mails are not retried: the forecast log is the record, and a
  retry would repeat the list.
- **A mail problem must never cost a forecast.** Every send is contained here,
  and the poller contains this whole call as well.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bot.config import PROJECT_ROOT
from bot.forecast import ForecastResult

STATE_DIR = PROJECT_ROOT / "data" / "metaculus"
QUIET_HOURS = 24.0
MAX_REMEMBERED = 3000  # question ids per tournament; a season is a few hundred
MAX_LISTED = 20  # new questions named in one opening mail


def question_url(post_id: int) -> str:
    return f"https://www.metaculus.com/questions/{post_id}/"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _ago(delta: dt.timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours / 24:.0f} days" if hours >= 48 else f"{hours:.0f} hours"


def answers_email(
    tournament: str, results: Sequence[ForecastResult], when: dt.datetime
) -> tuple[str, str] | None:
    """(subject, body) for one sweep, or None when it submitted and failed nothing."""
    submitted = [r for r in results if r.submitted]
    failed = [r for r in results if r.error and not r.error.startswith("skipped")]
    if not submitted and not failed:
        return None
    subject = f"devinjones-bot answered {_plural(len(submitted), 'question')} in {tournament}"
    if failed:
        subject += f", {len(failed)} FAILED"
    lines = [
        f"{tournament}, sweep at {when:%Y-%m-%d %H:%M}Z: "
        f"{len(submitted)} submitted, {len(failed)} failed.",
        "",
    ]
    for number, result in enumerate(submitted, start=1):
        runs = result.runs_ok + result.runs_failed
        note = f"   {result.runs_ok}/{runs} models answered"
        if result.suspect:
            note += "; the extremeness check flagged it, so it was capped at 10-90%"
        lines += [
            f"{number}. {result.title}",
            f"   Answer: {result.answer or '(see the question page)'}",
            note,
            *(f"   dropped: {reason[:200]}" for reason in result.run_errors),
            f"   {question_url(result.post_id)}",
            "",
        ]
    if failed:
        lines.append("FAILED, nothing submitted:")
        for result in failed:
            lines += [
                f"- {result.title}",
                f"  {(result.error or '')[:400]}",
                f"  {question_url(result.post_id)}",
            ]
    return subject, "\n".join(lines).rstrip() + "\n"


def _parse(stamp: object) -> dt.datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        return dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None


def opening_email(
    tournament: str,
    results: Sequence[ForecastResult],
    state: dict[str, Any] | None,
    now: dt.datetime,
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    """Pure: an 'opening' mail if new questions arrived after a quiet spell, and the next state.

    State is {"seen": [question ids], "last_new": iso timestamp or None}.
    """
    open_now = {r.question_id: r for r in results}
    if state is None:
        return None, {"seen": sorted(open_now), "last_new": None}
    seen = set(state.get("seen") or [])
    new = [r for qid, r in open_now.items() if qid not in seen]
    if not new:
        return None, state
    last_new = _parse(state.get("last_new"))
    next_state = {
        "seen": sorted(seen | set(open_now))[-MAX_REMEMBERED:],
        "last_new": now.isoformat(),
    }
    if last_new is not None and now - last_new < dt.timedelta(hours=QUIET_HOURS):
        return None, next_state
    since = (
        "the first since the bot began watching"
        if last_new is None
        else f"the first in {_ago(now - last_new)}"
    )
    subject = f"devinjones-bot: {tournament} is open, {_plural(len(new), 'new question')}"
    lines = [
        f"{_plural(len(new), 'new question')} in {tournament}, {since}.",
        "The bot is forecasting them now; each sweep's answers follow in their own email.",
        "",
    ]
    for result in new[:MAX_LISTED]:
        lines += [f"- {result.title}", f"  {question_url(result.post_id)}"]
    if len(new) > MAX_LISTED:
        lines.append(f"... and {len(new) - MAX_LISTED} more.")
    return (subject, "\n".join(lines) + "\n"), next_state


def state_path(tournament: str, state_dir: Path | None = None) -> Path:
    return (state_dir or STATE_DIR) / f"notify-{tournament}.json"


def read_state(path: Path) -> dict[str, Any] | None:
    """None when missing or garbled, which means: record a baseline, announce nothing."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("seen"), list):
        return None
    return raw


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Temp file and rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state) + "\n", encoding="utf-8")
    tmp.replace(path)


def _send(send: Callable[[str, str], None], subject: str, body: str) -> bool:
    try:
        send(subject, body)
    except Exception as exc:  # noqa: BLE001 - a mail problem must never cost a forecast
        print(f"  EMAIL NOT SENT ({subject[:60]}): {type(exc).__name__}: {exc}")
        return False
    return True


def notify_sweep(
    tournament: str,
    results: Sequence[ForecastResult],
    send: Callable[[str, str], None],
    *,
    now: dt.datetime,
    state_dir: Path | None = None,
) -> list[str]:
    """Mail what this sweep owes the owner. Returns the subjects the mail server accepted."""
    path = state_path(tournament, state_dir)
    state = read_state(path)
    opening, next_state = opening_email(tournament, results, state, now)
    sent: list[str] = []
    if opening is None:
        if next_state is not state:
            write_state(path, next_state)
    elif _send(send, *opening):
        write_state(path, next_state)
        sent.append(opening[0])
    answers = answers_email(tournament, results, now)
    if answers is not None and _send(send, *answers):
        sent.append(answers[0])
    return sent
