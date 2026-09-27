"""Tell the owner how the answers turned out: each resolution, and the score it earned.

The poller calls `check_results` on every tick; there is no CLI.

WHY THIS EXISTS
---------------

Asked for on 2026-09-27, the day the warmup's 60 answers were all in and none
had resolved: "do we have alerts for when the answers come through?" There
were none. The bot forgot a question the moment it forecast it, and results
reached the owner only if someone went and looked.

WHAT GETS MAILED
----------------

One mail per hourly check that finds anything newly resolved:

  scored     the outcome, what the bot said, and its spot peer score
  missed     resolved with no forecast from the bot: scores nothing
  annulled   does not count
  unscored   answered and resolved, but still without a score after
             SCORE_GRACE. Said out loud, not dropped

and, for each cycle in the mail, the running total with its n and standard
error, because one day's average of a handful of questions is mostly noise
(+23.9/q was n=10).

HOW IT COULD LIE
----------------

- **A parked poller is a silent one.** At the credit floor the poller stops
  sweeping, and the donated key had ~13 questions left when this was written.
  Results arrive whatever the balance, so this runs on every tick, parked or
  not, like the funding alerts. It spends no model credit.
- **A resolution can land before its score does.** An answered question with
  no score yet is left for the next check, and reported as unscored only
  after SCORE_GRACE, so a slow scorer never becomes a missing row.
- **A result mail that fails to send must not be forgotten.** Results are
  recorded only after the mail server accepts the mail; until then, every
  check finds them new again.
- **Two cycles under one slug would blur into one average.** MiniBench reuses
  "minibench" cycle after cycle, so totals are kept per project id.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from bot.config import PROJECT_ROOT
from bot.venues.metaculus import Resolution

STATE_DIR = PROJECT_ROOT / "data" / "metaculus"
CHECK_EVERY = dt.timedelta(hours=1)
SCORE_GRACE = dt.timedelta(hours=48)
MAX_POSTS_PER_CHECK = 100  # detail reads; the rest wait an hour, never lost
MAX_MISSED_LISTED = 20  # the 08-24 cycle alone would have listed 50
VOIDED = {"annulled", "ambiguous"}
OUT_OF_RANGE = {"above_upper_bound": "above the range", "below_lower_bound": "below the range"}

SCORE_NOTE = (
    "Spot peer score: how the bot did against everyone else who forecast the question. "
    "0 is average, and the leaderboard ranks bots on its sum."
)


class ResultsClient(Protocol):
    def resolved_post_ids(self, tournament: str) -> list[int]: ...

    def resolutions(self, post_id: int) -> list[Resolution]: ...


def question_url(post_id: int) -> str:
    return f"https://www.metaculus.com/questions/{post_id}/"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _percent(p: float) -> str:
    value = p * 100
    return f"{value:.1f}%" if value < 10 or value > 90 else f"{value:.0f}%"


def _number(value: float, qtype: str) -> str:
    if qtype == "date" and value > 1e8:  # date questions are scaled in Unix seconds
        return dt.datetime.fromtimestamp(value, dt.UTC).strftime("%Y-%m-%d")
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _outcome(r: Resolution) -> str:
    raw = r.resolution
    if raw is None:
        return "not shown"
    if raw in OUT_OF_RANGE:
        return OUT_OF_RANGE[raw]
    if r.qtype in ("numeric", "discrete"):
        try:
            return _number(float(raw), r.qtype)
        except ValueError:
            return raw
    if r.qtype == "date":
        return raw[:10]
    return raw


def _said(r: Resolution) -> str:
    """What the bot forecast, in words, measured against the outcome."""
    f = r.forecast
    if f is None:
        return "(see the question page)"
    if r.qtype == "binary":
        return f"The bot said {_percent(float(f))} yes."
    if r.qtype == "multiple_choice" and isinstance(f, dict):
        if r.resolution in f:
            return f"The bot gave it {_percent(float(f[r.resolution]))}."
        top = max(f, key=lambda option: float(f[option]))
        return f"The bot's favourite was {top} at {_percent(float(f[top]))}."
    return f"The bot's median: {_number(float(f), r.qtype)}."


def _kind(r: Resolution, now: dt.datetime) -> str | None:
    """scored / missed / annulled / unscored, or None to look again next check."""
    if not r.answered:
        return "missed"
    if r.resolution in VOIDED:
        return "annulled"
    if r.spot_peer is not None:
        return "scored"
    try:
        resolved = dt.datetime.fromisoformat(r.resolved_at.replace("Z", "+00:00"))
    except ValueError:
        return "unscored"
    return "unscored" if now - resolved >= SCORE_GRACE else None


def _entry(r: Resolution, kind: str) -> dict[str, Any]:
    return {
        "post_id": r.post_id,
        "project_id": r.project_id,
        "project": r.project,
        "title": r.title,
        "kind": kind,
        "spot_peer": r.spot_peer,
        "resolved_at": r.resolved_at,
    }


def _totals(entries: Sequence[dict[str, Any]]) -> str:
    """One cycle's running total, with its n and standard error."""
    scores = [float(e["spot_peer"]) for e in entries if e["kind"] == "scored"]
    missed = sum(1 for e in entries if e["kind"] == "missed")
    other = sum(1 for e in entries if e["kind"] in ("annulled", "unscored"))
    parts: list[str] = []
    if scores:
        n = len(scores)
        mean = sum(scores) / n
        line = f"{_plural(n, 'question')} scored, total {sum(scores):+.1f}, average {mean:+.1f}"
        if n >= 2:
            sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
            line += f" per question (standard error {sd / math.sqrt(n):.1f})"
        else:
            line += " (one question: no spread to measure yet)"
        parts.append(line)
    else:
        parts.append("no scored questions yet")
    if missed:
        parts.append(f"{missed} missed")
    if other:
        parts.append(f"{other} annulled or unscored")
    return "; ".join(parts)


def results_email(
    tournament: str,
    new: Sequence[tuple[Resolution, str]],
    ledger: dict[str, dict[str, Any]],
    now: dt.datetime,
) -> tuple[str, str]:
    """(subject, body) for newly resolved questions. `ledger` already includes them."""
    scored = [r for r, kind in new if kind == "scored"]
    missed = [r for r, kind in new if kind == "missed"]
    subject = f"devinjones-bot: {_plural(len(new), 'result')} in {tournament}"
    if scored:
        mean = sum(float(r.spot_peer or 0.0) for r in scored) / len(scored)
        subject += f", {mean:+.1f} spot peer on average over {len(scored)}"
    if missed:
        subject += f", {len(missed)} MISSED"
    lines = [
        f"{_plural(len(new), 'question')} resolved in {tournament} "
        f"(checked {now:%Y-%m-%d %H:%M}Z).",
        SCORE_NOTE,
        "",
    ]
    for number, (r, kind) in enumerate([(r, k) for r, k in new if k != "missed"], start=1):
        if kind == "scored":
            score = f"   Spot peer: {float(r.spot_peer or 0.0):+.1f}"
        elif kind == "annulled":
            score = "   Annulled: it doesn't count."
        else:
            score = f"   No score from Metaculus {SCORE_GRACE.days} days after it resolved."
        lines += [
            f"{number}. {r.title}",
            f"   Outcome: {_outcome(r)}. {_said(r)}",
            score,
            f"   {question_url(r.post_id)}",
            "",
        ]
    if missed:
        lines.append("MISSED: resolved with no forecast from the bot, so they score nothing:")
        for r in missed[:MAX_MISSED_LISTED]:
            lines += [f"- {r.title}", f"  {question_url(r.post_id)}"]
        if len(missed) > MAX_MISSED_LISTED:
            lines.append(f"... and {len(missed) - MAX_MISSED_LISTED} more.")
        lines.append("")
    cycles: dict[Any, list[dict[str, Any]]] = {}
    for entry in ledger.values():
        cycles.setdefault(entry.get("project_id"), []).append(entry)
    for project_id in dict.fromkeys(r.project_id for r, _ in new):
        entries = cycles.get(project_id, [])
        label = next((e["project"] for e in entries if e.get("project")), tournament)
        lines.append(f"So far in {label}: {_totals(entries)}.")
    return subject, "\n".join(lines).rstrip() + "\n"


def state_path(tournament: str, state_dir: Path | None = None) -> Path:
    return (state_dir or STATE_DIR) / f"results-{tournament}.json"


def read_state(path: Path) -> dict[str, Any]:
    """Missing or garbled reads as empty: at worst, results are mailed a second time."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"reported": {}}
    if not isinstance(raw, dict) or not isinstance(raw.get("reported"), dict):
        return {"reported": {}}
    return raw


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Temp file and rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state) + "\n", encoding="utf-8")
    tmp.replace(path)


def check_results(
    tournament: str,
    client: ResultsClient,
    send: Callable[[str, str], None],
    *,
    now: dt.datetime,
    state_dir: Path | None = None,
) -> list[str]:
    """Mail anything newly resolved. Returns the subjects the mail server accepted.

    Reads nothing more often than CHECK_EVERY. Network errors propagate: the
    poller contains them and prints them.
    """
    path = state_path(tournament, state_dir)
    state = read_state(path)
    try:
        last = dt.datetime.fromisoformat(str(state.get("checked_at")))
    except ValueError:
        last = None
    if last is not None and now - last < CHECK_EVERY:
        return []

    reported: dict[str, dict[str, Any]] = state["reported"]
    # A post with a question still waiting for its score is read again, even
    # when a sibling in the same group post has been reported.
    waiting = {int(pid) for pid in state.get("waiting") or []}
    known = {int(entry["post_id"]) for entry in reported.values()} - waiting
    fresh = [pid for pid in client.resolved_post_ids(tournament) if pid not in known]
    new: list[tuple[Resolution, str]] = []
    still_waiting: set[int] = set()
    for post_id in fresh[:MAX_POSTS_PER_CHECK]:
        for r in client.resolutions(post_id):
            if str(r.question_id) in reported:
                continue
            kind = _kind(r, now)
            if kind is None:
                still_waiting.add(post_id)
            else:
                new.append((r, kind))

    state["checked_at"] = now.isoformat()
    state["waiting"] = sorted(still_waiting | (waiting - set(fresh[:MAX_POSTS_PER_CHECK])))
    sent: list[str] = []
    if new:
        ledger = {**reported, **{str(r.question_id): _entry(r, kind) for r, kind in new}}
        subject, body = results_email(tournament, new, ledger, now)
        try:
            send(subject, body)
        except Exception as exc:  # noqa: BLE001 - the next check finds them again
            print(f"  EMAIL NOT SENT ({subject[:60]}): {type(exc).__name__}: {exc}")
        else:
            state["reported"] = ledger
            sent.append(subject)
    write_state(path, state)
    return sent
