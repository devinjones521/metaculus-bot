"""Is every poller on the box actually TICKING?

    uv run python -m bot.health              inspect the box over SSH
    uv run python -m bot.health --contract   print the contract, no network

WHY `is-active` IS NOT ENOUGH
-----------------------------

This rig descends from free-money, where `systemctl` reported a recorder as
`active` for two days while it crash-looped 593 times and wrote zero rows. A
supervisor makes a dead service look like a live one. So this command never
asks "is it running". It asks "did it tick when a tick was due", and judges that
by the age of the heartbeat file the loop overwrites every interval, measured
against the box's own clock.

THE STATES
----------

  crashloop  unit not active                          loud
  silent     ACTIVE, but no heartbeat in 90 minutes   the one that hides
  flapping   ticking, but it has restarted before     loud
  parked     ticking, at the credit floor             not a fault: resumes on top-up
  overdue    past one interval, inside grace          watch, do not page

`parked` exists because a bot at the credit floor writes one log row and then
nothing for days, which from the log alone looks exactly like `silent`.

HOW THIS COULD LIE
------------------

  - The 90-minute grace (4.5 intervals) is a judgement call, not a measurement.
    The heartbeat is written when a tick STARTS, and a sweep over a fresh batch
    of MiniBench questions can take tens of minutes. Questions live ~3h, so 90
    minutes of silence is already half a question's life. It was chosen, not
    derived.
  - A ticking loop that submits nothing passes the liveness check. The
    submitted count is printed next to it for that reason. Read both.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
from dataclasses import dataclass
from typing import Any

from bot.config import PROJECT_ROOT

INTERVAL_S = 20 * 60  # bot.poll's DEFAULT_INTERVAL_MINUTES, which the unit uses
DEFAULT_GRACE = 4.5  # arbitrary; see the docstring


@dataclass(frozen=True)
class Poller:
    """One deployed poller: its unit, its tournament, and how often it must tick."""

    name: str
    unit: str
    tournament: str
    interval_s: int = INTERVAL_S
    grace: float = DEFAULT_GRACE

    @property
    def overdue_after_s(self) -> float:
        return self.interval_s * self.grace


# Every line of deploy/units must appear here and nothing else may. The
# `poller contract` invariant in bot.verify holds the two lists together.
POLLERS: tuple[Poller, ...] = (
    Poller("minibench", "metaculus-poll@minibench", "minibench"),
    Poller("fall", "metaculus-poll@fall-futureeval-2026", "fall-futureeval-2026"),
)

OK = "ok"
PARKED = "parked"
OVERDUE = "overdue"
FLAPPING = "flapping"
SILENT = "silent"
CRASHLOOP = "crashloop"
UNKNOWN = "unknown"

FAILING = (SILENT, CRASHLOOP)


@dataclass(frozen=True)
class Verdict:
    name: str
    severity: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.severity in (OK, PARKED, OVERDUE)


def _epoch(stamp: object) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)).timestamp()


def judge(
    poller: Poller,
    active: str | None,
    restarts: int,
    beat: dict[str, Any] | None,
    now_epoch: float,
) -> Verdict:
    """Pure verdict for one poller. The clock is an argument, never read here."""
    if active is None:
        return Verdict(poller.name, UNKNOWN, "unit not found on the box")
    if active != "active":
        return Verdict(
            poller.name,
            CRASHLOOP,
            f"unit is '{active}' with {restarts} restart(s) — it is not running",
        )

    beat_at = _epoch(beat.get("ts")) if beat else None
    if beat_at is None:
        return Verdict(
            poller.name,
            SILENT,
            f"unit is active but heartbeat-{poller.tournament}.json is missing or "
            "unreadable — it has never completed a tick",
        )

    age = now_epoch - beat_at
    if age > poller.overdue_after_s:
        return Verdict(
            poller.name,
            SILENT,
            f"unit is ACTIVE but last ticked {_duration(age)} ago — over the "
            f"{_duration(poller.overdue_after_s)} limit. This is the failure that hides.",
        )
    if restarts > 0:
        return Verdict(
            poller.name,
            FLAPPING,
            f"ticking (last {_duration(age)} ago) but has restarted {restarts} time(s)",
        )
    credit = beat.get("credit") if beat else None
    money = f"${credit:.2f}" if isinstance(credit, int | float) else "balance unknown"
    if beat and beat.get("state") == "parked":
        return Verdict(
            poller.name,
            PARKED,
            f"ticking (last {_duration(age)} ago), parked at the credit floor with {money} — "
            "resumes by itself when the key is topped up",
        )
    if age > poller.interval_s:
        return Verdict(
            poller.name,
            OVERDUE,
            f"last ticked {_duration(age)} ago, past its interval but inside grace ({money})",
        )
    return Verdict(poller.name, OK, f"ticking, last {_duration(age)} ago, {money} on the key")


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86_400:.1f}d"


# -- deploy drift -----------------------------------------------------------


def local_hashes() -> dict[str, str]:
    """md5 of every bot/*.py here, keyed by path relative to the project root."""
    out: dict[str, str] = {}
    for path in sorted((PROJECT_ROOT / "bot").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        out[path.relative_to(PROJECT_ROOT).as_posix()] = digest
    return out


@dataclass(frozen=True)
class Drift:
    missing: list[str]
    differing: list[str]
    extra: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.differing)


def compare_trees(local: dict[str, str], remote: dict[str, str]) -> Drift:
    """What the box is missing or running a stale copy of. `extra` is not a failure."""
    missing = sorted(k for k in local if k not in remote)
    differing = sorted(k for k in local if k in remote and local[k] != remote[k])
    extra = sorted(k for k in remote if k not in local)
    return Drift(missing=missing, differing=differing, extra=extra)


# -- reporting --------------------------------------------------------------


def print_contract() -> int:
    print("POLLER LIVENESS CONTRACT")
    for p in POLLERS:
        print(
            f"  {p.name:<10} {p.unit:<38} tick every {_duration(p.interval_s):>4}, "
            f"silent after {_duration(p.overdue_after_s)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--contract":
        return print_contract()

    from bot.venues.box import gather  # local import keeps the offline path clean

    state = gather([p.unit for p in POLLERS], [p.tournament for p in POLLERS])
    print("=" * 78)
    print("POLLER HEALTH — is devinjones-bot actually ticking?")
    print("=" * 78)
    if not state.reachable:
        print(f"\n  BOX UNREACHABLE: {state.error}")
        print("  A verdict on the connection, not on the pollers. Nothing was measured.")
        return 2

    verdicts = [
        judge(
            p,
            state.units[p.unit][0] if p.unit in state.units else None,
            state.units.get(p.unit, ("", 0))[1],
            state.beats.get(p.tournament),
            state.box_epoch,
        )
        for p in POLLERS
    ]
    print(f"\n  box clock {state.box_epoch:.0f} (its own, not this laptop's)\n")
    for v in verdicts:
        mark = v.severity if v.ok else v.severity.upper()
        print(f"  {v.name:<10} {mark:<10} {v.detail}")
    print(f"\n  forecasts submitted from the box, all time: {state.submitted}")
    if state.last_poll:
        print(f"  last poll row: {state.last_poll}")

    drift = compare_trees(local_hashes(), state.hashes)
    print("\n  deploy: ", end="")
    if drift.ok:
        print(f"in sync ({len(local_hashes())} modules match)")
    else:
        stale = drift.missing + drift.differing
        print(f"{len(stale)} module(s) differ from this repo — run `bash deploy/push.sh`")
        for path in stale:
            print(f"    {path}")

    failures = [v for v in verdicts if v.severity in FAILING]
    print("\n" + "=" * 78)
    if failures or not drift.ok:
        print("UNHEALTHY")
        return 1
    print("HEALTHY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
