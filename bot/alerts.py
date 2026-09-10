"""Email the owner before the money runs out: once per change of level, never once per tick.

    uv run python -m bot.alerts --status   each key's balance and level now; sends nothing
    uv run python -m bot.alerts --test     send one test email, proving the path end to end

WHY THIS EXISTS
---------------

The donated key covers about one MiniBench cycle (~$69 of its $100). Metaculus
raises it automatically on above-average MiniBench results, but says it may
not, and the Fall season runs to 2027-03. At the credit floor the poller parks,
correctly, and forecasts nothing until money arrives. From outside that is
invisible: the heartbeat says "parked" to whoever runs `bot.health`, and nobody
runs it at 3am. A missed question scores like a wrong one, so what an email
buys is lead time to top up.

There are two keys, because the roster spans two (see `bot.forecast`):

  donated   research and four ensemble runs. Empty means NO forecasts at all.
  personal  Grok only. Empty means a four-model ensemble, not a stop.

LEVELS
------

  ok    nothing to do
  low   about LOW_QUESTIONS questions of headroom left above the floor
  out   donated: the pollers have parked. personal: Grok can no longer run.

A mail goes out on every change of level, in either direction, so a top-up is
confirmed too. The last level mailed is kept in data/metaculus/alerts.json,
which both pollers share: a restart does not re-send, and two pollers watching
one key send one mail between them.

HOW IT COULD LIE
----------------

- **A dead poller sends nothing, and silence reads as health.** This module
  speaks only when a live poller reads a balance. It is not a liveness check;
  `bot.health` is.
- **A failed send must not count as sent.** The level is recorded only after
  the mail server accepts the message, so a failure is retried next tick.
- **An unreadable balance is not an empty one.** None changes nothing: one bad
  OpenRouter response must not mail "out of money".
- **Both pollers can tick in the same second** and both send. That race was
  judged cheaper than a lock: a duplicate email costs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bot.config import PROJECT_ROOT, alert_mail_settings, openrouter_key, openrouter_personal_key

STATE_PATH = PROJECT_ROOT / "data" / "metaculus" / "alerts.json"

OK = "ok"
LOW = "low"
OUT = "out"
LEVELS = (OK, LOW, OUT)

# The early warning, in questions of headroom. A MiniBench cycle is ~60 questions
# and most arrive in its first days, so 25 is a day or two of warning at the
# busiest time. Chosen, not derived.
LOW_QUESTIONS = 25

# The personal key pays for Grok alone. Measured 2026-08-24: grok-4.6 cost $0.34
# across 11 questions. Its max_tokens=8000 reservation is $0.048 (OpenRouter
# pricing, 2026-09-10); below that every Grok run 402s.
GROK_COST_PER_QUESTION_USD = 0.03
GROK_FLOOR_USD = 0.05


@dataclass(frozen=True)
class Wallet:
    """One OpenRouter key, and what running it dry does to the bot."""

    name: str
    label: str
    floor: float  # below this, nothing the key pays for can run
    per_question: float
    when_empty: str
    to_fix: str

    @property
    def out_below(self) -> float:
        return self.floor + self.per_question

    @property
    def low_below(self) -> float:
        return self.floor + LOW_QUESTIONS * self.per_question

    def questions_left(self, balance: float) -> int:
        return max(0, int((balance - self.floor) // self.per_question))


def funding_wallets(min_credit: float, cost_per_question: float) -> dict[str, Wallet]:
    """The two keys. The donated key's floor and rate are the poller's own, so its
    'out' is exactly the condition under which `bot.poll` parks."""
    return {
        "donated": Wallet(
            name="donated",
            label="donated key",
            floor=min_credit,
            per_question=cost_per_question,
            when_empty="both pollers park and NO forecasts are submitted until credit arrives.",
            to_fix=(
                "Metaculus raises this key on above-average MiniBench results. If nothing "
                "arrives, ask Metaculus, or put a funded key in OPENROUTER_API_KEY and run "
                "`bash deploy/push.sh --env`."
            ),
        ),
        "personal": Wallet(
            name="personal",
            label="personal key (Grok)",
            floor=GROK_FLOOR_USD,
            per_question=GROK_COST_PER_QUESTION_USD,
            when_empty="Grok's runs fail and each question runs on four models; forecasting "
            "continues.",
            to_fix="top up at https://openrouter.ai/settings/credits. Nothing needs redeploying.",
        ),
    }


def level(balance: float | None, wallet: Wallet) -> str | None:
    """OK, LOW or OUT. None when the balance is unknown, because unknown is not empty."""
    if balance is None:
        return None
    if balance < wallet.out_below:
        return OUT
    if balance < wallet.low_below:
        return LOW
    return OK


@dataclass(frozen=True)
class Alert:
    wallet: Wallet
    was: str
    now: str
    balance: float

    def render(self) -> tuple[str, str]:
        """(subject, body). The subject alone should be enough to act on."""
        wallet, left = self.wallet, self.wallet.questions_left(self.balance)
        money = f"${self.balance:.2f}"
        if self.now == OK:
            subject = f"devinjones-bot: {wallet.label} funded again ({money})"
        elif self.now == LOW:
            subject = f"devinjones-bot: {wallet.label} LOW, {money} (~{left} questions left)"
        else:
            subject = f"devinjones-bot: {wallet.label} EMPTY, {money}"
        lines = [
            f"The {wallet.label} has {money} (level {self.was} -> {self.now}).",
            f"At about ${wallet.per_question:.2f} a question, that is roughly {left} more "
            f"question(s) before the ${wallet.floor:.2f} floor.",
            "",
        ]
        if self.now == OK:
            lines.append("Nothing to do: the pollers resume by themselves.")
        else:
            lines.append(f"When it is empty: {wallet.when_empty}")
            lines.append(f"To fix: {wallet.to_fix}")
        lines += [
            "",
            "Check the pollers with `uv run python -m bot.health`.",
            "Sent by bot.alerts on the box, once per change of level.",
        ]
        return subject, "\n".join(lines)


def decide(
    balances: Mapping[str, float | None],
    wallets: Mapping[str, Wallet],
    last: Mapping[str, str],
) -> list[Alert]:
    """Pure: the alerts a tick with these balances owes, given the levels last mailed."""
    owed: list[Alert] = []
    for name, wallet in wallets.items():
        balance = balances.get(name)
        now = level(balance, wallet)
        if balance is None or now is None:
            continue
        was = last.get(name, OK)
        if now != was:
            owed.append(Alert(wallet, was, now, balance))
    return owed


def read_state(path: Path) -> dict[str, str]:
    """The level last mailed, per key. Missing or garbled means nothing was mailed yet."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(name): value for name, value in raw.items() if value in LEVELS}


def write_state(path: Path, state: Mapping[str, str]) -> None:
    """Temp file and rename, so the other poller never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(state), sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def check_funding(
    balances: Mapping[str, float | None],
    wallets: Mapping[str, Wallet],
    send: Callable[[str, str], None],
    *,
    state_path: Path | None = None,
) -> list[str]:
    """Mail whatever this tick owes. Returns the subjects the mail server accepted."""
    path = state_path or STATE_PATH
    state = read_state(path)
    sent: list[str] = []
    for alert in decide(balances, wallets, state):
        subject, body = alert.render()
        try:
            send(subject, body)
        except Exception as exc:  # noqa: BLE001 - an alert failure must never stop forecasting
            print(f"  ALERT NOT SENT, retrying next tick: {type(exc).__name__}: {exc}")
            continue
        state[alert.wallet.name] = alert.now
        write_state(path, state)
        sent.append(subject)
    return sent


def current_balances() -> dict[str, float | None]:
    """Both keys' balances, read directly. For the CLI; the poller reads its own."""
    from bot.venues.llm import OpenRouterClient

    out: dict[str, float | None] = {}
    for name, key in (("donated", openrouter_key()), ("personal", openrouter_personal_key())):
        value = None
        if key:
            client = OpenRouterClient(key)
            try:
                value = client.key_limits().get("limit_remaining")
            except Exception:  # noqa: BLE001 - unreadable is reported as unknown
                value = None
            finally:
                client.close()
        out[name] = float(value) if isinstance(value, int | float) else None
    return out


def main(argv: list[str] | None = None) -> int:
    from bot.poll import DEFAULT_COST_PER_QUESTION_USD, DEFAULT_MIN_CREDIT_USD

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="print each key's level; send nothing")
    mode.add_argument("--test", action="store_true", help="send one test email")
    args = parser.parse_args(argv)

    known = funding_wallets(DEFAULT_MIN_CREDIT_USD, DEFAULT_COST_PER_QUESTION_USD)
    balances = current_balances()
    report: list[str] = []
    for name, wallet in known.items():
        balance = balances.get(name)
        shown = "unset/unknown" if balance is None else f"${balance:.2f}"
        report.append(
            f"{wallet.label:<20} {shown:>14}  level={level(balance, wallet)}  "
            f"(low below ${wallet.low_below:.2f}, empty below ${wallet.out_below:.2f})"
        )
    print("\n".join(report))
    print(f"last levels mailed: {read_state(STATE_PATH) or 'none yet'}")
    if args.status:
        return 0

    settings = alert_mail_settings()
    if settings is None:
        print("ALERT_SMTP_USER / ALERT_SMTP_PASSWORD are not set in .env: nothing to test.")
        return 2
    from bot.venues.mail import send_mail

    body = "A test of devinjones-bot's funding alerts. Balances now:\n\n" + "\n".join(report)
    try:
        send_mail(settings, "devinjones-bot: test alert", body)
    except Exception as exc:  # noqa: BLE001 - reported, and the exit code says so
        print(f"test email FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"test email sent to {settings.to}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
