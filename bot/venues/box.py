"""SSH access to the Hetzner box that runs the pollers. The only module that talks to it.

All network access lives in bot/venues/, and reaching another host over SSH is
network access even though it imports subprocess rather than httpx. The rule is
about having one place to audit, not about a particular library.

READ-ONLY BY CONSTRUCTION. Every command sent here is an inspection:
systemctl show, cat, tail, grep, md5sum, date. It never restarts a unit and
never writes anywhere. That box also serves live websites. A health checker that
can repair things can also destroy things at 3am on a bad heuristic. Deploying
is `deploy/push.sh`, run deliberately.

How the gathering could lie:
  - Clock skew between the box and this laptop would corrupt every freshness
    verdict, so the box's own `date +%s` comes back and judging uses that.
  - A heartbeat proves the loop is ticking, not that forecasts are landing. The
    submitted count from forecasts.jsonl is gathered alongside for that reason.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BOX_HOST = "root@204.168.148.150"
BOX_KEY = "~/.ssh/id_ed25519"
BOX_ROOT = "/opt/metaculus-bot"
SSH_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class BoxState:
    """Everything one SSH round trip can tell us. `reachable=False` means nothing else is known."""

    reachable: bool
    box_epoch: float = 0.0
    units: dict[str, tuple[str, int]] = field(default_factory=dict)
    beats: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    last_poll: dict[str, Any] | None = None
    submitted: int = 0
    hashes: dict[str, str] = field(default_factory=dict)
    error: str = ""


def _remote_script(units: list[str], tournaments: list[str]) -> str:
    """One script, one round trip. Emits tab-separated records."""
    lines = [f"R={BOX_ROOT}", "D=$R/data/metaculus"]
    for unit in units:
        lines.append(
            f'printf "unit\\t{unit}\\t%s\\t%s\\n" '
            f'"$(systemctl is-active {unit} 2>/dev/null)" '
            f'"$(systemctl show {unit} -p NRestarts --value 2>/dev/null)"'
        )
    for tournament in tournaments:
        lines.append(
            f'printf "beat\\t{tournament}\\t%s\\n" '
            f"\"$(tr -d '\\n' < $D/heartbeat-{tournament}.json 2>/dev/null)\""
        )
    lines.append('printf "poll\\t%s\\n" "$(tail -n 1 $D/polls.jsonl 2>/dev/null)"')
    lines.append(
        'printf "submitted\\t%s\\n" "$(grep -c \'\\"submitted\\": true\' '
        '$D/forecasts.jsonl 2>/dev/null || echo 0)"'
    )
    lines.append(
        'cd $R 2>/dev/null && find bot -name "*.py" | sort | while read -r f; do '
        'printf "hash\\t%s\\t%s\\n" "$f" "$(md5sum "$f" | cut -d" " -f1)"; done'
    )
    lines.append('printf "now\\t%s\\n" "$(date +%s)"')
    return "\n".join(lines)


def gather(units: list[str], tournaments: list[str]) -> BoxState:
    """Inspect the box. Never raises: unreachable is a verdict, not a crash."""
    argv = [
        "ssh",
        "-i",
        str(Path(BOX_KEY).expanduser()),
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
        BOX_HOST,
        _remote_script(units, tournaments),
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SSH_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BoxState(reachable=False, error=f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        return BoxState(reachable=False, error=(proc.stderr or "ssh failed").strip()[:400])
    return parse(proc.stdout)


def _json_or_none(raw: str) -> dict[str, Any] | None:
    """A missing or half-written file is None — never an exception, never {}."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse(text: str) -> BoxState:
    """Pure parser, so the wire format is testable without a box."""
    units: dict[str, tuple[str, int]] = {}
    beats: dict[str, dict[str, Any] | None] = {}
    hashes: dict[str, str] = {}
    last_poll: dict[str, Any] | None = None
    submitted = 0
    box_epoch = 0.0
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        kind = parts[0]
        if kind == "unit" and len(parts) >= 4:
            restarts = int(parts[3]) if parts[3].strip().isdigit() else 0
            units[parts[1]] = (parts[2].strip(), restarts)
        elif kind == "beat" and len(parts) >= 2:
            beats[parts[1]] = _json_or_none(parts[2] if len(parts) >= 3 else "")
        elif kind == "poll":
            last_poll = _json_or_none(parts[1] if len(parts) >= 2 else "")
        elif kind == "submitted" and len(parts) >= 2:
            submitted = int(parts[1]) if parts[1].strip().isdigit() else 0
        elif kind == "hash" and len(parts) >= 3:
            hashes[parts[1].replace("\\", "/")] = parts[2].strip()
        elif kind == "now" and len(parts) >= 2:
            box_epoch = float(parts[1])
    return BoxState(
        reachable=True,
        box_epoch=box_epoch,
        units=units,
        beats=beats,
        last_poll=last_poll,
        submitted=submitted,
        hashes=hashes,
    )
