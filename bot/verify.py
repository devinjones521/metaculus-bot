"""One command, one exit code: lint -> format -> types -> tests -> invariants.

The invariants are the point of this module. None is catchable by a linter:

  1. No network access outside bot/venues/ — one place to audit where bytes
     leave the machine.
  2. No skipped tests — a skipped test is a green light nobody earned.
  3. No credential in any file git would publish. This repo is PUBLIC; the
     Metaculus token submits as devinjones-bot and the OpenRouter key spends
     donated credit. `.gitignore` protects one filename; this protects content.
  4. Every deployed unit has a liveness contract in bot.health, and vice versa.

Anything an agent must *remember* to do, it will eventually not do. Anything
wired into the one command with the one exit code happens every time.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bot.config import PROJECT_ROOT, load_dotenv

# `imaplib`/`ssl` belong here even though nothing imports them today: a list of
# network modules is only as good as its last update, and the free-money repo
# this was split from once let a mail client walk straight past a shorter list.
NETWORK_MODULES = {
    "httpx",
    "requests",
    "urllib",
    "urllib3",
    "socket",
    "aiohttp",
    "http",
    "imaplib",
    "smtplib",
    "poplib",
    "ftplib",
    "telnetlib",
    "ssl",
    "asyncio",
}
VENUE_PACKAGE = "bot/venues"
EXCLUDED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        line = f"  [{mark}] {self.name}"
        if self.detail:
            line += f"\n         {self.detail.rstrip()}"
        return line


def python_sources() -> list[Path]:
    """Every .py file we own. Excludes the venv and caches."""
    return sorted(
        p for p in PROJECT_ROOT.rglob("*.py") if not set(p.parts) & (EXCLUDED_DIRS | {"scratch"})
    )


# -- invariant 1 ------------------------------------------------------------


def check_venue_isolation() -> Result:
    """All network access goes through one package, so there is one place to audit."""
    offenders: list[str] = []
    for path in python_sources():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith(VENUE_PACKAGE) or rel.startswith("tests/") or rel == "bot/verify.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in NETWORK_MODULES:
                        offenders.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in NETWORK_MODULES
            ):
                offenders.append(f"{rel}:{node.lineno}: from {node.module} import ...")
    if offenders:
        return Result(
            "venue isolation",
            False,
            "network imports outside bot/venues/:\n         " + "\n         ".join(offenders),
        )
    return Result("venue isolation", True, "no network imports outside bot/venues/")


# -- invariant 2 ------------------------------------------------------------

SKIP_PATTERNS = (
    re.compile(r"@pytest\.mark\.skip"),
    re.compile(r"@pytest\.mark\.xfail"),
    re.compile(r"pytest\.skip\("),
    re.compile(r"pytest\.xfail\("),
)


def check_no_skipped_tests() -> Result:
    offenders: list[str] = []
    for path in python_sources():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel == "bot/verify.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in SKIP_PATTERNS):
                offenders.append(f"{rel}:{number}: {line.strip()[:70]}")
    if offenders:
        return Result(
            "no skipped tests",
            False,
            "\n         ".join(offenders) + "\n         Never weaken a test to go green.",
        )
    return Result("no skipped tests", True, "no skip/xfail markers")


# -- invariant 3 ------------------------------------------------------------

# Shapes, for keys that are not in this machine's .env (a teammate's, an old
# one pasted into a doc). The exact .env values are checked separately below,
# which catches a leaked secret of ANY shape — including the Metaculus token,
# which is bare hex and has no prefix to match on.
SECRET_SHAPES = (
    re.compile(r"sk-or-v1-[0-9a-f]{32,}"),  # OpenRouter
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{32,}"),  # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{32,}"),  # OpenAI
    re.compile(r"Token [0-9a-f]{40}\b"),  # a Metaculus/DRF token in a header
)
# Values this short are not secrets (booleans, model ids are not in .env as
# secrets either, but a 5-char value would match half the codebase).
MIN_LITERAL_LENGTH = 16


def scan_text_for_secrets(text: str, literals: Iterable[str] = ()) -> list[tuple[int, str]]:
    """Pure helper so the rule is testable without planting a real key in the tree.

    Returns (line number, reason) — never the matched text, so the report that
    catches a leak does not become a second copy of it.
    """
    wanted = [v for v in literals if len(v) >= MIN_LITERAL_LENGTH]
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for shape in SECRET_SHAPES:
            if shape.search(line):
                found.append((number, f"credential-shaped ({shape.pattern[:14]}...)"))
        for value in wanted:
            if value in line:
                found.append((number, "a literal value from .env"))
    return found


def publishable_files() -> list[Path]:
    """What `git add -A` would publish: tracked plus untracked-not-ignored.

    Untracked files count because the leak that matters is the one about to be
    committed, not only the one already committed.
    """
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()[:200]}")
    return [PROJECT_ROOT / line for line in proc.stdout.splitlines() if line.strip()]


def check_no_published_secrets(env_path: Path | None = None) -> Result:
    literals = list(load_dotenv(env_path).values())
    try:
        files = publishable_files()
    except RuntimeError as exc:
        # Unknown is not clean. A check that passes because it could not look is
        # the failure that looks like success.
        return Result("no published secrets", False, str(exc))
    hits: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, reason in scan_text_for_secrets(text, literals):
            hits.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{number}: {reason}")
    if hits:
        return Result(
            "no published secrets",
            False,
            "\n         ".join(hits) + "\n         This repo is public. Move it to .env.",
        )
    return Result(
        "no published secrets",
        True,
        f"{len(files)} publishable files, {len(literals)} .env values checked",
    )


# -- invariant 4 ------------------------------------------------------------


def check_poller_contract(manifest: Path | None = None) -> Result:
    """Every deployed unit must declare what 'alive' means for it, and vice versa.

    The offline half of liveness. `bot.health` does the network half and cannot
    live here: verify is a pre-commit gate and must pass without a network, or
    it gets disabled. What this catches instead is a unit added to deploy/units
    that no health check knows about — a poller nobody would notice dying.
    """
    from bot.health import POLLERS

    manifest = manifest or (PROJECT_ROOT / "deploy" / "units")
    if not manifest.exists():
        return Result("poller contract", False, f"{manifest} is missing")
    declared = {p.unit for p in POLLERS}
    deployed = {
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    problems = []
    if uncovered := sorted(deployed - declared):
        problems.append(
            "deployed with no liveness contract: " + ", ".join(uncovered) + " — add it to "
            "bot.health.POLLERS; a poller nobody checks is one that dies quietly."
        )
    if phantom := sorted(declared - deployed):
        problems.append("under contract but not deployed: " + ", ".join(phantom))
    if problems:
        return Result("poller contract", False, "\n         ".join(problems))
    return Result("poller contract", True, f"{len(deployed)} deployed unit(s), all under contract")


# -- external tools ---------------------------------------------------------


def run_tool(name: str, argv: list[str]) -> Result:
    proc = subprocess.run(
        argv, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return Result(name, False, output[-3000:])
    tail = output.splitlines()[-1] if output else "ok"
    return Result(name, True, tail)


def main() -> int:
    # Windows consoles default to cp1252; a non-ASCII test message would crash
    # a passing run. Fail loudly about real problems, never about the codepage.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("verify: lint -> format -> types -> tests -> invariants")
    print("=" * 72)

    results: list[Result] = []
    print("\nTOOLS")
    for name, argv in [
        ("ruff (lint)", [sys.executable, "-m", "ruff", "check", "."]),
        ("ruff (format)", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("mypy (types)", [sys.executable, "-m", "mypy", "bot"]),
        ("pytest", [sys.executable, "-m", "pytest"]),
    ]:
        result = run_tool(name, argv)
        results.append(result)
        print(result.render())

    print("\nINVARIANTS")
    for check in (
        check_venue_isolation,
        check_no_skipped_tests,
        check_no_published_secrets,
        check_poller_contract,
    ):
        result = check()
        results.append(result)
        print(result.render())

    failed = [r for r in results if not r.ok]
    print("\n" + "=" * 72)
    if failed:
        print(f"FAILED ({len(failed)} of {len(results)}): " + ", ".join(r.name for r in failed))
        print("=" * 72)
        return 1
    print(f"GREEN ({len(results)} checks)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
