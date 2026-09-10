"""Configuration loaded from .env — no secrets in source, ever."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Minimal .env reader. Deliberately not python-dotenv: one less dependency
    on the path that touches credentials."""
    path = path or (PROJECT_ROOT / ".env")
    values: dict[str, str] = {}
    if not path.exists():
        return values
    # utf-8-sig: PowerShell 5.1 writes a BOM, and a BOM on line one silently
    # renames the first key.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def openrouter_key() -> str | None:
    """LLM access for the forecaster: Metaculus-donated credits and personal
    keys are both OpenRouter keys, so one env var covers both funding routes."""
    env = {**load_dotenv(), **os.environ}
    key = env.get("OPENROUTER_API_KEY", "").strip()
    return key or None


def asknews_credentials() -> tuple[str, str] | None:
    """AskNews client id + secret, or None. A missing bonus research source
    degrades research breadth; it must never block a submission window."""
    env = {**load_dotenv(), **os.environ}
    client_id = env.get("ASKNEWS_CLIENT_ID", "").strip()
    secret = env.get("ASKNEWS_SECRET", "").strip()
    if client_id and secret:
        return client_id, secret
    return None


def metaculus_token() -> str | None:
    """The bot-account API token (devinjones-bot, created 2026-08-18), or None.

    None rather than raising: code that reads tournaments can be imported and
    tested on machines where the token has never been set, and the caller that
    actually needs to submit fails loudly at the client boundary instead.
    """
    env = {**load_dotenv(), **os.environ}
    token = env.get("METACULUS_TOKEN", "").strip()
    return token or None
