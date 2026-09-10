"""Configuration loaded from .env — no secrets in source, ever."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def openrouter_personal_key() -> str | None:
    """The owner's own OpenRouter key, for the providers the donated key refuses.

    Measured 2026-09-10: the donated key's account allows only openai, anthropic
    and google-ai-studio, and x-ai/grok-4.6 returns HTTP 404. None when unset:
    the roster then drops the models that need it, rather than sending every
    question a run that is certain to fail.
    """
    env = {**load_dotenv(), **os.environ}
    key = env.get("OPENROUTER_PERSONAL_KEY", "").strip()
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


@dataclass(frozen=True)
class MailSettings:
    """Where funding alerts are sent from and to. The password is a Gmail app password."""

    user: str
    password: str
    to: str

    def __repr__(self) -> str:
        """Never let the password reach a log line or a traceback."""
        return f"MailSettings(user={self.user!r}, password=***, to={self.to!r})"


def alert_mail_settings() -> MailSettings | None:
    """The Gmail login for funding alerts, or None when alerts are not configured.

    None switches alerts off, with a note at startup; it never blocks
    forecasting. ALERT_EMAIL_TO defaults to the sending address, so the owner
    mails themselves. Google shows app passwords in groups of four with spaces;
    the spaces are dropped, so a pasted password works as shown.
    """
    env = {**load_dotenv(), **os.environ}
    user = env.get("ALERT_SMTP_USER", "").strip()
    password = env.get("ALERT_SMTP_PASSWORD", "").replace(" ", "").strip()
    to = env.get("ALERT_EMAIL_TO", "").strip() or user
    if user and password:
        return MailSettings(user=user, password=password, to=to)
    return None
