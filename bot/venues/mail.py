"""Outbound email: the funding alerts, and nothing else. Gmail SMTP on port 587.

WHY 587 AND STARTTLS
--------------------

Measured from the Hetzner box on 2026-09-10: outbound 25 and 465 time out
(Hetzner blocks both by default) and 587 connects. So this is SMTP submission
on 587, upgraded with STARTTLS before the login is sent — never plaintext.

The login is a Gmail *app password*, not the account password: it can send
mail and nothing else, and revoking it breaks nothing but these alerts.

HOW IT COULD LIE
----------------

- **A send that fails quietly is an alert nobody got.** `send_mail` raises on
  every failure. The caller decides what that means; for alerts it means the
  level is not recorded as mailed, so the next tick tries again.
- **Tests prove the message, not the path.** `uv run python -m bot.alerts
  --test`, run on the box, is what proves the path end to end.
"""

from __future__ import annotations

import smtplib
import ssl
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

from bot.config import MailSettings

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
TIMEOUT_SECONDS = 30.0


def send_mail(
    settings: MailSettings,
    subject: str,
    body: str,
    *,
    smtp_factory: Callable[..., Any] | None = None,
) -> None:
    """Send one plain-text email. Raises on any failure: never a silent drop."""
    message = EmailMessage()
    message["From"] = settings.user
    message["To"] = settings.to
    message["Subject"] = subject
    message.set_content(body)
    factory = smtp_factory or smtplib.SMTP
    with factory(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_SECONDS) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(settings.user, settings.password)
        smtp.send_message(message)
