"""The mail venue against a fake SMTP server, and the settings that feed it."""

from __future__ import annotations

import smtplib
from typing import Any

import pytest

from bot import config
from bot.config import MailSettings, alert_mail_settings
from bot.venues.mail import send_mail

SETTINGS = MailSettings(user="bot@example.com", password="fake-app-password", to="me@example.com")


class FakeSMTP:
    """Records the conversation in order."""

    def __init__(self, calls: list[tuple[Any, ...]], host: str, port: int, timeout: float) -> None:
        self.calls = calls
        self.calls.append(("connect", host, port))

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append(("quit",))

    def starttls(self, context: Any) -> None:
        self.calls.append(("starttls",))

    def login(self, user: str, password: str) -> None:
        self.calls.append(("login", user))

    def send_message(self, message: Any) -> None:
        self.calls.append(("send", message["To"], message["Subject"], message.get_content()))


def test_tls_is_up_before_the_password_is_sent() -> None:
    calls: list[tuple[Any, ...]] = []
    send_mail(SETTINGS, "subject", "body", smtp_factory=lambda *a, **k: FakeSMTP(calls, *a, **k))
    assert [c[0] for c in calls] == ["connect", "starttls", "login", "send", "quit"]
    assert calls[0][1:] == ("smtp.gmail.com", 587)  # the box blocks 25 and 465
    assert calls[3][1:3] == ("me@example.com", "subject")
    assert calls[3][3].strip() == "body"


def test_a_refused_login_raises_rather_than_dropping_the_alert() -> None:
    class Refusing(FakeSMTP):
        def login(self, user: str, password: str) -> None:
            raise smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")

    calls: list[tuple[Any, ...]] = []
    with pytest.raises(smtplib.SMTPAuthenticationError):
        send_mail(SETTINGS, "s", "b", smtp_factory=lambda *a, **k: Refusing(calls, *a, **k))
    assert "send" not in [c[0] for c in calls]


def test_settings_accept_a_pasted_app_password_and_never_print_it(monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda path=None: {})
    monkeypatch.setenv("ALERT_SMTP_USER", "bot@example.com")
    monkeypatch.setenv("ALERT_SMTP_PASSWORD", "abcd efgh ijkl mnop")  # as Google displays it
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)

    settings = alert_mail_settings()

    assert settings is not None
    assert settings.password == "abcdefghijklmnop"
    assert settings.to == "bot@example.com"  # the owner mails themselves by default
    assert "abcd" not in repr(settings)


def test_alerts_are_off_without_a_password(monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda path=None: {})
    monkeypatch.setenv("ALERT_SMTP_USER", "bot@example.com")
    monkeypatch.delenv("ALERT_SMTP_PASSWORD", raising=False)
    assert alert_mail_settings() is None
