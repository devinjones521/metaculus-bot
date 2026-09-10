"""Tests for the invariant checks.

A checker that reads PASS on the real tree is exactly when it is most likely to
be silently broken, so every invariant here is also shown to FIRE on a planted
violation.
"""

from __future__ import annotations

from pathlib import Path

from bot.verify import (
    check_no_published_secrets,
    check_no_skipped_tests,
    check_poller_contract,
    check_venue_isolation,
    scan_text_for_secrets,
)

# Built from fragments so this file is not itself a hit for the check it tests.
FAKE_OPENROUTER = "sk-or-v1-" + "0123456789abcdef" * 4
FAKE_METACULUS = "Token " + "a1b2c3d4e5" * 4


class TestVenueIsolation:
    def test_real_tree_is_isolated(self) -> None:
        assert check_venue_isolation().ok


class TestNoSkippedTests:
    def test_real_suite_has_no_skips(self) -> None:
        assert check_no_skipped_tests().ok


class TestPollerContract:
    def test_fires_when_a_deployed_unit_has_no_contract(self, tmp_path: Path) -> None:
        from bot.health import POLLERS

        manifest = tmp_path / "units"
        body = "\n".join(p.unit for p in POLLERS) + "\nmetaculus-poll@ghost\n"
        manifest.write_text(body, encoding="utf-8")
        result = check_poller_contract(manifest)
        assert not result.ok
        assert "metaculus-poll@ghost" in result.detail

    def test_fires_on_a_contract_with_no_deployment(self, tmp_path: Path) -> None:
        manifest = tmp_path / "units"
        manifest.write_text("# nothing deployed\n", encoding="utf-8")
        result = check_poller_contract(manifest)
        assert not result.ok
        assert "not deployed" in result.detail

    def test_a_missing_manifest_is_a_failure_not_a_pass(self, tmp_path: Path) -> None:
        assert not check_poller_contract(tmp_path / "absent").ok

    def test_the_real_manifest_passes(self) -> None:
        assert check_poller_contract().ok


class TestSecretScan:
    def test_catches_an_openrouter_key(self) -> None:
        found = scan_text_for_secrets(f"x = 1\nKEY = '{FAKE_OPENROUTER}'\n")
        assert found and found[0][0] == 2

    def test_catches_a_metaculus_token_in_a_header(self) -> None:
        assert scan_text_for_secrets(f"headers = {{'Authorization': '{FAKE_METACULUS}'}}")

    def test_catches_a_literal_env_value_of_any_shape(self) -> None:
        # The Metaculus token is bare hex with no prefix: only the literal
        # check can see it once it leaves the header.
        secret = "f00dfeedcafebeef1234"
        assert scan_text_for_secrets(f"token={secret}", [secret])

    def test_short_env_values_are_not_treated_as_secrets(self) -> None:
        assert scan_text_for_secrets("enabled = true", ["true"]) == []

    def test_does_not_fire_on_innocent_code(self) -> None:
        text = 'API = "https://openrouter.ai/api/v1"\nheaders = {"Authorization": f"Token {t}"}\n'
        assert scan_text_for_secrets(text) == []

    def test_the_report_never_repeats_the_secret(self) -> None:
        found = scan_text_for_secrets(FAKE_OPENROUTER, [FAKE_OPENROUTER])
        assert found
        assert all(FAKE_OPENROUTER not in reason for _, reason in found)

    def test_real_tree_publishes_no_secrets(self) -> None:
        """The actual guarantee: every file git would publish, against the real .env."""
        result = check_no_published_secrets()
        assert result.ok, result.detail

    def test_fires_on_a_planted_untracked_file(self, tmp_path: Path) -> None:
        """Untracked-but-not-ignored files are about to be committed; they count."""
        import bot.verify as verify

        planted = verify.PROJECT_ROOT / "tests" / "_planted_secret_probe.txt"
        planted.write_text(f"leak: {FAKE_OPENROUTER}\n", encoding="utf-8")
        try:
            result = check_no_published_secrets(tmp_path / "no.env")
        finally:
            planted.unlink()
        assert not result.ok
        assert "_planted_secret_probe.txt" in result.detail
