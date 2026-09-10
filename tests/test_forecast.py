"""End-to-end pipeline tests through `run_tournament` with a fake wire.

The 2026-08-08 lesson, verbatim from the free-money HANDOVER: "an end-to-end test through
main() found a real bug that reasoning had not... The wiring between tested
pieces is untested by construction." These tests drive the real pipeline —
research, ensemble, aggregation, payload building, submission, containment —
with only the network faked at the Protocol seam.
"""

from __future__ import annotations

from typing import Any

from bot.cdf import validate_cdf
from bot.elicit import NUMERIC_PERCENTILES
from bot.forecast import forecast_question, run_tournament
from bot.venues.metaculus import Question

TODAY = "2026-08-18"
MODELS = ("model-a", "model-b", "model-c")


def _question(qtype: str = "binary", **overrides: Any) -> Question:
    base: dict[str, Any] = {
        "post_id": 100,
        "question_id": 200,
        "title": "Will X happen?",
        "qtype": qtype,
        "description": "Background.",
        "resolution_criteria": "Resolves YES if X.",
        "fine_print": "",
        "options": (),
        "open_time": "2026-08-10T14:00:00Z",
        "close_time": "2026-08-22T14:00:00Z",
        "unit": "",
        "range_min": None,
        "range_max": None,
        "open_lower_bound": None,
        "open_upper_bound": None,
        "zero_point": None,
    }
    base.update(overrides)
    return Question(**base)


class FakeClient:
    def __init__(self, questions: list[Question], standing: set[int] | None = None) -> None:
        self.questions = questions
        self.standing = standing or set()
        self.submitted: list[Any] = []
        self.comments: list[tuple[int, str]] = []

    def open_questions(self, tournament: str) -> list[Question]:
        return self.questions

    def forecast_standing(self, post_id: int) -> set[int]:
        return self.standing

    def submit(self, payloads: Any) -> None:
        self.submitted.extend(payloads)

    def comment(self, post_id: int, text: str) -> None:
        self.comments.append((post_id, text))


def _binary_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
    if "research assistant" in prompt:
        return "Nothing decisive found. REMAINING GAPS: none."
    if "left gaps" in prompt:
        return "No new findings."
    return "Reasoning...\nProbability: 30%"


def test_binary_question_flows_to_a_capped_submission_and_comment() -> None:
    client = FakeClient([_question()])
    results = run_tournament(
        "minibench", client, _binary_llm, MODELS, "research-model", today=TODAY
    )
    assert len(results) == 1 and results[0].submitted
    (payload,) = client.submitted
    assert payload["probability_yes"] == 0.30
    assert client.comments and client.comments[0][0] == 100
    assert "Probability: 30%" in client.comments[0][1]


def test_one_unparseable_run_is_dropped_not_fatal() -> None:
    calls = {"n": 0}

    def flaky(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt:
            return "brief. REMAINING GAPS: none"
        calls["n"] += 1
        if calls["n"] == 2:
            return "I refuse to give a number."
        return "Probability: 40%"

    client = FakeClient([_question()])
    results = run_tournament("t", client, flaky, MODELS, "r", today=TODAY)
    assert results[0].submitted
    assert results[0].runs_ok == 2
    assert results[0].runs_failed == 1


def test_a_dead_question_never_stops_the_sweep() -> None:
    """Containment: question 1's total failure is recorded; question 2 still
    gets forecast and submitted."""

    def dies_on_first(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "Will DOOMED" in prompt:
            raise RuntimeError("provider exploded")
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        return "Probability: 25%"

    doomed = _question(title="Will DOOMED happen?", question_id=201)
    fine = _question(title="Will FINE happen?", question_id=202)
    client = FakeClient([doomed, fine])
    results = run_tournament("t", client, dies_on_first, MODELS, "r", today=TODAY)
    assert results[0].error is not None and not results[0].submitted
    assert results[1].submitted
    assert len(client.submitted) == 1


def test_already_forecast_questions_are_skipped() -> None:
    """The one-forecast-per-question rule: both the API flag and the local log
    must gate resubmission."""
    api_flagged = _question(question_id=301, already_forecast=True)
    locally_logged = _question(question_id=302)
    fresh = _question(question_id=303)
    client = FakeClient([api_flagged, locally_logged, fresh])
    results = run_tournament("t", client, _binary_llm, MODELS, "r", already_done={302}, today=TODAY)
    assert [r.submitted for r in results] == [False, False, True]
    assert len(client.submitted) == 1


def test_api_standing_forecast_blocks_resubmission() -> None:
    """Found live 2026-08-18: the LIST endpoint never fills my_forecasts, so
    the list-level flag is vacuous. The detail-endpoint check must catch a
    standing forecast even when the flag and the local log both say nothing."""
    question = _question(question_id=400)  # already_forecast=False, not in log
    client = FakeClient([question], standing={400})
    results = run_tournament("t", client, _binary_llm, MODELS, "r", today=TODAY)
    assert results[0].submitted is False
    assert results[0].error == "skipped: already forecast"
    assert client.submitted == []


def test_extreme_ensemble_triggers_check_and_suspect_pulls_back() -> None:
    def certain(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        if "SOUND" in prompt and "SUSPECT" in prompt:
            return "SUSPECT\nThe resolution source may already show this resolved."
        return "Probability: 99%"

    client = FakeClient([_question()])
    results = run_tournament("t", client, certain, MODELS, "r", today=TODAY)
    assert results[0].suspect is True
    (payload,) = client.submitted
    assert payload["probability_yes"] == 0.90  # BINARY_CAP_SUSPECT, not 0.98


def test_numeric_question_produces_a_server_valid_cdf() -> None:
    def numeric_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        return "\n".join(
            f"Percentile {int(p * 100)}: {60 + 40 * p:.1f}" for p in NUMERIC_PERCENTILES
        )

    question = _question(
        qtype="numeric",
        unit="$/bbl",
        range_min=55.0,
        range_max=130.0,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    client = FakeClient([question])
    results = run_tournament("t", client, numeric_llm, MODELS, "r", today=TODAY)
    assert results[0].submitted
    (payload,) = client.submitted
    cdf = payload["continuous_cdf"]
    validate_cdf(cdf, open_lower_bound=True, open_upper_bound=True, n_points=201)


def test_multiple_choice_flows_with_exact_options() -> None:
    def mc_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        return "Texas: 20%\nArizona: 55%\nOther: 25%"

    question = _question(qtype="multiple_choice", options=("Texas", "Arizona", "Other"))
    client = FakeClient([question])
    results = run_tournament("t", client, mc_llm, MODELS, "r", today=TODAY)
    assert results[0].submitted
    (payload,) = client.submitted
    distribution = payload["probability_yes_per_category"]
    assert set(distribution) == {"Texas", "Arizona", "Other"}
    assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_research_failure_still_forecasts() -> None:
    """A research outage degrades the brief; it must never cost the submission
    window. The forecast models are told research was unavailable."""

    def no_research(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            raise RuntimeError("search provider down")
        assert "UNAVAILABLE" in prompt  # the brief admits its own thinness
        return "Probability: 50%"

    client = FakeClient([_question()])
    results = run_tournament("t", client, no_research, MODELS, "r", today=TODAY)
    assert results[0].submitted


def test_unknown_question_type_is_refused_not_guessed() -> None:
    question = _question(qtype="ranked_list")
    try:
        forecast_question(question, _binary_llm, MODELS, "brief", TODAY)
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        assert "all" in str(exc)


# --------------------------------------------------- dropped-run diagnostics
#
# 2026-08-24, first live MiniBench entry: two questions died with "all 5
# ensemble runs failed" and three more were submitted on a thinned ensemble.
# The real reason — OpenRouter HTTP 402, "you requested up to 8000 tokens but
# can only afford 4775" — was collected and then discarded, so an hour went
# into rediscovering what the provider had already said in plain English.
# These tests pin the reasons to the error, the result and the log row.


def _credit_starved_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
    """model-b is out of credits; everything else answers normally."""
    if "research assistant" in prompt or "left gaps" in prompt:
        return "Nothing decisive found. REMAINING GAPS: none."
    if model == "model-b":
        raise RuntimeError("HTTP 402: requires more credits, or fewer max_tokens")
    return "Reasoning...\nProbability: 30%"


def test_total_ensemble_failure_names_every_reason() -> None:
    """The 45464 case: when nothing survives, the raised error must carry each
    run's reason. 'all N ensemble runs failed' alone is not a diagnosis."""

    def all_starved(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        raise RuntimeError("HTTP 402: requires more credits, or fewer max_tokens")

    try:
        forecast_question(_question(), all_starved, MODELS, "brief", TODAY)
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        message = str(exc)
        assert "all 3 ensemble runs failed" in message
        assert "HTTP 402" in message
        for model in MODELS:  # every dropped run is named, not just the last
            assert model in message


def test_thinned_ensemble_keeps_each_dropped_run_reason() -> None:
    """A question that survives on fewer models still lost ensemble breadth —
    the design's largest measured edge — so the reason must survive too."""
    client = FakeClient([_question()])
    results = run_tournament("t", client, _credit_starved_llm, MODELS, "model-r", today=TODAY)
    (result,) = results
    assert result.submitted
    assert (result.runs_ok, result.runs_failed) == (2, 1)
    assert len(result.run_errors) == 1
    assert "model-b" in result.run_errors[0]
    assert "HTTP 402" in result.run_errors[0]


def test_main_logs_dropped_run_reasons_and_announces_a_thinned_ensemble(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    """Driven through main(), because the row assembly and the failure summary
    live there and nowhere else — the wiring is untested by construction."""
    import json

    from bot import forecast as fc

    class FakeLlmClient:
        total_cost_usd = 0.5
        cost_by_model = {"model-a": 0.5}

        def __init__(self, key: str, **kwargs: Any) -> None:
            self.key = key

        def models(self) -> set[str]:
            return {"model-a", "model-b", "model-r"}

        def complete(self, model: str, prompt: str, *, temperature: float = 1.0) -> str:
            return _credit_starved_llm(model, prompt, temperature=temperature)

        def key_limits(self) -> dict[str, Any]:
            return {"limit_remaining": 4.2}

        def close(self) -> None:
            return None

    class ClosableClient(FakeClient):
        def close(self) -> None:
            return None

    monkeypatch.setenv("METAC_FORECAST_MODELS", "model-a,model-b")
    monkeypatch.setenv("METAC_RESEARCH_MODEL", "model-r")
    monkeypatch.setattr(fc, "metaculus_token", lambda: "token")
    monkeypatch.setattr(fc, "openrouter_key", lambda: "key")
    monkeypatch.setattr(fc, "openrouter_personal_key", lambda: None)
    monkeypatch.setattr(fc, "asknews_credentials", lambda: None)
    monkeypatch.setattr(fc, "MetaculusClient", lambda token: ClosableClient([_question()]))
    monkeypatch.setattr("bot.venues.llm.OpenRouterClient", FakeLlmClient)
    monkeypatch.setattr(fc, "FORECAST_LOG", tmp_path / "forecasts.jsonl")
    monkeypatch.setattr(fc, "POLL_LOG", tmp_path / "polls.jsonl")

    assert fc.main(["--tournament", "t"]) == 0

    row = json.loads((tmp_path / "forecasts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["submitted"] is True
    assert (row["runs_ok"], row["runs_failed"]) == (1, 1)
    assert len(row["run_errors"]) == 1
    assert "HTTP 402" in row["run_errors"][0]

    printed = capsys.readouterr().out
    assert "THINNED" in printed  # a quiet degradation is now a loud one
    assert "HTTP 402" in printed


# ------------------------------------------------------ two keys, one roster
#
# 2026-09-10: the donated key's OpenRouter account refuses x-ai (HTTP 404, "your
# account's allowed-providers setting permits only: openai, anthropic,
# google-ai-studio"), and Grok was the fourth family in the 08-24 roster that
# scored +23.9/q. It runs on the owner's own key; everything else stays donated.


class KeyedLlmClient:
    """Records which key served which model."""

    served: list[tuple[str, str]] = []

    def __init__(self, key: str, **kwargs: Any) -> None:
        self.key = key
        self.total_cost_usd = 0.0
        self.cost_by_model: dict[str, float] = {}

    def models(self) -> set[str]:
        return {"model-a", "x-ai/grok-test", "model-r"}

    def complete(self, model: str, prompt: str, *, temperature: float = 1.0) -> str:
        KeyedLlmClient.served.append((self.key, model))
        return _binary_llm(model, prompt, temperature=temperature)

    def key_limits(self) -> dict[str, Any]:
        return {"limit_remaining": 99.0 if self.key == "donated" else 4.5}

    def close(self) -> None:
        return None


class ClosableFakeClient(FakeClient):
    def close(self) -> None:
        return None


def _two_key_env(monkeypatch: Any, tmp_path: Any, personal: str | None) -> Any:
    from bot import forecast as fc

    KeyedLlmClient.served = []
    monkeypatch.setenv("METAC_FORECAST_MODELS", "model-a,x-ai/grok-test")
    monkeypatch.setenv("METAC_RESEARCH_MODEL", "model-r")
    monkeypatch.setattr(fc, "metaculus_token", lambda: "token")
    monkeypatch.setattr(fc, "openrouter_key", lambda: "donated")
    monkeypatch.setattr(fc, "openrouter_personal_key", lambda: personal)
    monkeypatch.setattr(fc, "asknews_credentials", lambda: None)
    monkeypatch.setattr(fc, "MetaculusClient", lambda token: ClosableFakeClient([_question()]))
    monkeypatch.setattr("bot.venues.llm.OpenRouterClient", KeyedLlmClient)
    monkeypatch.setattr(fc, "FORECAST_LOG", tmp_path / "forecasts.jsonl")
    monkeypatch.setattr(fc, "POLL_LOG", tmp_path / "polls.jsonl")
    return fc


def _first_row(path: Any) -> dict[str, Any]:
    import json

    return dict(json.loads(path.read_text(encoding="utf-8").splitlines()[0]))


def test_grok_runs_on_the_personal_key_and_everything_else_on_the_donated_one(
    tmp_path: Any, monkeypatch: Any
) -> None:
    fc = _two_key_env(monkeypatch, tmp_path, personal="personal")

    assert fc.main(["--tournament", "t"]) == 0

    served = set(KeyedLlmClient.served)
    assert ("personal", "x-ai/grok-test") in served
    assert {("donated", "model-a"), ("donated", "model-r")} <= served
    assert all(model.startswith("x-ai/") for key, model in served if key == "personal")
    assert not any(model.startswith("x-ai/") for key, model in served if key == "donated")
    row = _first_row(tmp_path / "forecasts.jsonl")
    assert (row["runs_ok"], row["runs_failed"]) == (2, 0)  # the whole ensemble, none thinned


def test_without_a_personal_key_grok_is_dropped_not_sent_to_a_key_that_refuses_it(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    """Sent to the donated key, Grok would fail on every question and every row
    would read THINNED, burying the failures that matter under an expected one."""
    fc = _two_key_env(monkeypatch, tmp_path, personal=None)

    assert fc.main(["--tournament", "t"]) == 0

    assert all(model != "x-ai/grok-test" for _, model in KeyedLlmClient.served)
    row = _first_row(tmp_path / "forecasts.jsonl")
    assert (row["runs_ok"], row["runs_failed"]) == (1, 0)  # dropped at startup, not per question
    assert "OPENROUTER_PERSONAL_KEY absent" in capsys.readouterr().out


def test_the_poll_log_records_both_keys_balances(tmp_path: Any, monkeypatch: Any) -> None:
    """The personal key runs dry on its own schedule; the log must show it."""
    fc = _two_key_env(monkeypatch, tmp_path, personal="personal")

    fc.main(["--tournament", "t"])

    row = _first_row(tmp_path / "polls.jsonl")
    assert row["credits_remaining"] == 99.0
    assert row["personal_credits_remaining"] == 4.5


# ------------------------------------------------- the answer, in words
#
# 2026-09-10: the owner is mailed what the bot answered. The payload is not
# readable (a 201-point CDF), so the pipeline records the answer in words.


def test_the_answer_is_recorded_in_words_for_every_question_type() -> None:
    binary = run_tournament("t", FakeClient([_question()]), _binary_llm, MODELS, "r", today=TODAY)
    assert binary[0].answer == "30%"

    def mc_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        return "Texas: 20%\nArizona: 55%\nOther: 25%"

    mc_question = _question(qtype="multiple_choice", options=("Texas", "Arizona", "Other"))
    mc = run_tournament("t", FakeClient([mc_question]), mc_llm, MODELS, "r", today=TODAY)
    assert mc[0].answer == "Arizona 55%, Other 25%, Texas 20%"

    def numeric_llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if "research assistant" in prompt or "left gaps" in prompt:
            return "brief. REMAINING GAPS: none"
        return "\n".join(
            f"Percentile {int(p * 100)}: {60 + 40 * p:.1f}" for p in NUMERIC_PERCENTILES
        )

    numeric_question = _question(
        qtype="numeric",
        unit="$/bbl",
        range_min=55.0,
        range_max=130.0,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    numeric = run_tournament(
        "t", FakeClient([numeric_question]), numeric_llm, MODELS, "r", today=TODAY
    )
    assert numeric[0].answer == "median 80 $/bbl (80% range 64–96 $/bbl)"


def test_date_answers_read_as_dates_and_tails_keep_a_decimal() -> None:
    import datetime as dt

    from bot.forecast import _describe_percentiles, _percent

    stamps = {0.1: 1790000000.0, 0.5: 1795000000.0, 0.9: 1800000000.0}
    day = [
        dt.datetime.fromtimestamp(stamps[p], dt.UTC).strftime("%Y-%m-%d") for p in (0.5, 0.1, 0.9)
    ]
    text = _describe_percentiles(_question(qtype="date"), stamps)
    assert text == f"median {day[0]} (80% range {day[1]}–{day[2]})"
    assert (_percent(0.3), _percent(0.015), _percent(0.985)) == ("30%", "1.5%", "98.5%")
