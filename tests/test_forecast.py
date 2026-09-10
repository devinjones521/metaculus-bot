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
