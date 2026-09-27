"""The forecaster: research → ensemble → aggregate → submit, per open question.

    uv run python -m bot.forecast --tournament minibench --dry-run
    uv run python -m bot.forecast --tournament bot-testing-area   # live rehearsal
    uv run python -m bot.forecast --tournament minibench          # the real thing

WHAT THIS IS
------------

The design (`docs/METHOD.md`) is the Fall 2025 evidence base, not taste:
multiple research sources (r=0.42 with score), a 5-run ensemble with median
aggregation (+1,799 points measured), tail caps (r=+0.48 within winners),
explicit base-rate prompting (r=+0.38), an
already-resolved guard (the known catastrophic mode), and CDF construction in
code (the known 30%-failure mode). Roughly 8–12 LLM calls per question against
winners' median of 28 — a deliberate first-cycle floor, to be tuned on
MiniBench feedback, not in advance of it.

RELIABILITY BEFORE CLEVERNESS
-----------------------------

The named anti-pattern in Metaculus's own synthesis is "adding scaffolding
before getting reliability right" (a units bug: −80 points; ten missed
questions: −150). Questions open at random hours for 1.5–3 hours, only the
last forecast before close counts, and sum-of-scores ranking makes a missed
question a real loss. Hence:

  - any single question's failure is contained: one bad question never stops
    the sweep;
  - a question is submitted if at least ONE ensemble run parsed — a lone
    frontier-model forecast still beats absence;
  - every submission and every poll appends to data/metaculus/ so
    `bot.health` can later prove liveness and so post-mortems and Platt
    recalibration have rows to work with (the calibration analysis needs ~500).

HOW IT COULD LIE
----------------

- **A clean sweep over zero questions looks identical to a broken listing.**
  The poll log records questions_seen for every run; rule 6 applies before
  believing quiet output.
- **The one-forecast rule.** Bot tournaments want ONE forecast per question.
  Both the API's my_forecasts flag and the local submission log gate
  resubmission; --resubmit exists for the testing area only.
- **Dry runs must not be able to submit.** --dry-run swaps the client's write
  methods for recorders at the seam; there is no 'if' deep in the pipeline to
  get wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bot.cdf import build_cdf, validate_cdf
from bot.config import (
    PROJECT_ROOT,
    asknews_credentials,
    load_dotenv,
    metaculus_token,
    openrouter_key,
    openrouter_personal_key,
)
from bot.elicit import (
    ParseError,
    binary_prompt,
    extreme_check_prompt,
    gap_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    parse_binary,
    parse_extreme_check,
    parse_multiple_choice,
    parse_numeric,
    research_prompt,
)
from bot.ensemble import (
    aggregate_binary,
    aggregate_multiple_choice,
    aggregate_percentiles,
    is_extreme,
)
from bot.venues.metaculus import (
    MetaculusClient,
    Question,
    binary_payload,
    multiple_choice_payload,
    numeric_payload,
)

DATA_DIR = PROJECT_ROOT / "data" / "metaculus"
FORECAST_LOG = DATA_DIR / "forecasts.jsonl"
POLL_LOG = DATA_DIR / "polls.jsonl"

# The ensemble roster. Diversity across model families is the evidenced shape
# (Mantic's 4-family ensemble; nostreambot's decorrelation member). This is the
# 2026-08-24 roster that scored +23.9/q (n=10, se 7.3) with one substitution:
# Gemini 3.1 Pro 429'd on the donated key from 2026-09-10, and 3.8 Flash took
# its slot (evidence in docs/METHOD.md). Fable is sampled twice on purpose, and
# first, because the extremeness check runs on models[0]. Grok needs
# OPENROUTER_PERSONAL_KEY (see PERSONAL_KEY_PROVIDERS) and is dropped without it.
# Gemini is tried on the donated key and, when that fails, once more on the
# personal key (see FALLBACK_TO_PERSONAL_KEY).
# Overridable via METAC_FORECAST_MODELS / METAC_RESEARCH_MODEL without a code
# change, and validated against the live OpenRouter id list at startup so a
# typo fails before question one, not during question five.
DEFAULT_FORECAST_MODELS = (
    "anthropic/claude-fable-5",
    "anthropic/claude-fable-5",
    "openai/gpt-5.5",
    "google/gemini-3.8-flash",
    "x-ai/grok-4.6",
)
DEFAULT_RESEARCH_MODEL = "anthropic/claude-sonnet-5:online"

MAX_RESEARCH_CHARS = 9000  # context hygiene: the forecaster reads a brief, not a dump
MAX_COMMENT_CHARS = 8000
MAX_RUN_ERROR_CHARS = 240  # the provider's own message, kept; not its stack


class TournamentClient(Protocol):
    """The three verbs the pipeline needs. A Protocol so the end-to-end test
    can drive the REAL pipeline with a fake wire — the untested wiring between
    tested pieces is this project's most repeated defect."""

    def open_questions(self, tournament: str) -> list[Question]: ...

    def forecast_standing(self, post_id: int) -> set[int]: ...

    def submit(
        self, payloads: Sequence[Mapping[str, Any]], *, post_id: int | None = None
    ) -> None: ...

    def comment(self, post_id: int, text: str) -> None: ...


class DryRunClient:
    """Reads through to the real client; writes print instead of submitting.

    A wrapper at the seam, not an `if` deep in the pipeline: a dry run must be
    structurally incapable of submitting, not merely expected not to."""

    def __init__(self, inner: TournamentClient) -> None:
        self._inner = inner

    def open_questions(self, tournament: str) -> list[Question]:
        return self._inner.open_questions(tournament)

    def forecast_standing(self, post_id: int) -> set[int]:
        return self._inner.forecast_standing(post_id)

    def submit(self, payloads: Sequence[Mapping[str, Any]], *, post_id: int | None = None) -> None:
        print(f"[dry-run] submit: {list(payloads)}")

    def comment(self, post_id: int, text: str) -> None:
        print(f"[dry-run] comment on {post_id}: {text[:200]}...")


@dataclass
class ForecastResult:
    question_id: int
    post_id: int
    title: str
    qtype: str
    submitted: bool
    forecast: Any = None
    runs_ok: int = 0
    runs_failed: int = 0
    suspect: bool = False
    run_errors: list[str] = field(default_factory=list)
    # Runs that failed on the donated key and then answered on the personal
    # one: not dropped, but not healthy either (see FALLBACK_TO_PERSONAL_KEY).
    fallbacks: list[str] = field(default_factory=list)
    error: str | None = None
    # Set when the forecast went in but its reasoning did not. Separate from
    # `error`, which means nothing was submitted: a question that is answered
    # but uncommented is not a failed question, it is an ineligible one, and
    # no later sweep will revisit it — the standing forecast makes it skip.
    comment_error: str | None = None
    log: list[str] = field(default_factory=list)
    answer: str = ""  # the forecast in words, for the owner's email


# ------------------------------------------------------------------ research


def gather_research(
    question: Question,
    llm: Callable[..., str],
    research_model: str,
    asknews_search: Callable[[str], list[Any] | None] | None,
    today: str,
) -> str:
    """Two-round agentic-lite web research plus AskNews, merged into one brief.

    Every source can fail independently; the brief says which sources actually
    contributed so a thin brief is visibly thin rather than quietly thin.
    """
    sections: list[str] = []
    briefing = ""
    try:
        briefing = llm(research_model, research_prompt(question, today), temperature=0.3)
        sections.append("== WEB RESEARCH (round 1) ==\n" + briefing.strip())
    except Exception as exc:  # noqa: BLE001
        sections.append(f"== WEB RESEARCH (round 1) == UNAVAILABLE ({exc})")
    if briefing and "REMAINING GAPS" in briefing.upper():
        try:
            followup = llm(research_model, gap_prompt(question, briefing, today), temperature=0.3)
            sections.append("== WEB RESEARCH (round 2, gap-filling) ==\n" + followup.strip())
        except Exception as exc:  # noqa: BLE001
            sections.append(f"== WEB RESEARCH (round 2) == UNAVAILABLE ({exc})")
    if asknews_search is not None:
        articles = asknews_search(question.title)
        if articles is None:
            sections.append("== NEWS WIRE == UNAVAILABLE (AskNews call failed)")
        elif articles:
            rendered = "\n\n".join(a.render() for a in articles[:8])
            sections.append("== NEWS WIRE (AskNews) ==\n" + rendered)
        else:
            sections.append("== NEWS WIRE (AskNews) == no matching articles found")
    brief = "\n\n".join(sections)
    return brief[:MAX_RESEARCH_CHARS]


# ----------------------------------------------------------------- pipeline


def _percent(p: float) -> str:
    """37%, or 1.5% / 98.5% in the tails, where a whole percent would hide the forecast."""
    value = p * 100
    return f"{value:.1f}%" if value < 10 or value > 90 else f"{value:.0f}%"


def _number(value: float, qtype: str) -> str:
    if qtype == "date" and value > 1e8:  # date questions are scaled in Unix seconds
        return dt.datetime.fromtimestamp(value, dt.UTC).strftime("%Y-%m-%d")
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _describe_percentiles(question: Question, percentiles: Mapping[float, float]) -> str:
    """The submitted distribution in words: its median and 80% range, in the question's unit."""
    unit = f" {question.unit}" if question.unit and question.qtype != "date" else ""
    q = question.qtype
    return (
        f"median {_number(percentiles[0.5], q)}{unit} "
        f"(80% range {_number(percentiles[0.1], q)}–{_number(percentiles[0.9], q)}{unit})"
    )


def forecast_question(
    question: Question,
    llm: Callable[..., str],
    models: tuple[str, ...],
    research: str,
    today: str,
    *,
    fallback: Callable[..., str] | None = None,
) -> tuple[dict[str, Any], str, int, int, bool, list[str], str, list[str]]:
    """One question through ensemble → aggregate → payload.

    Returns (payload, rationale, runs_ok, runs_failed, suspect, run_errors,
    answer, fallbacks). Raises only if NO run parsed — the caller records that
    as a loud failure.

    Every dropped run's reason is kept and returned, because a dropped run costs
    ensemble breadth whether or not the question survives. On 2026-08-24 three
    forecasts went in on a thinned ensemble and two questions died outright,
    while the reasons — plain HTTP 402 'out of credits' — were discarded right
    here, leaving 'all 5 ensemble runs failed' as the only surviving evidence.

    A run on a FALLBACK_TO_PERSONAL_KEY model that fails is tried once more
    through `fallback`. One that then answers is not a dropped run, but its
    first failure is returned in `fallbacks`, so a slot that only ever answers
    on its second route stays visible instead of reading as healthy.
    """
    runs: list[Any] = []
    rationales: list[str] = []
    run_errors: list[str] = []
    fallbacks: list[str] = []
    for model in models:
        try:
            text, run = _one_run(question, llm, model, research, today)
            rationales.append(f"### Run ({model})\n{text.strip()}")
            runs.append(run)
            continue
        except Exception as exc:  # noqa: BLE001 - a bad run is dropped, loudly
            why = f"{type(exc).__name__}: {exc}"[:MAX_RUN_ERROR_CHARS]
        if fallback is not None and falls_back_to_personal_key(model):
            try:
                text, run = _one_run(question, fallback, model, research, today)
                rationales.append(f"### Run ({model}, second route)\n{text.strip()}")
                runs.append(run)
                fallbacks.append(f"{model}: {why}")
                continue
            except Exception as exc:  # noqa: BLE001 - both routes failed; both reasons kept
                second = f"{type(exc).__name__}: {exc}"[:MAX_RUN_ERROR_CHARS]
                why = f"{why} | then on the personal key: {second}"
        run_errors.append(f"{model}: {why}")
        rationales.append(f"### Run ({model}) FAILED: {why}")
    if not runs:
        raise RuntimeError(f"all {len(models)} ensemble runs failed — " + " | ".join(run_errors))

    suspect = False
    if question.qtype == "binary":
        raw = aggregate_binary(runs)
        if is_extreme(raw):
            try:
                verdict = llm("check", extreme_check_prompt(question, raw, today), temperature=0.2)
                suspect = parse_extreme_check(verdict)
                rationales.append(f"### Extremeness check\n{verdict.strip()}")
            except Exception as exc:  # noqa: BLE001 - unverifiable extreme = suspect
                suspect = True
                rationales.append(f"### Extremeness check FAILED ({exc}) — treating as SUSPECT")
        final = aggregate_binary(runs, suspect=suspect)
        answer = _percent(final)
        payload = binary_payload(question.question_id, round(final, 4))
    elif question.qtype == "multiple_choice":
        distribution = aggregate_multiple_choice(runs, question.options)
        ranked = sorted(distribution.items(), key=lambda item: -item[1])
        answer = ", ".join(f"{option} {_percent(p)}" for option, p in ranked)
        payload = multiple_choice_payload(question.question_id, distribution, question.options)
    else:
        percentiles = aggregate_percentiles(runs)
        answer = _describe_percentiles(question, percentiles)
        if question.range_min is None or question.range_max is None:
            raise RuntimeError("continuous question without scaling range")
        cdf = build_cdf(
            percentiles,
            range_min=question.range_min,
            range_max=question.range_max,
            zero_point=question.zero_point,
            open_lower_bound=bool(question.open_lower_bound),
            open_upper_bound=bool(question.open_upper_bound),
            n_points=question.cdf_points,
        )
        validate_cdf(
            cdf,
            open_lower_bound=bool(question.open_lower_bound),
            open_upper_bound=bool(question.open_upper_bound),
            n_points=question.cdf_points,
        )
        payload = numeric_payload(question.question_id, cdf, question.cdf_points)

    rationale = "\n\n".join(rationales)[:MAX_COMMENT_CHARS]
    return payload, rationale, len(runs), len(run_errors), suspect, run_errors, answer, fallbacks


def _one_run(
    question: Question,
    llm: Callable[..., str],
    model: str,
    research: str,
    today: str,
) -> tuple[str, Any]:
    """One model's answer and its parse. Raises on either failing."""
    if question.qtype == "binary":
        text = llm(model, binary_prompt(question, research, today))
        return text, parse_binary(text)
    if question.qtype == "multiple_choice":
        text = llm(model, multiple_choice_prompt(question, research, today))
        return text, parse_multiple_choice(text, question.options)
    if question.qtype in ("numeric", "discrete", "date"):
        text = llm(model, numeric_prompt(question, research, today))
        return text, parse_numeric(text)
    raise ParseError(f"unhandled question type {question.qtype!r}")


def run_tournament(
    tournament: str,
    client: TournamentClient,
    llm: Callable[..., str],
    models: tuple[str, ...],
    research_model: str,
    *,
    asknews_search: Callable[[str], list[Any] | None] | None = None,
    already_done: set[int] | None = None,
    resubmit: bool = False,
    limit: int | None = None,
    today: str | None = None,
    fallback: Callable[..., str] | None = None,
) -> list[ForecastResult]:
    """Sweep one tournament. One question's failure never stops the sweep."""
    today = today or dt.date.today().isoformat()
    already_done = already_done or set()
    questions = client.open_questions(tournament)
    results: list[ForecastResult] = []
    todo: list[Question] = []
    # One detail call per post, cached: the list endpoint's my_forecasts is
    # always empty (verified live), so the API-side dedup check must read the
    # detail endpoint or it is a guard that passes by being vacuous.
    standing_cache: dict[int, set[int]] = {}
    for question in questions:
        skip = False
        if not resubmit:
            if question.already_forecast or question.question_id in already_done:
                skip = True
            else:
                if question.post_id not in standing_cache:
                    standing_cache[question.post_id] = client.forecast_standing(question.post_id)
                skip = question.question_id in standing_cache[question.post_id]
        if skip:
            results.append(
                ForecastResult(
                    question_id=question.question_id,
                    post_id=question.post_id,
                    title=question.title,
                    qtype=question.qtype,
                    submitted=False,
                    error="skipped: already forecast",
                )
            )
            continue
        todo.append(question)
    if limit is not None:
        todo = todo[:limit]

    for question in todo:
        result = ForecastResult(
            question_id=question.question_id,
            post_id=question.post_id,
            title=question.title,
            qtype=question.qtype,
            submitted=False,
        )
        try:
            research = gather_research(question, llm, research_model, asknews_search, today)
            (payload, rationale, ok, failed, suspect, run_errors, answer, fallbacks) = (
                forecast_question(question, llm, models, research, today, fallback=fallback)
            )
            # Written down BEFORE the network call. When a submission died on
            # the wire on 2026-09-21 the row read `runs_ok: 0, forecast: null`,
            # which is the shape of a question nothing was computed for — while
            # five runs had in fact been paid for and only the write was lost.
            result.forecast = _summarise_payload(payload)
            result.runs_ok, result.runs_failed, result.suspect = ok, failed, suspect
            result.run_errors = run_errors
            result.fallbacks = fallbacks
            result.answer = answer
            # Under `resubmit` a standing forecast may predate this sweep, so it
            # cannot tell the venue whether a dropped write landed.
            client.submit([payload], post_id=None if resubmit else question.post_id)
            result.submitted = True
        except Exception as exc:  # noqa: BLE001 - contain, record, continue
            result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue
        # A second write, which fails on its own terms. The forecast is already
        # on the board, so this is never `error`: q45942 was logged as FAILED
        # on 2026-09-21 with its forecast standing on Metaculus.
        try:
            client.comment(question.post_id, rationale)
        except Exception as exc:  # noqa: BLE001 - recorded, loudly, and carried on
            result.comment_error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def _summarise_payload(payload: dict[str, Any]) -> Any:
    if payload.get("probability_yes") is not None:
        return payload["probability_yes"]
    if payload.get("probability_yes_per_category") is not None:
        return payload["probability_yes_per_category"]
    cdf = payload.get("continuous_cdf") or []
    return {"cdf_points": len(cdf), "cdf_head": cdf[:3], "cdf_tail": cdf[-3:]}


# ------------------------------------------------------------------ logging


def previously_forecast_ids() -> set[int]:
    if not FORECAST_LOG.exists():
        return set()
    ids: set[int] = set()
    for line in FORECAST_LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("submitted"):
            ids.add(int(row["question_id"]))
    return ids


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------- cli


def _env_models() -> tuple[tuple[str, ...], str]:
    env = {**load_dotenv(), **os.environ}
    roster = env.get("METAC_FORECAST_MODELS", "").strip()
    research = env.get("METAC_RESEARCH_MODEL", "").strip() or DEFAULT_RESEARCH_MODEL
    models = (
        tuple(m.strip() for m in roster.split(",") if m.strip())
        if roster
        else DEFAULT_FORECAST_MODELS
    )
    return models, research


def _validate_roster(available: set[str], models: Iterable[str]) -> list[str]:
    """Roster ids not present on OpenRouter (':online' suffix stripped)."""
    return [m for m in models if re.sub(r":online$", "", m) not in available]


class SetupError(RuntimeError):
    """Configuration is missing or wrong. Carries the operator-facing message."""


# Providers the donated key's OpenRouter account refuses, so their models run on
# the owner's own key. Measured 2026-09-10: x-ai/grok-4.6 returned HTTP 404,
# "your account's allowed-providers setting permits only: openai, anthropic,
# google-ai-studio". That setting is Metaculus's, not ours to change.
PERSONAL_KEY_PROVIDERS = ("x-ai/",)


def needs_personal_key(model: str) -> bool:
    return model.startswith(PERSONAL_KEY_PROVIDERS)


# Models that run on the donated key first and, when that fails, once more on
# the personal key. The donated key reaches Gemini through Metaculus's own
# Google account (`is_byok: true`, $0 to this bot), and that route failed 50
# times in the 60 warmup questions of 2026-09-21..23: 32 quota 429s, 12
# in-band "high demand" errors that were logged as empty completions, and 6
# dropped connections. Gemini 3.5 Flash is no way round it: probed on
# 2026-09-27, it failed on the same route in the same minutes (3 of 3, "high
# demand") as 3.8 did (4 of 4). The personal key reaches the same model through
# OpenRouter's own Google capacity, and answered 2 of 2 in the minute the
# donated route refused, at about $0.01 a run.
FALLBACK_TO_PERSONAL_KEY = ("google/",)


def falls_back_to_personal_key(model: str) -> bool:
    return model.startswith(FALLBACK_TO_PERSONAL_KEY)


def _balance(llm_client: Any) -> float | None:
    """Dollars left on one OpenRouter key, or None if the key won't say.

    None means UNKNOWN, and every caller must treat it as such. Returning
    0.0 on a failed lookup would stop the poller dead on a transient
    network blip; returning a large number would let it spend into a wall.
    """
    try:
        limits = llm_client.key_limits()
    except Exception:  # noqa: BLE001 - an unreadable balance is not a fatal one
        return None
    value = limits.get("limit_remaining")
    return float(value) if isinstance(value, int | float) else None


@dataclass
class Session:
    """Everything a sweep needs, built once.

    Built once and reused because `bot.poll` sweeps the same tournament for
    days: re-validating the roster and reopening sockets every 20 minutes is
    waste, and a mid-cycle roster typo should fail before question one of the
    first sweep, not silently on hour nine.
    """

    client: TournamentClient
    llm_client: Any
    llm: Callable[..., str]
    models: tuple[str, ...]
    research_model: str
    asknews_search: Callable[[str], list[Any] | None] | None
    metaculus_client: Any
    # The owner's own key, for PERSONAL_KEY_PROVIDERS. None when it is not set.
    personal_llm_client: Any = None
    # The second route for FALLBACK_TO_PERSONAL_KEY models. None without that key.
    fallback_llm: Callable[..., str] | None = None

    def credits_remaining(self) -> float | None:
        """Dollars left on the donated key, which pays for research and most runs."""
        return _balance(self.llm_client)

    def personal_credits_remaining(self) -> float | None:
        """Dollars left on the personal key; None if there is none or it won't say."""
        return None if self.personal_llm_client is None else _balance(self.personal_llm_client)

    def llm_clients(self) -> list[Any]:
        return [c for c in (self.llm_client, self.personal_llm_client) if c is not None]

    def close(self) -> None:
        for closable in (*self.llm_clients(), self.metaculus_client):
            close = getattr(closable, "close", None)
            if close is not None:
                close()


def build_session(*, dry_run: bool = False) -> Session:
    """Validate configuration and open the clients. Raises SetupError."""
    token = metaculus_token()
    if not token:
        raise SetupError("METACULUS_TOKEN missing from .env — cannot even read questions.")
    key = openrouter_key()
    if not key:
        raise SetupError(
            "OPENROUTER_API_KEY missing from .env — the forecaster has no model access.\n"
            "Either the Metaculus credit key (form on the resources page) or a personal\n"
            "OpenRouter key unblocks this; nothing else in the pipeline is missing."
        )

    from bot.venues.asknews import AskNewsClient
    from bot.venues.llm import OpenRouterClient

    models, research_model = _env_models()
    llm_client = OpenRouterClient(key)
    missing = _validate_roster(llm_client.models(), [*models, research_model])
    if missing:
        raise SetupError(
            f"model ids not on OpenRouter: {missing}\n"
            "Set METAC_FORECAST_MODELS / METAC_RESEARCH_MODEL in .env to valid ids."
        )

    personal_key = openrouter_personal_key()
    personal_client = OpenRouterClient(personal_key) if personal_key else None
    if personal_client is None:
        # Sent to the donated key, these runs would fail on every question: every
        # ensemble thinned, and real failures buried under expected ones. Dropping
        # them gives the same ensemble, said once at startup.
        dropped = [m for m in models if needs_personal_key(m)]
        models = tuple(m for m in models if not needs_personal_key(m))
        if dropped:
            print(
                f"note: OPENROUTER_PERSONAL_KEY absent — {dropped} dropped from the roster; "
                f"the ensemble runs {len(models)} model(s)."
            )
        if not models or needs_personal_key(research_model):
            raise SetupError(
                "this roster needs OPENROUTER_PERSONAL_KEY: the donated key refuses "
                f"{', '.join(PERSONAL_KEY_PROVIDERS)} models."
            )
    on_personal = sorted({m for m in models if needs_personal_key(m)})
    second_route = sorted({m for m in models if falls_back_to_personal_key(m)})
    print(
        f"roster: {', '.join(models)}"
        + (f" ({', '.join(on_personal)} on the personal key)" if on_personal else "")
        + (
            f"; {', '.join(second_route)} falls back to the personal key"
            if second_route and personal_client
            else ""
        )
    )

    def llm(model: str, prompt: str, *, temperature: float = 1.0) -> str:
        if model == "check":  # the extremeness check runs on the first roster model
            model = models[0]
        route = personal_client if personal_client and needs_personal_key(model) else llm_client
        return route.complete(model, prompt, temperature=temperature)

    asknews = None
    creds = asknews_credentials()
    if creds:
        asknews = AskNewsClient(*creds).search
    else:
        print("note: AskNews credentials absent — research runs on web search only.")

    live_client = MetaculusClient(token)
    client: TournamentClient = DryRunClient(live_client) if dry_run else live_client
    return Session(
        client=client,
        llm_client=llm_client,
        llm=llm,
        models=models,
        research_model=research_model,
        asknews_search=asknews,
        metaculus_client=live_client,
        personal_llm_client=personal_client,
        fallback_llm=personal_client.complete if personal_client else None,
    )


@dataclass
class SweepSummary:
    seen: int
    submitted: int
    failed: int
    cost_usd: float  # THIS sweep only — see sweep_and_log
    credits_remaining: float | None
    results: list[ForecastResult] = field(default_factory=list)


def _spend(session: Session) -> tuple[float, dict[str, float]]:
    """Running cost across both keys' clients: the total, and per model.

    Summed per model, not merged: a FALLBACK_TO_PERSONAL_KEY model is served
    by both clients, and a merge would keep only one key's share.
    """
    total = 0.0
    by_model: dict[str, float] = {}
    for llm_client in session.llm_clients():
        total += float(getattr(llm_client, "total_cost_usd", 0.0))
        for model, cost in dict(getattr(llm_client, "cost_by_model", {})).items():
            by_model[model] = by_model.get(model, 0.0) + float(cost)
    return total, by_model


def sweep_and_log(
    session: Session,
    tournament: str,
    *,
    dry_run: bool = False,
    resubmit: bool = False,
    limit: int | None = None,
) -> SweepSummary:
    """One sweep of a tournament, with its rows written and its summary printed.

    Cost is reported as the DELTA over this sweep, not the client's running
    total. Across a three-day poll the client is long-lived, so logging
    `total_cost_usd` on every sweep would write a cumulative figure into a
    per-run field — a number that looks like a cost and is a sum, which is
    exactly the class of quiet wrongness `data/metaculus/polls.jsonl` exists
    to make impossible to miss.
    """
    started = dt.datetime.now(dt.UTC).isoformat()
    cost_before, by_model_before = _spend(session)

    results = run_tournament(
        tournament,
        session.client,
        session.llm,
        session.models,
        session.research_model,
        asknews_search=session.asknews_search,
        # Re-read every sweep: the dedup set grows as this poll submits, and a
        # set captured once would re-forecast everything on the second sweep.
        already_done=previously_forecast_ids(),
        resubmit=resubmit,
        limit=limit,
        fallback=session.fallback_llm,
    )

    submitted = sum(1 for r in results if r.submitted)
    failed = [r for r in results if r.error and not r.error.startswith("skipped")]
    uncommented = [r for r in results if r.submitted and r.comment_error]
    for result in results:
        row = {
            "ts": started,
            "tournament": tournament,
            "question_id": result.question_id,
            "post_id": result.post_id,
            "title": result.title,
            "qtype": result.qtype,
            "submitted": result.submitted and not dry_run,
            "dry_run": dry_run,
            "forecast": result.forecast,
            "runs_ok": result.runs_ok,
            "runs_failed": result.runs_failed,
            "suspect": result.suspect,
            "run_errors": result.run_errors,
            "fallbacks": result.fallbacks,
            "error": result.error,
            "comment_error": result.comment_error,
        }
        if result.submitted or (result.error and not result.error.startswith("skipped")):
            append_jsonl(FORECAST_LOG, row)

    cost_now, by_model_now = _spend(session)
    sweep_cost = cost_now - cost_before
    sweep_by_model = {
        model: round(cost - by_model_before.get(model, 0.0), 4)
        for model, cost in by_model_now.items()
        if round(cost - by_model_before.get(model, 0.0), 4)
    }
    remaining = session.credits_remaining()
    append_jsonl(
        POLL_LOG,
        {
            "ts": started,
            "tournament": tournament,
            "questions_seen": len(results),
            "submitted": submitted,
            "failed": len(failed),
            # Counted apart from `failed`, and never folded into it: these
            # questions were answered. They are just not prize-eligible until
            # someone posts the reasoning by hand.
            "comment_failed": len(uncommented),
            "dry_run": dry_run,
            "credits_remaining": remaining,
            "personal_credits_remaining": session.personal_credits_remaining(),
            "llm_cost_usd": round(sweep_cost, 4),
            "llm_cost_cumulative_usd": round(cost_now, 4),
            "llm_cost_by_model": sweep_by_model,
        },
    )
    if sweep_cost:
        per_question = sweep_cost / max(submitted, 1)
        print(f"LLM spend this sweep: ${sweep_cost:.2f} (${per_question:.2f}/question submitted)")

    print(f"{tournament}: {len(results)} question(s) seen, {submitted} submitted.")
    for result in failed:
        print(f"  FAILED {result.question_id} {result.title[:60]}: {result.error}")
    # Louder than a thinned ensemble: no sweep will come back to these, because
    # the forecast that is already standing makes the next sweep skip them.
    for result in uncommented:
        print(f"  NO COMMENT {result.question_id} {result.title[:50]}: {result.comment_error}")
        print(f"      forecast IS in; post the reasoning by hand on post {result.post_id}")
    # A thinned ensemble is a quieter defect than a dead question and costs the
    # design's largest measured edge, so it is reported, not left to the log.
    for result in results:
        if result.submitted and result.runs_failed:
            total = result.runs_ok + result.runs_failed
            print(
                f"  THINNED {result.question_id} {result.title[:50]}: {result.runs_ok}/{total} runs"
            )
            for reason in result.run_errors:
                print(f"      {reason}")
    # Whole ensembles, but only because the second route answered. Quieter
    # than THINNED and printed all the same: without it a slot that fails
    # every time on the donated key looks exactly like a healthy one.
    for result in results:
        if result.submitted and result.fallbacks:
            print(f"  FELL BACK {result.question_id} {result.title[:50]}: personal key answered")
            for reason in result.fallbacks:
                print(f"      donated key: {reason}")
    return SweepSummary(
        seen=len(results),
        submitted=submitted,
        failed=len(failed),
        cost_usd=sweep_cost,
        credits_remaining=remaining,
        results=results,
    )


def use_utf8_stdio() -> None:
    """Windows consoles default to cp1252; a '≥' in a question title crashed
    the failure-summary printer at the end of the first live rehearsal —
    a cosmetic encoding bug masking a 9/11 successful run."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tournament", default="minibench")
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not submit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="ignore existing forecasts (bot-testing-area ONLY; tournaments want one forecast)",
    )
    args = parser.parse_args(argv)

    if args.resubmit and args.tournament != "bot-testing-area":
        print("--resubmit is allowed only with --tournament bot-testing-area.")
        return 2
    try:
        session = build_session(dry_run=args.dry_run)
    except SetupError as exc:
        print(str(exc))
        return 2

    try:
        summary = sweep_and_log(
            session,
            args.tournament,
            dry_run=args.dry_run,
            resubmit=args.resubmit,
            limit=args.limit,
        )
    finally:
        session.close()
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
