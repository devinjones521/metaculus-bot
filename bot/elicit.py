"""Prompts and parsers: what the forecast models are asked, and how answers are read.

Pure module — no network. Every prompt encodes a failure mode this project or the
field has already paid for, with the receipts in `docs/METHOD.md`:

  - the "already resolved" trap (bots submitting 99% on open questions) gets an
    explicit guard sentence, because it is the single most catastrophic known
    error under a log score;
  - the status-quo outcome is asked for by name — the world changes more slowly
    than headlines imply, and Metaculus's own template treats this as
    load-bearing;
  - an explicit base-rate step, the strongest within-winners strategy from the
    Fall 2025 survey (r = +0.38) after tail-capping;
  - units and today's date appear verbatim, because "insufficient prompt details
    such as units and assumed start date" is a named cause of bad forecasts;
  - the word "Bayesian" appears nowhere: prompts titled that way measurably
    underperformed in both Metaculus prompt-optimization studies.

Parsers are strict where wrongness is unrecoverable (missing options, decreasing
percentile values → ParseError, run dropped) and lenient where formatting noise
is harmless (commas, currency signs, bold markers).
"""

from __future__ import annotations

import re

from bot.venues.metaculus import Question

# Elicited percentiles for continuous questions. Wider than the template's
# 10–90 because model distributions run too narrow, and the tails are where a
# log score hurts.
NUMERIC_PERCENTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class ParseError(ValueError):
    """The model's answer could not be read as a forecast. The run is dropped."""


# ---------------------------------------------------------------- prompts --


def _question_block(question: Question, today: str) -> str:
    lines = [
        f"Today's date: {today}.",
        f"The question closes for forecasting on: {question.close_time or 'unknown'}.",
        "",
        f"QUESTION: {question.title}",
        "",
        f"BACKGROUND:\n{question.description or '(none provided)'}",
        "",
        f"RESOLUTION CRITERIA:\n{question.resolution_criteria or '(none provided)'}",
        "",
        f"FINE PRINT:\n{question.fine_print or '(none)'}",
    ]
    if question.unit:
        lines.append(f"\nUNIT for all numeric answers: {question.unit}")
    return "\n".join(lines)


GUARDS = """IMPORTANT DISCIPLINE:
- This question has NOT yet resolved. If your research suggests it already
  resolved, treat that as a signal to re-read the resolution criteria and dates
  carefully — most such impressions come from misreading an adjacent event.
- Weight the status quo heavily: the world changes more slowly than news
  coverage implies, and unresolved situations usually persist to the deadline.
- State an explicit base rate: how often do events of this reference class
  happen in a comparable time window? Start from that number, then adjust.
- Good forecasters are humble about tails: 97% means a 1-in-33 error rate
  across your career of such statements. Reserve extremes for questions where
  the outcome is nearly mechanical.
- Use ONLY the research provided plus general knowledge; your training data
  ends before today, so recent developments come from the research section."""


def binary_prompt(question: Question, research: str, today: str) -> str:
    return f"""You are a careful professional forecaster in a scored tournament.

{_question_block(question, today)}

YOUR RESEARCH ASSISTANTS REPORT:
{research}

{GUARDS}

Before answering, write short sections for:
(a) time remaining until the outcome is known
(b) the status-quo outcome if nothing changes from today
(c) the base rate for this reference class of events
(d) the strongest scenario for NO
(e) the strongest scenario for YES

The final line of your answer must be exactly:
Probability: ZZ.Z%
"""


def multiple_choice_prompt(question: Question, research: str, today: str) -> str:
    option_lines = "\n".join(f"- {option}" for option in question.options)
    example = "\n".join(f"{option}: NN%" for option in question.options)
    return f"""You are a careful professional forecaster in a scored tournament.

{_question_block(question, today)}

THE ONLY POSSIBLE OPTIONS ARE:
{option_lines}

YOUR RESEARCH ASSISTANTS REPORT:
{research}

{GUARDS}
- Leave a little probability on every option — surprises happen, and a log
  score punishes a zero on the realised option without bound.

Before answering, write short sections for:
(a) time remaining until the outcome is known
(b) the status-quo outcome if nothing changes from today
(c) the base rate or natural frequency for each option
(d) what news would most change this distribution

Finish with one line per option, in the exact order given, formatted exactly:
{example}
"""


def numeric_prompt(question: Question, research: str, today: str) -> str:
    bounds = []
    if question.range_min is not None and question.open_lower_bound is False:
        bounds.append(f"The outcome cannot be below {question.range_min}.")
    if question.range_max is not None and question.open_upper_bound is False:
        bounds.append(f"The outcome cannot be above {question.range_max}.")
    if question.range_min is not None and question.open_lower_bound:
        bounds.append(f"Values below {question.range_min} are possible but off-scale.")
    if question.range_max is not None and question.open_upper_bound:
        bounds.append(f"Values above {question.range_max} are possible but off-scale.")
    bound_text = "\n".join(bounds) or "(no explicit bounds provided)"
    percentile_lines = "\n".join(f"Percentile {int(p * 100)}: XX" for p in NUMERIC_PERCENTILES)
    return f"""You are a careful professional forecaster in a scored tournament.

{_question_block(question, today)}

RANGE INFORMATION:
{bound_text}

YOUR RESEARCH ASSISTANTS REPORT:
{research}

{GUARDS}
- Forecasters' distributions are almost always TOO NARROW. Set your 5th and
  95th percentiles wide enough to be genuinely surprised if breached — think
  about data revisions, definitional quirks and unlikely-but-real shocks.

Before answering, write short sections for:
(a) time remaining until the outcome is known
(b) the current value / status-quo outcome, with its date
(c) the recent trend and its plausible continuation
(d) a low scenario and a high scenario

Finish with exactly these lines (numbers only, in {question.unit or "the question's unit"},
no commas in numbers, no ranges, one number per line):
{percentile_lines}
"""


def research_prompt(question: Question, today: str) -> str:
    return f"""Today is {today}. You are a research assistant with web access,
briefing a forecaster on this question:

{_question_block(question, today)}

Search the web for the latest relevant information. Produce a compact briefing:
- the current state of play, with dates on every fact;
- the key numbers (latest values, official figures) with their sources;
- what has changed in the last two weeks;
- any scheduled events before the close date that could decide the outcome;
- what the resolution source currently shows, if you can access it.

Cite the publication date of everything. If you find little or nothing, say so
plainly — do NOT fill the gap with speculation, and never invent a fact.
End with a line 'REMAINING GAPS:' listing what you could not establish."""


def gap_prompt(question: Question, briefing: str, today: str) -> str:
    return f"""Today is {today}. A first research pass on the question below left gaps.

QUESTION: {question.title}
RESOLUTION CRITERIA: {question.resolution_criteria or "(none)"}

FIRST BRIEFING:
{briefing}

Search the web specifically to close the REMAINING GAPS listed above (and any
obvious omissions). Report only NEW findings, with dates and sources. If a gap
cannot be closed, state that explicitly."""


def extreme_check_prompt(question: Question, probability: float, today: str) -> str:
    return f"""Today is {today}. A forecasting ensemble has produced an extreme
probability of {probability:.0%} for this question:

{_question_block(question, today)}

Extreme forecasts are where bots have historically lost the most points, usually
by (1) believing the question already resolved when it has not, (2) misreading
the resolution criteria or its dates, or (3) a unit or magnitude error.

Answer with exactly one word on the first line: either SOUND (the extreme
probability is justified and the criteria were read correctly) or SUSPECT
(there is a live risk of one of the failure modes above), then one short
paragraph of justification."""


# ---------------------------------------------------------------- parsers --

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def _clean_number(raw: str) -> float:
    return float(raw.replace(",", "").replace("$", "").strip())


def parse_binary(text: str) -> float:
    """The LAST 'Probability: NN%' (or bare NN%) in the text, as [0,1]."""
    matches = re.findall(rf"[Pp]robability:?\s*\*{{0,2}}\s*({_NUMBER})\s*%", text)
    if not matches:
        matches = re.findall(rf"({_NUMBER})\s*%", text)
    if not matches:
        raise ParseError("no probability found in binary answer")
    value = _clean_number(matches[-1]) / 100.0
    if not (0.0 <= value <= 1.0):
        raise ParseError(f"probability {value} outside [0,1]")
    return value


def parse_multiple_choice(text: str, options: tuple[str, ...]) -> dict[str, float]:
    """One probability per option, keyed exactly by the given labels.

    Every option must appear; a missing option means the model answered a
    different question, and normalising around the hole would hide that.
    """
    out: dict[str, float] = {}
    for option in options:
        pattern = rf"{re.escape(option)}\s*\*{{0,2}}:?\s*\*{{0,2}}\s*({_NUMBER})\s*%"
        matches = re.findall(pattern, text)
        if not matches:
            raise ParseError(f"option {option!r} missing from answer")
        out[option] = _clean_number(matches[-1]) / 100.0
    total = sum(out.values())
    if total <= 0:
        raise ParseError("all option probabilities zero")
    if any(not (0.0 <= v <= 1.0) for v in out.values()):
        raise ParseError(f"option probability outside [0,1]: {out}")
    return {k: v / total for k, v in out.items()}


def parse_numeric(text: str) -> dict[float, float]:
    """'Percentile NN: value' lines → {0.05: v, ...}. Values must not decrease."""
    out: dict[float, float] = {}
    for match in re.finditer(rf"[Pp]ercentile\s+(\d+)\s*:?\s*\*{{0,2}}\s*({_NUMBER})", text):
        pct = int(match.group(1)) / 100.0
        out[pct] = _clean_number(match.group(2))
    missing = [p for p in NUMERIC_PERCENTILES if p not in out]
    if missing:
        raise ParseError(f"missing percentiles: {missing}")
    out = {p: out[p] for p in NUMERIC_PERCENTILES}
    values = list(out.values())
    if any(later < earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ParseError(f"percentile values decrease: {values}")
    return out


def parse_extreme_check(text: str) -> bool:
    """True = SUSPECT (pull the forecast in from the tail)."""
    first = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
    return first.startswith("SUSPECT")
